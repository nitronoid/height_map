#include <iostream>
#include <stdlib.h>
#include <cusp/array2d.h>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <cusparse_v2.h>

#include "stbi_raii.hpp"
#include <cub/cub.cuh>
#include <cusp/transpose.h>
#include <cusp/gallery/grid.h>
#include <cusp/gallery/poisson.h>
#include <cusp/print.h>
#include <cusp/convert.h>
#include <cusp/relaxation/sor.h>
#include <cusp/relaxation/jacobi.h>
#include <cusp/krylov/cg.h>
#include <cusp/krylov/bicgstab.h>
#include <cusp/linear_operator.h>
#include <cusp/precond/diagonal.h>
#include <cusp/monitor.h>
#include <cusp/io/matrix_market.h>

#include "matrix_functional.cuh"
#include "zip_it.cuh"
#include "cycle_iterator.cuh"
using real = float;

void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(
      stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
    {
      exit(code);
    }
  }
}

void gpuErrchk(cudaError_t ans)
{
  gpuAssert((ans), __FILE__, __LINE__);
}
static const char* _cusparseGetErrorEnum(cusparseStatus_t error)
{
  switch (error)
  {
  case CUSPARSE_STATUS_SUCCESS: return "CUSPARSE_STATUS_SUCCESS";

  case CUSPARSE_STATUS_NOT_INITIALIZED:
    return "CUSPARSE_STATUS_NOT_INITIALIZED";

  case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED";

  case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";

  case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH";

  case CUSPARSE_STATUS_MAPPING_ERROR: return "CUSPARSE_STATUS_MAPPING_ERROR";

  case CUSPARSE_STATUS_EXECUTION_FAILED:
    return "CUSPARSE_STATUS_EXECUTION_FAILED";

  case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR";

  case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

  case CUSPARSE_STATUS_ZERO_PIVOT: return "CUSPARSE_STATUS_ZERO_PIVOT";
  }

  return "<unknown>";
}
inline void
__cusparseSafeCall(cusparseStatus_t err, const char* file, const int line)
{
  if (CUSPARSE_STATUS_SUCCESS != err)
  {
    fprintf(stderr,
            "CUSPARSE error in file '%s', line %d, error %s\nterminating!\n",
            __FILE__,
            __LINE__,
            _cusparseGetErrorEnum(err));
    assert(0);
  }
}

extern "C" void cusparseSafeCall(cusparseStatus_t err)
{
  __cusparseSafeCall(err, __FILE__, __LINE__);
}

template <typename T>
constexpr T sqr(T val) noexcept
{
  return val * val;
}

template <typename T>
void strided_copy(const T* i_src,
                  T* i_dest,
                  int src_stride,
                  int dest_stride,
                  int n,
                  cudaMemcpyKind i_kind)
{
  cudaMemcpy2D(i_dest,
               sizeof(T) * dest_stride,
               i_src,
               sizeof(T) * src_stride,
               sizeof(T),
               n,
               i_kind);
}

void make_device_image(gsl::not_null<const real*> h_image,
                       cusp::array2d<real, cusp::device_memory>::view d_image)
{
  const int npixels = d_image.num_cols;
  const int nchannels = d_image.num_rows;
  for (int c = 0; c < nchannels; ++c)
  {
    auto d_image_channel = d_image.values.begin().base().get() + npixels * c;
    const auto h_image_channel = h_image.get() + c;
    strided_copy(h_image_channel,
                 d_image_channel,
                 nchannels,
                 1,
                 npixels,
                 cudaMemcpyHostToDevice);
  }
}

void make_host_image(cusp::array2d<real, cusp::device_memory>::view d_image,
                     gsl::not_null<real*> h_image)
{
  const int npixels = d_image.num_cols;
  const int nchannels = d_image.num_rows;
  for (int c = 0; c < nchannels; ++c)
  {
    auto d_image_channel = d_image.values.begin().base().get() + npixels * c;
    const auto h_image_channel = h_image.get() + c;
    strided_copy(d_image_channel,
                 h_image_channel,
                 1,
                 nchannels,
                 npixels,
                 cudaMemcpyDeviceToHost);
  }
}

cusp::csr_matrix<int, real, cusp::device_memory>
cusparse_add(cusp::coo_matrix<int, real, cusp::device_memory>::const_view di_A,
             cusp::coo_matrix<int, real, cusp::device_memory>::const_view di_B)
{
  cusp::array1d<int, cusp::device_memory> A_row_offsets(di_A.num_rows + 1);
  cusp::indices_to_offsets(di_A.row_indices, A_row_offsets);
  cusp::array1d<int, cusp::device_memory> B_row_offsets(di_B.num_rows + 1);
  cusp::indices_to_offsets(di_B.row_indices, B_row_offsets);

  cusparseHandle_t handle;
  cusparseSafeCall(cusparseCreate(&handle));

  cusparseMatDescr_t A_description;
  cusparseSafeCall(cusparseCreateMatDescr(&A_description));
  cusparseMatDescr_t B_description;
  cusparseSafeCall(cusparseCreateMatDescr(&B_description));
  cusparseMatDescr_t C_description;
  cusparseSafeCall(cusparseCreateMatDescr(&C_description));

  // Coefficients
  const real alpha = 1.f;
  const real beta = 0.1f * 2.f;

  int C_base, C_nnz;
  // Not sure if this is needed
  int* nnz_total = &C_nnz;
  cusparseSafeCall(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
  cusp::array1d<int, cusp::device_memory> C_row_offsets(di_A.num_rows + 1);

  cusparseSafeCall(cusparseXcsrgeamNnz(handle,
                                       di_A.num_rows,
                                       di_A.num_cols,
                                       A_description,
                                       di_A.num_entries,
                                       A_row_offsets.begin().base().get(),
                                       di_A.column_indices.begin().base().get(),
                                       B_description,
                                       di_B.num_entries,
                                       B_row_offsets.begin().base().get(),
                                       di_B.column_indices.begin().base().get(),
                                       C_description,
                                       C_row_offsets.begin().base().get(),
                                       nnz_total));
  if (nnz_total != NULL)
  {
    C_nnz = *nnz_total;
  }
  else
  {
    C_nnz = C_row_offsets.back();
    C_nnz -= C_row_offsets[0];
  }

  cusp::csr_matrix<int, real, cusp::device_memory> do_C(
    di_A.num_rows, di_A.num_cols, C_nnz);
  do_C.row_offsets = std::move(C_row_offsets);

  // Now actually do the add
  cusparseSafeCall(cusparseScsrgeam(handle,
                                    di_A.num_rows,
                                    di_A.num_cols,
                                    &alpha,
                                    A_description,
                                    di_A.num_entries,
                                    di_A.values.begin().base().get(),
                                    A_row_offsets.begin().base().get(),
                                    di_A.column_indices.begin().base().get(),
                                    &beta,
                                    B_description,
                                    di_B.num_entries,
                                    di_B.values.begin().base().get(),
                                    B_row_offsets.begin().base().get(),
                                    di_B.column_indices.begin().base().get(),
                                    C_description,
                                    do_C.values.begin().base().get(),
                                    do_C.row_offsets.begin().base().get(),
                                    do_C.column_indices.begin().base().get()));
  return do_C;
}

void check_symmetry(
  cusp::csr_matrix<int, real, cusp::device_memory>::const_view di_M)
{
  // Copy to host
  cusp::csr_matrix<int, real, cusp::host_memory> M = di_M;
  // Transpose
  cusp::csr_matrix<int, real, cusp::host_memory> MT(
    M.num_cols, M.num_rows, M.num_entries);
  cusp::transpose(M, MT);

  printf("Checking for symmetry\n");

  for (int i = 0; i < di_M.num_entries; ++i)
  {
    const real value = M.values[i];
    const real valueT = MT.values[i];
    if (value != valueT)
    {
      printf("BAD symmetry at: %d with value: %f and value^T: %f\n",
             i,
             value,
             valueT);
    }
  }
}

void build_M(float3 L,
             cusp::coo_matrix<int, real, cusp::device_memory>::view do_M)
{
  const int nnormals = do_M.num_rows / 3;
  // Outer product of the lighting direction
  // clang-format off
  real LLT[9] = {L.x * L.x, L.x * L.y, L.x * L.z,
                 L.x * L.y, L.y * L.y, L.y * L.z,
                 L.x * L.z, L.y * L.z, L.z * L.z};
  printf("LLT:\n [%f, %f, %f,\n %f, %f, %f,\n %f, %f, %f]\n", 
      LLT[0], LLT[1], LLT[2],
      LLT[3], LLT[4], LLT[5],
      LLT[6], LLT[7], LLT[8]);
  // clang-format on
  // Copy to the device
  thrust::device_vector<real> d_LLT(LLT, LLT + 9);
  // Perform a kronecker product of LLT with the Identity matrix
  // We want to iterate over each row of LLT, n times where n is the number of
  // normals
  const auto LLT_row = detail::make_row_iterator(nnormals * 3);
  // We want to iterate over each column of LLT, in a repeating cycle for each n
  const auto LLT_col = detail::make_column_iterator(3);
  // Now we can combine the two
  const auto LLT_i = thrust::make_transform_iterator(
    detail::zip_it(LLT_row, LLT_col),
    [=] __host__ __device__(const thrust::tuple<int, int>& coord) {
      return coord.get<0>() * 3 + coord.get<1>();
    });
  // Use the look up index to get the real value from LLT
  const auto LLT_v = thrust::make_permutation_iterator(d_LLT.begin(), LLT_i);
  // Copy the values across to M
  thrust::copy_n(LLT_v, nnormals * 9, do_M.values.begin());
  // The row keys will be i / 3, as we only have 3 values per row and column
  const auto count = thrust::make_counting_iterator(0);
  thrust::transform(count,
                    count + nnormals * 9,
                    do_M.row_indices.begin(),
                    detail::unary_divides<int>(3));
  // To write the column keys we need a repeating sequence of 0, 1, 2 * n to
  // give 0, n, 2n, and then we offset by the row % n
  thrust::transform(LLT_col,
                    LLT_col + nnormals * 9,
                    do_M.row_indices.begin(),
                    do_M.column_indices.begin(),
                    [=] __host__ __device__(int s, int r) {
                      return (r % nnormals) + s * nnormals;
                    });

  using tup3 = thrust::tuple<int, int, real>;
  const auto inc_diag = [=] __host__ __device__(tup3 entry) {
    // Add one to the diagonals
    if (entry.get<0>() == entry.get<1>())
    {
      entry.get<2>() += 1;
    }
    return entry;
  };
  auto entry_it = detail::zip_it(
    do_M.row_indices.begin(), do_M.column_indices.begin(), do_M.values.begin());
  // Fix the boundary cell diagonals
  // thrust::transform(entry_it, entry_it + nnormals*3, entry_it, inc_diag);
}

void build_B(const int m,
             const int n,
             cusp::coo_matrix<int, real, cusp::device_memory>::view do_B)
{
  const int nsb = do_B.num_entries / 3;
  const int nnormals = m * n;
  auto entry_it = detail::zip_it(
    do_B.row_indices.begin(), do_B.column_indices.begin(), do_B.values.begin());
  // Build the discrete Poisson problem matrix
  cusp::coo_matrix<int, real, cusp::device_memory> d_temp;
  cusp::gallery::poisson5pt(d_temp, n, m);
  const auto temp_begin = detail::zip_it(d_temp.row_indices.begin(),
                                         d_temp.column_indices.begin(),
                                         d_temp.values.begin());
  thrust::copy_n(temp_begin, nsb, entry_it);

  using tup3 = thrust::tuple<int, int, real>;
  const auto fix_bnds = [=] __host__ __device__(tup3 entry) {
    // Fix boundary cell diagonals
    if (entry.get<0>() == entry.get<1>())
    {
      const int r = entry.get<0>() / n;
      const int c = entry.get<0>() % n;
      // If we're in a boundary cell we subtract one from the valence
      entry.get<2>() -= (r == 0 || r == (m - 1));
      entry.get<2>() -= (c == 0 || c == (n - 1));
    }
    return entry;
  };
  // Fix the boundary cell diagonals
  thrust::transform(entry_it, entry_it + nsb, entry_it, fix_bnds);

  // Correct the boundaries which don't have valence of 4 and copy the
  // corrected B for each channel of the normal vectors (3 times).
  // Tuple of [Row, Column, Value]
  auto entry_s = detail::make_cycle_iterator(entry_it, nsb);

  // Copy sB 3 times, offsetting by it's width and height for each new copy
  const auto op = [=] __host__ __device__(tup3 entry, int count) {
    // Work out what channel we're in
    const int channel = count / nsb;
    // Offset for the channel
    entry.get<0>() += channel * nnormals;
    entry.get<1>() += channel * nnormals;
    return entry;
  };
  const auto count = thrust::make_counting_iterator(nsb);
  thrust::transform(entry_s, entry_s + nsb * 2, count, entry_it + nsb, op);
}

template <typename T>
constexpr __host__ __device__ real signum(T val)
{
  return (T(0) <= val) - (val < T(0));
}

template <typename T>
constexpr __host__ __device__ T clamp(const T& n,
                                      const T& lower,
                                      const T& upper)
{
  return max(lower, min(n, upper));
}
template <typename T>
constexpr __host__ __device__ T iclamp(const T& n, const T& e)
{
  return max(e, std::abs(n)) * signum(n);
}

struct relative_height_from_normals
{
  using vec2 = thrust::tuple<real, real>;

  __host__ __device__ real dot(const vec2& n1, const vec2& n2) const noexcept
  {
    return n1.get<0>() * n2.get<0>() + n1.get<1>() * n2.get<1>();
  }

  __host__ __device__ real det(const vec2& n1, const vec2& n2) const noexcept
  {
    return n1.get<0>() * n2.get<1>() - n1.get<1>() * n2.get<0>();
  }

  __host__ __device__ vec2 normalize(const vec2& n) const noexcept
  {
    const auto norm = std::sqrt(dot(n, n));
    return thrust::make_tuple(n.get<0>() / norm, n.get<1>() / norm);
  }

  __host__ __device__ real operator()(vec2 n1,
                                      vec2 n2,
                                      bool debug = false) const noexcept
  {
    // Normalize n1 and n2
    n1 = normalize(n1);
    n2 = normalize(n2);
    const real x = n1.get<0>() - n2.get<0>();
    const real y = n1.get<1>() - n2.get<1>();

    real q;
    constexpr float epsilon = 0.0000001f;
    if (std::abs(x) > epsilon)
    {
      q = y / x;
    }
    else
    {
      const auto inf = std::numeric_limits<real>::infinity();
      const real g1 =
        n1.get<0>() == 0.f ? inf : n1.get<1>() / n1.get<0>();
      if (g1 == inf)
        q = 0.f;
      else if (g1 == 0.f)
        q = 1.f / epsilon;
      else
        q = 1.f / g1;
    }

    return q;
  }
};

void normalize(cusp::array1d<real, cusp::device_memory>::view dio_v)
{
  // Subtract the minimum value
  const real min = *thrust::min_element(dio_v.begin(), dio_v.end());
  const detail::unary_minus<real> subf(min);
  thrust::transform(dio_v.begin(), dio_v.end(), dio_v.begin(), subf);
  // Divide by the maximum value
  const real scale = 1.f / *thrust::max_element(dio_v.begin(), dio_v.end());
  const detail::unary_multiplies<real> mulf(scale);
  thrust::transform(dio_v.begin(), dio_v.end(), dio_v.begin(), mulf);
}

void print_range_avg(cusp::array1d<real, cusp::device_memory>::const_view di_v)
{
  const real min = *thrust::min_element(di_v.begin(), di_v.end());
  const real max = *thrust::max_element(di_v.begin(), di_v.end());
  const real avg = thrust::reduce(di_v.begin(), di_v.end()) / di_v.size();
  std::cout << "min: " << min << ", max: " << max << ", avg: " << avg << '\n';
}

void build_Q_values(cusp::array2d<real, cusp::device_memory>::view di_normals,
                    cusp::coo_matrix<int, real, cusp::device_memory>::view do_Q)
{
  // Iterate over the normals with their index
  const auto count = thrust::make_counting_iterator(0);
  const auto normal_begin = detail::zip_it(di_normals.row(0).begin(),
                                           di_normals.row(1).begin(),
                                           di_normals.row(2).begin(),
                                           count);
  // Iterate over pairs of normals using the matrix coordinates
  const auto n1_begin =
    thrust::make_permutation_iterator(normal_begin, do_Q.row_indices.begin());
  const auto n2_begin = thrust::make_permutation_iterator(
    normal_begin, do_Q.column_indices.begin());
  const auto n1_end = n1_begin + do_Q.num_entries;

  using vec = thrust::tuple<real, real, real, int>;
  thrust::transform(n1_begin,
                    n1_end,
                    n2_begin,
                    do_Q.values.begin(),
                    [] __host__ __device__(const vec& i_n1, const vec& i_n2) {
                      // Check whether these normals are vertical or horizontal
                      // neighbors and project the normals accordingly
                      auto n1 = thrust::make_tuple(0.f, i_n1.get<2>());
                      auto n2 = thrust::make_tuple(0.f, i_n2.get<2>());
                      if (std::abs(i_n1.get<3>() - i_n2.get<3>()) == 1)
                      {
                        n1.get<0>() = i_n1.get<0>();
                        n2.get<0>() = i_n2.get<0>();
                      }
                      else
                      {
                        n1.get<0>() = i_n1.get<1>();
                        n2.get<0>() = i_n2.get<1>();
                      }
                      // in lower triangle
                      const bool lower = i_n1.get<3>() > i_n2.get<3>();
                      const real q = relative_height_from_normals{}(n1, n2);
                      return lower ? -q : q;
                    });
}

void apply_sor(
    cusp::csr_matrix<int, real, cusp::device_memory>::const_view di_A,
    cusp::array1d<real, cusp::device_memory>::const_view di_b,
    cusp::array1d<real, cusp::device_memory>::view do_x,
    const real i_w,
    const real i_tol,
    const int i_max_iter,
    const bool verbose)
{
  // Linear SOR operator
  cusp::relaxation::sor<real, cusp::device_memory> M(di_A, i_w);
  // Array to store the residual
  cusp::array1d<real, cusp::device_memory> d_r(di_b.size());
  // Compute the initial residual
  const auto compute_residual = [&] __host__ {
    cusp::multiply(di_A, do_x, d_r);
    cusp::blas::axpy(di_b, d_r, -1.f);
  };
  compute_residual();
  // Monitor the convergence
  cusp::monitor<real> monitor(di_b, i_max_iter, i_tol, 0, verbose);
  // Iterate until convergence criteria is met
  for (; !monitor.finished(d_r); ++monitor)
  {
    // Apply the SOR linear operator to iterate on our solution
    M(di_A, di_b, do_x);
    // Compute the residual
    compute_residual();
  }
}

int main(int argc, char* argv[])
{
  assert(argc >= 5);
  auto h_image = stbi::loadf(argv[1], 3);
  printf("Loaded image with dim: %dx%dx%d\n",
         h_image.width(),
         h_image.height(),
         h_image.n_channels());

  const real azimuth = std::stof(argv[2]) * M_PI / 180.0f;
  const real polar = std::stof(argv[3]) * M_PI / 180.0f;
  // Lighting direction
  float3 L{std::stof(argv[2]), std::stof(argv[3]), std::stof(argv[4])};
  // float3 L{std::sin(polar) * std::cos(azimuth),
  //         std::sin(polar) * std::sin(azimuth),
  //         std::cos(polar)};
  const real L_rlen = 1.f / std::sqrt(L.x * L.x + L.y * L.y + L.z * L.z);
  L.x *= L_rlen;
  L.y *= L_rlen;
  L.z *= L_rlen;
  printf("L: [%f, %f, %f]\n", L.x, L.y, L.z);

  cusp::array2d<real, cusp::device_memory> d_image(h_image.n_channels(),
                                                   h_image.n_pixels());
  make_device_image(h_image.get(), d_image);
  auto d_intensity = d_image.row(0);
  cusp::blas::scal(d_intensity, 2.f);
  // cusp::io::read_matrix_market_file(d_intensity, "shading.mtx");
  print_range_avg(d_intensity);
  // normalize(d_intensity);
  print_range_avg(d_intensity);

  const int width = h_image.width();
  const int height = h_image.height();
  const int nnormals = width * height;
  printf("Num pixels: %d rows * %d cols = %d\n", height, width, nnormals);

  cusp::coo_matrix<int, real, cusp::device_memory> d_M(
    nnormals * 3, nnormals * 3, nnormals * 9);
  build_M(L, d_M);
  printf("M has been built %dx%d\n", d_M.num_rows, d_M.num_cols);

  // B is our pixel 4-neighborhood adjacency matrix
  cusp::coo_matrix<int, real, cusp::device_memory> d_B(
    nnormals * 3, nnormals * 3, 3 * (height * (5 * width - 2) - 2 * width));
  build_B(height, width, d_B);
  printf("B has been built %dx%d\n", d_B.num_rows, d_B.num_cols);

  // Now we build A using M and B
  // A = M + 8lmI -2lmB <=> A = M + 2lm(4I - B)
  // So we use cuSparse to compute alpha * M + beta * B, where beta is 2lm
  // Now we can add M
  auto d_A = cusparse_add(d_M, d_B);
  printf("A has been built %dx%d\n", d_A.num_rows, d_A.num_cols);
  // cusp::print(d_A.values.subarray(0, 10));
  check_symmetry(d_A);

  // The b vector of the system is (shading intensity * L), where L repeats
  // Copy L to the device
  thrust::device_vector<real> d_L(&L.x, (&L.x) + 3);
  // Iterate over one component of L per channel of the normals
  const auto cyclic_L = thrust::make_permutation_iterator(
    d_L.begin(), detail::make_row_iterator(nnormals));
  const thrust::multiplies<real> mul;
  // Loop over for each dimension of the normals
  const auto cyclic_i =
    detail::make_cycle_iterator(d_intensity.begin(), nnormals);
  // Allocate the b vector
  cusp::array1d<real, cusp::device_memory> d_b(nnormals * 3);
  // Write the b vector
  thrust::transform(
    cyclic_i, cyclic_i + nnormals * 3, cyclic_L, d_b.begin(), mul);
  printf("b has been built %dx%d\n", d_b.size(), 1);

  // Now we can solve for the relative normals via SOR
  cusp::array1d<real, cusp::device_memory> d_x(3 * nnormals, 1.f);
  thrust::tabulate(
    d_x.begin(), d_x.end(), [=] __host__ __device__(int x) -> real {
      return x >= nnormals * 2;
    });
  {
    apply_sor(d_A, d_b, d_x, 1.f, 1e-5f, 1500, true);
    // Normalize
    using vec3 = thrust::tuple<real, real, real>;
    const auto normalize_vec = [] __host__ __device__(vec3 normal) {
      const real rmag =
        1.f / std::sqrt(sqr(normal.get<0>()) + sqr(normal.get<1>()) +
                        sqr(normal.get<2>()));
      normal.get<0>() *= rmag;
      normal.get<1>() *= rmag;
      normal.get<2>() = std::abs(normal.get<2>()) * rmag;
      return normal;
    };
    // Normalize our resulting normals
    auto norm_begin = detail::zip_it(
      d_x.begin(), d_x.begin() + nnormals, d_x.begin() + nnormals * 2);
    auto norm_end = norm_begin + nnormals;
    thrust::transform(norm_begin, norm_end, norm_begin, normalize_vec);
  }
  printf("Done\n");
  auto d_initial_normals = cusp::make_array2d_view(
    3, nnormals, nnormals, cusp::make_array1d_view(d_x), cusp::row_major{});
  cusp::array2d<real, cusp::device_memory> normal_copy = d_initial_normals;
  thrust::transform(normal_copy.values.begin(),
                    normal_copy.values.end(),
                    normal_copy.values.begin(),
                    detail::unary_plus<real>(1.f));
  thrust::transform(normal_copy.values.begin(),
                    normal_copy.values.end(),
                    normal_copy.values.begin(),
                    detail::unary_multiplies<real>(0.5f));
  make_host_image(normal_copy, h_image.get());
  stbi::writef("initial_normals.png", h_image);

  // Now that we have relative normals, we calculate the relative heights
  cusp::coo_matrix<int, real, cusp::device_memory> d_Q(
    nnormals, nnormals, height * (4 * width - 2) - 2 * width);
  // Initialize a grid matrix using CUSP
  cusp::gallery::grid2d(d_Q, width, height);
  build_Q_values(d_initial_normals, d_Q);

  // Now we can assemble a poisson problem to solve the absolute heights
  cusp::array1d<real, cusp::device_memory> d_pb(nnormals);
  thrust::reduce_by_key(d_Q.row_indices.begin(),
                        d_Q.row_indices.end(),
                        d_Q.values.begin(),
                        thrust::make_discard_iterator(),
                        d_pb.begin());

  // The A matrix
  cusp::coo_matrix<int, real, cusp::device_memory> d_pA(
    nnormals, nnormals, height * (5 * width - 2) - 2 * width);
  cusp::gallery::poisson5pt(d_pA, width, height);
  auto pA_begin = detail::zip_it(
    d_pA.row_indices.begin(), d_pA.column_indices.begin(), d_pA.values.begin());
  using tup3 = thrust::tuple<int, int, real>;
  const auto fix_bnds = [=] __host__ __device__(tup3 entry) {
    // Fix boundary cell diagonals
    if (entry.get<0>() == entry.get<1>())
    {
      const int r = entry.get<0>() / width;
      const int c = entry.get<0>() % width;
      // If we're in a boundary cell we subtract one from the valence
      entry.get<2>() -= (r == 0 || r == (height - 1));
      entry.get<2>() -= (c == 0 || c == (width - 1));
    }
    return entry;
  };
  // Fix the boundary cell diagonals
  thrust::transform(pA_begin, pA_begin + d_pA.num_entries, pA_begin, fix_bnds);
  // To get a result we need to "pin" the solution by setting an arbitrary
  // value to some constant. I use the first height.
  // Make the first equation a trivial solution 1*h0 = x
  d_pA.values.begin()[0] = 1.f;
  d_pA.values.begin()[1] = 0.f;
  d_pA.values.begin()[2] = 0.f;
  // Need to replace any references to the final solution with constants in b
  d_pA.values.begin()[3] = 0.f;
  d_pA.values.begin()[4 * width - 2] = 0.f;
  d_pb[0] = 0.5f;
  d_pb.begin()[1] += d_pb[0];
  d_pb.begin()[width] += d_pb[0];

  cusp::array1d<real, cusp::device_memory> d_h(nnormals, 0.5f);
  d_h[0] = d_pb[0];
  {
    cusp::csr_matrix<int, real, cusp::device_memory> pA(
      d_pA.num_rows, d_pA.num_cols, d_pA.num_entries);
    cusp::indices_to_offsets(d_pA.row_indices, pA.row_offsets);
    pA.column_indices = d_pA.column_indices;
    pA.values = d_pA.values;
    apply_sor(pA, d_pb, d_h, 0.9f, 1e-4f, std::stoi(argv[5]), true);
  }
  printf("H0: %f, H1: %f, H2:%f, H4:%f, Q01: %f, Q10: %f, Q12: %f, Q14: %f\n",
         (real)d_h[0],
         (real)d_h[1],
         (real)d_h[2],
         (real)d_h[4],
         (real)d_Q.values[0],
         (real)d_Q.values[2],
         (real)d_Q.values[3],
         (real)d_Q.values[4]);
  print_range_avg(d_h);
  normalize(d_h);

  const auto h_out = detail::zip_it(d_h.begin(), d_h.begin(), d_h.begin());
  const auto rn_begin = detail::zip_it(d_initial_normals.row(0).begin(),
                                       d_initial_normals.row(1).begin(),
                                       d_initial_normals.row(2).begin());
  thrust::copy_n(h_out, nnormals, rn_begin);

  make_host_image(d_initial_normals, h_image.get());
  stbi::writef("height.png", h_image);
}

