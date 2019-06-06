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
  const real beta = 0.001f * 2.f;

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
  //thrust::transform(entry_it, entry_it + nnormals*3, entry_it, inc_diag);
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

  __host__ __device__ real operator()(vec2 n1, vec2 n2, bool debug = false) const
    noexcept
  {
    // Normalize n1 and n2
    n1 = normalize(n1);
    n2 = normalize(n2);
#if 0

    const real theta = std::acos(dot(n1, n2));
    const real gamma = (real(M_PI) - theta) * 0.5f;
    const real delta = std::abs(std::atan(n1.get<1>() / n1.get<0>()));
    const real alpha = gamma - delta;
    const real q = std::tan(alpha);

    return (n1.get<0>() > 0.f && n2.get<0>() < 0.f) ? -q : q;
#else
    const auto inf = std::numeric_limits<real>::infinity();

    const real g1 =
      n1.get<0>() == 0.f ? inf : std::abs(n1.get<1>() / n1.get<0>());
    const real g2 = 
      n2.get<0>() == 0.f ? inf : std::abs(n2.get<1>() / n2.get<0>());

    const real x = n1.get<0>() - n2.get<0>();
    const real y = n1.get<1>() - n2.get<1>();

    if (debug)
    {
      printf("N1: [%.15f, %.15f]\n", n1.get<0>(), n1.get<1>());
      printf("N2: [%.15f, %.15f]\n", n2.get<0>(), n2.get<1>());
      printf("G1: %.15f, G2: %.15f\n", g1, g2);
      printf("N: [%.15f, %.15f]\n", x, y);
      printf("N.x == 0: %d\n", x == 0.f);
    }

    real q;
    constexpr float epsilon = 0.00001f;
    if (std::abs(x) > epsilon)
    {
      if (debug)
        printf("standard\n");
      q = std::abs(y / x);
    }
    else
    {
      if (g1 == inf)
        q = 0.f;
      else if (g1 == 0.f)
        q = 0.f; //inf
      else
        q = 1.f / g1;
    }

    return g1 > g2 ? -q : q;
#endif
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
                      bool print = i_n1.get<3>() == 0 && i_n2.get<3>() == 1;
                      const real q = relative_height_from_normals{}(n1, n2, print);
                      // if ((i_n1.get<3>() == 1 || i_n1.get<3>() == 0) &&
                      // (i_n2.get<3>() == 1 || i_n2.get<3>() == 0))
                      //  printf("%d %d %f\n", i_n1.get<3>(), i_n2.get<3>(), q);
                      return q;
                    });
}

void build_poisson_b(
  const int m,
  const int n,
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view di_Q,
  cusp::array1d<real, cusp::device_memory>::view do_b)
{
  // Build I - M, essentially a Poisson, with 1 down the diagonal
  cusp::coo_matrix<int, real, cusp::device_memory> d_temp;
  cusp::gallery::poisson5pt(d_temp, n, m);
  auto temp_begin = detail::zip_it(d_temp.row_indices.begin(),
                                   d_temp.column_indices.begin(),
                                   d_temp.values.begin());
  using tup3 = thrust::tuple<int, int, real>;
  thrust::transform(temp_begin,
                    temp_begin + d_temp.num_entries,
                    temp_begin,
                    [=] __host__ __device__(tup3 entry) {
                      // Fix boundary cell diagonals
                      entry.get<2>() =
                        (entry.get<1>() == entry.get<1>()) ? 1.f : -1.f;
                      return entry;
                    });
  // SpMv (I - M) * O, where O is the vector of 0.5's
  cusp::array1d<real, cusp::device_memory> d_O(do_b.size(), 0.5f);
  cusp::multiply(d_temp, d_O, do_b);
  // SpMv Q * ((I - M) * O)
  thrust::copy(do_b.begin(), do_b.begin(), d_O.begin());
  cusp::multiply(di_Q, d_O, do_b);
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
  // thrust::transform(d_image.values.begin(),
  //                  d_image.values.end(),
  //                  d_image.values.begin(),
  //                  [] __host__ __device__(real x) {
  //                    return min(max(x, 1.f / 255.f), 254.f / 255.f);
  //                  });
  //auto d_shading_intensity = d_image.row(0);
  cusp::array1d<real, cusp::device_memory> d_shading_intensity(
    h_image.n_pixels());
  cusp::io::read_matrix_market_file(d_shading_intensity, "shading.mtx");
  cusp::print(d_shading_intensity.subarray(0, 9));
  print_range_avg(d_shading_intensity);
  // normalize(d_shading_intensity);
  print_range_avg(d_shading_intensity);

  const int width = h_image.width();
  const int height = h_image.height();
  const int nnormals = width * height;
  printf("Num pixels: %d rows * %d cols = %d\n", height, width, nnormals);

  cusp::coo_matrix<int, real, cusp::device_memory> d_M(
    nnormals * 3, nnormals * 3, nnormals * 9);
  build_M(L, d_M);
  printf("M has been built %dx%d\n", d_M.num_rows, d_M.num_cols);
  //cusp::print(d_M.values.subarray(0, 10));
  //cusp::print(d_M.row_indices.subarray(0, 10));
  //cusp::print(d_M.column_indices.subarray(0, 10));

  // B is our pixel 4-neighborhood adjacency matrix
  cusp::coo_matrix<int, real, cusp::device_memory> d_B(
    nnormals * 3, nnormals * 3, 3 * (height * (5 * width - 2) - 2 * width));
  build_B(height, width, d_B);
  printf("B has been built %dx%d\n", d_B.num_rows, d_B.num_cols);
  //cusp::print(d_B.row_indices.subarray(0, 10));
  //cusp::print(d_B.column_indices.subarray(0, 10));
  // cusp::print(d_B);

  // Now we build A using M and B
  // A = M + 8lmI -2lmB <=> A = M + 2lm(4I - B)
  // So we use cuSparse to compute alpha * M + beta * B, where beta is 2lm
  // Now we can add M
  auto d_A = cusparse_add(d_M, d_B);
  printf("A has been built %dx%d\n", d_A.num_rows, d_A.num_cols);
  //cusp::print(d_A.values.subarray(0, 10));
  check_symmetry(d_A);
  //cusp::print(d_A.row_offsets.subarray(0, 10));
  //cusp::print(d_A.column_indices.subarray(0, 10));

  // The b vector of the system is (shading intensity * L), where L repeats
  // Copy L to the device
  thrust::device_vector<real> d_L(&L.x, (&L.x) + 3);
  // Iterate over one component of L per channel of the normals
  const auto cyclic_L = thrust::make_permutation_iterator(
    d_L.begin(), detail::make_row_iterator(nnormals));
  // Iterate over the shading intensity, cycling back for each channel
  const auto cyclic_Si =
    detail::make_cycle_iterator(d_shading_intensity.begin(), nnormals);
  // Allocate the b vector
  cusp::array1d<real, cusp::device_memory> d_b(nnormals * 3);
  // Multiply Si with L to get the resulting b vector
  thrust::multiplies<real> op;
  thrust::transform(
    cyclic_Si, cyclic_Si + nnormals * 3, cyclic_L, d_b.begin(), op);
  printf("b has been built %dx%d\n", d_b.size(), 1);
  //cusp::print(d_b.subarray(0, 25));

  // Now we can solve for the relative normals via SOR
  cusp::array1d<real, cusp::device_memory> d_x(3 * nnormals, 1.f);
  thrust::tabulate(
   d_x.begin(), d_x.end(), [=] __host__ __device__(int x) -> real {
     return x >= nnormals * 2;
   });
#if 0
  {
    using vec3 = thrust::tuple<real, real, real>;
    const auto normalize_vec = [] __host__ __device__(vec3 normal) {
      const real rlen =
        1.f / std::sqrt(sqr(normal.get<0>()) + sqr(normal.get<1>()) +
                        sqr(normal.get<2>()));
      normal.get<0>() *= rlen;
      normal.get<1>() *= rlen;
      normal.get<2>() = std::abs(normal.get<2>() * rlen);
      return normal;
    };
    auto norm_begin = detail::zip_it(
      d_x.begin(), d_x.begin() + nnormals, d_x.begin() + nnormals * 2);
    auto norm_end = norm_begin + nnormals;

    cusp::precond::diagonal<real, cusp::device_memory> M(d_A);
    cusp::monitor<real> monitor(d_b, 2000, 1e-8, 0, true);
    thrust::transform(norm_begin, norm_end, norm_begin, normalize_vec);
    cusp::krylov::cg(d_A, d_x, d_b, monitor, M);

    thrust::transform(norm_begin, norm_end, norm_begin, normalize_vec);
  }
#else
  {
    cusp::relaxation::sor<real, cusp::device_memory> M(d_A, 1.0f);
    // cusp::relaxation::jacobi<float, cusp::device_memory> M(d_A);
    cusp::array1d<real, cusp::device_memory> d_r(nnormals * 3);

    auto norm_begin = detail::zip_it(
      d_x.begin(), d_x.begin() + nnormals, d_x.begin() + nnormals * 2);
    auto norm_end = norm_begin + nnormals;

    using vec3 = thrust::tuple<real, real, real>;
    const auto normalize_vec = [] __host__ __device__(vec3 normal) {
      const real rlen =
        1.f / std::sqrt(sqr(normal.get<0>()) + sqr(normal.get<1>()) +
                        sqr(normal.get<2>()));
      normal.get<0>() *= rlen;
      normal.get<1>() *= rlen;
      normal.get<2>() = std::abs(normal.get<2>() * rlen);
      return normal;
    };
    const auto normalize_all = [=] __host__ __device__ {
      thrust::transform(norm_begin, norm_end, norm_begin, normalize_vec);
    };
    // Compute the initial residual
    normalize_all();
    cusp::multiply(d_A, d_x, d_r);
    cusp::blas::axpy(d_b, d_r, -1.f);

    // Monitor the convergence
    cusp::monitor<real> monitor(d_b, 25, 1e-7, 0, true);

    for (; !monitor.finished(d_r); ++monitor)
    {
      M(d_A, d_b, d_x);
      // Compute the residual
      cusp::multiply(d_A, d_x, d_r);
      cusp::blas::axpy(d_b, d_r, -1.f);
    }
    // Normalize
    normalize_all();
  }
#endif
  printf("Done\n");
  auto d_relative_normals = cusp::make_array2d_view(
    3, nnormals, nnormals, cusp::make_array1d_view(d_x), cusp::row_major{});
  make_host_image(d_relative_normals, h_image.get());
  stbi::writef("relative_normals.png", h_image);
  //cusp::print(d_x.subarray(0, 10));
  //cusp::print(d_x.subarray(nnormals, 10));
  //cusp::print(d_x.subarray(nnormals * 2, 10));

#if 1
  // Now that we have relative normals, we calculate the relative heights
  cusp::coo_matrix<int, real, cusp::device_memory> d_Q(
    nnormals, nnormals, height * (4 * width - 2) - 2 * width);
  // Initialize a grid matrix using CUSP
  cusp::gallery::grid2d(d_Q, width, height);
  build_Q_values(d_relative_normals, d_Q);
  //cusp::print(d_Q.values.subarray(0, 20));
  //cusp::print(d_Q.row_indices.subarray(0, 20));
  //cusp::print(d_Q.column_indices.subarray(0, 20));

  // Now we can assemble a poisson problem to solve the absolute heights
  cusp::array1d<real, cusp::device_memory> d_pb(nnormals);
  thrust::reduce_by_key(d_Q.row_indices.begin(),
                        d_Q.row_indices.end(),
                        d_Q.values.begin(),
                        thrust::make_discard_iterator(),
                        d_pb.begin());
  printf("pb: \n");
  //cusp::print(d_pb.subarray(0, 20));

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
  // value to some constant. I use the final height.
  // Make the last equation a trivial solution
  d_pA.values.begin()[0] = 1.f;
  d_pA.values.begin()[1] = 0.f;
  d_pA.values.begin()[2] = 0.f;
  // Need to replace any references to the final solution with constants in b
  d_pA.values.begin()[3] = 0.f;
  d_pA.values.begin()[4 * width - 2] = 0.f;
  d_pb[0] = 1.f;
  d_pb.begin()[1] += d_pb[0];
  d_pb.begin()[width] += d_pb[0];
  std::cout<<"q10: "<<d_Q.values[2]<<'\n';

  cusp::array1d<real, cusp::device_memory> d_h(nnormals, 1.f);
  d_h[0] = d_pb[0];
#if 0
  {
    cusp::precond::diagonal<real, cusp::device_memory> M(d_pA);
    cusp::monitor<real> monitor(d_pb, std::stoi(argv[5]), 1e-4, 0, true);
    cusp::krylov::cg(d_pA, d_h, d_pb, monitor, M);
  }
#else
  {
    cusp::csr_matrix<int, real, cusp::device_memory> pA(
      d_pA.num_rows, d_pA.num_cols, d_pA.num_entries);
    cusp::indices_to_offsets(d_pA.row_indices, pA.row_offsets);
    pA.column_indices = d_pA.column_indices;
    pA.values = d_pA.values;
    // cusp::relaxation::sor<real, cusp::device_memory> M(pA, 1.0);
    cusp::relaxation::jacobi<real, cusp::device_memory> M(pA);
    cusp::array1d<real, cusp::device_memory> d_r(nnormals, 1.f);
    // Monitor the convergence
    cusp::monitor<real> monitor(d_pb, std::stoi(argv[5]), 1e-4, 0, true);
    cusp::multiply(pA, d_h, d_r);
    cusp::blas::axpy(d_pb, d_r, -1.f);

    for (; !monitor.finished(d_r); ++monitor)
    {
      M(pA, d_pb, d_h);
      // Compute the residual
      cusp::multiply(pA, d_h, d_r);
      cusp::blas::axpy(d_pb, d_r, -1.f);
    }
  }
#endif
  printf("H0: %f, H1: %f, H2:%f, H4:%f, Q10: %f, Q12: %f, Q14: %f\n",
         (real)d_h[0],
         (real)d_h[1],
         (real)d_h[2],
         (real)d_h[4],
         (real)d_Q.values[2],
         (real)d_Q.values[3],
         (real)d_Q.values[4]);
  //cusp::print(d_Q);
  //cusp::print(d_relative_normals);

  print_range_avg(d_h);
  normalize(d_h);

  const auto h_out = detail::zip_it(d_h.begin(), d_h.begin(), d_h.begin());
  const auto rn_begin = detail::zip_it(d_relative_normals.row(0).begin(),
                                       d_relative_normals.row(1).begin(),
                                       d_relative_normals.row(2).begin());
  thrust::copy_n(h_out, nnormals, rn_begin);

  make_host_image(d_relative_normals, h_image.get());
  stbi::writef("height.png", h_image);
#endif
}

