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
#include <cusp/gallery/grid.h>
#include <cusp/gallery/poisson.h>
#include <cusp/print.h>
#include <cusp/convert.h>
#include <cusp/relaxation/sor.h>
#include <cusp/linear_operator.h>
#include <cusp/monitor.h>

#include "matrix_functional.cuh"
#include "zip_it.cuh"
#include "cycle_iterator.cuh"

template <typename T>
constexpr T sqr(T val) noexcept
{
  return val * val;
}

using real = float;
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
  cusparseCreate(&handle);

  cusparseMatDescr_t A_description;
  cusparseCreateMatDescr(&A_description);
  cusparseMatDescr_t B_description;
  cusparseCreateMatDescr(&B_description);
  cusparseMatDescr_t C_description;
  cusparseCreateMatDescr(&C_description);

  cusparseSetMatType(A_description, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(A_description, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(B_description, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(B_description, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(C_description, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(C_description, CUSPARSE_INDEX_BASE_ZERO);

  // Coefficients
  const real alpha = 1.f;
  const real beta = 0.001f;

  std::size_t buffer_size = 0;
  // Not sure if this is needed
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  // Compute the workspace size for the sparse matrix add
  cusparseScsrgeam2_bufferSizeExt(handle,
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
                                  // Not sure why the API forces these
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  &buffer_size);

  using device_temp_t = std::unique_ptr<void, cudaError_t (*)(void* const)>;
  device_temp_t workspace(thrust::device_malloc(buffer_size).get(), cudaFree);

  int C_nnz = 0;
  // &C_nnz points to a host variable

  cusp::array1d<int, cusp::device_memory> C_row_offsets(di_A.num_rows + 1);

  cusparseXcsrgeam2Nnz(handle,
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
                       &C_nnz,
                       workspace.get());
  C_nnz = C_row_offsets[di_A.num_rows] - C_row_offsets[0];

  cusp::csr_matrix<int, real, cusp::device_memory> do_C(
    di_A.num_rows, di_A.num_cols, C_nnz);
  do_C.row_offsets = std::move(C_row_offsets);

  // Now actually do the add
  cusparseScsrgeam2(handle,
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
                    do_C.column_indices.begin().base().get(),
                    workspace.get());
  return do_C;
}

int main(int argc, char* argv[])
{
  assert(argc >= 2);
  auto h_image = stbi::loadf(argv[1], 3);
  printf("Loaded image with dim: %dx%dx%d\n",
         h_image.width(),
         h_image.height(),
         h_image.n_channels());

  cusp::array2d<real, cusp::device_memory> d_image(h_image.n_channels(),
                                                   h_image.n_pixels());
  make_device_image(h_image.get(), d_image);
  // Pick any channel as the intensity (should all be equal)
  auto d_shading_intensity = d_image.row(0);

  const int width = h_image.width();
  const int height = h_image.height();
  const int nnormals = width * height;

  // Lighting direction
  float3 L{1.f, 2.f, 3.f};
  // Outer product of the lighting direction
  // clang-format off
  real LLT[9] = {L.x * L.x, L.x * L.y, L.x * L.z,
                 L.x * L.y, L.y * L.y, L.y * L.z,
                 L.x * L.z, L.y * L.z, L.z * L.z};
  // clang-format on
  // Copy to the device
  thrust::device_vector<real> d_LLT(LLT, LLT + 9);
  cusp::coo_matrix<int, real, cusp::device_memory> d_M(
    nnormals * 3, nnormals * 3, nnormals * 9);
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
  thrust::copy_n(LLT_v, nnormals * 9, d_M.values.begin());

  // The row keys will be i / 3, as we only have 3 values per row and column
  const auto count = thrust::make_counting_iterator(0);
  thrust::transform(count,
                    count + nnormals * 9,
                    d_M.row_indices.begin(),
                    detail::unary_divides<int>(3));

  // To write the column keys we need a repeating sequence of 0, 1, 2 * n to
  // give 0, n, 2n, and then we offset by the row % n
  thrust::transform(LLT_col,
                    LLT_col + nnormals * 9,
                    d_M.row_indices.begin(),
                    d_M.column_indices.begin(),
                    [=] __host__ __device__(int s, int r) {
                      return (r % nnormals) + s * nnormals;
                    });

  // B is our pixel 4-neighborhood adjacency matrix
  cusp::coo_matrix<int, real, cusp::device_memory> d_sB;
  // Build the discrete Poisson problem matrix
  cusp::gallery::poisson5pt(d_sB, width, height);
  // cusp::print(d_sB);
  cusp::coo_matrix<int, real, cusp::device_memory> d_B(
    d_sB.num_rows * 3, d_sB.num_cols * 3, d_sB.num_entries * 3);

  {
    const int nsb = d_sB.num_entries;
    // Correct the boundaries which don't have valence of 4 and copy the
    // corrected B for each channel of the normal vectors (3 times).
    // Tuple of [Row, Column, Value]
    const auto entry_in =
      detail::make_cycle_iterator(detail::zip_it(d_sB.row_indices.begin(),
                                                 d_sB.column_indices.begin(),
                                                 d_sB.values.begin()),
                                  nsb);
    auto entry_out = detail::zip_it(
      d_B.row_indices.begin(), d_B.column_indices.begin(), d_B.values.begin());

    using tup3 = thrust::tuple<int, int, real>;
    // Copy sB 3 times, offsetting by it's width and height for each new copy
    const auto op = [=] __host__ __device__(tup3 entry, int count) {
      // Work out what channel we're in
      const int channel = count / nsb;
      // Fix boundary cell diagonals
      if (entry.get<0>() == entry.get<1>())
      {
        const int r = entry.get<0>() / width;
        const int c = entry.get<0>() % width;
        // If we're in a boundary cell we subtract one from the valence
        entry.get<2>() -= (r == 0 || r == (height - 1));
        entry.get<2>() -= (c == 0 || c == (width - 1));
      }
      // Offset for the channel
      entry.get<0>() += channel * nnormals;
      entry.get<1>() += channel * nnormals;
      return entry;
    };
    thrust::transform(entry_in, entry_in + nsb * 3, count, entry_out, op);
  }
  // cusp::print(d_B);

  // Now we build A using M and B
  // A = M + 8lmI -2lmB <=> A = M + 2lm(4I - B)
  // So we use cuSparse to compute alpha * M + beta * B, where beta is 2lm

  const real lambda = 0.1f;
  // Now we can add M
  auto d_A = cusparse_add(d_M, d_B);
  // cusp::print(d_A);

  // The b vector of the system is (shading intensity * L), where L repeats
  // Copy L to the device
  thrust::device_vector<real> d_L((real*)&L, (real*)&L + 3);
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
  cusp::print(d_b.subarray(0, 25));
  cusp::print(d_shading_intensity.subarray(0, 25));

  // Now we can solve for the relative normals via SOR
  cusp::relaxation::sor<real, cusp::device_memory> M(d_A, 1.0f);
  cusp::array1d<real, cusp::device_memory> d_x(3 * nnormals, 1.f);
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
    normal.get<2>() *= rlen;
    return normal;
  };

  const auto normalize_all = [=] __host__ __device__ {
    thrust::transform(norm_begin, norm_end, norm_begin, normalize_vec);
  };
  normalize_all();
  // Compute the initial residual
  cusp::multiply(d_A, d_x, d_r);
  cusp::blas::axpy(d_b, d_r, -1.f);

  // Monitor the convergence
  cusp::monitor<real> monitor(d_b, 5000, 1e-4, 0, false);

  for (; !monitor.finished(d_r); ++monitor)
  {
    M(d_A, d_b, d_x);
    // Normalize
    normalize_all();
    // Ensure Z is pointing outwards
    //thrust::transform(d_x.begin() + nnormals * 2,
    //                  d_x.end(),
    //                  d_x.begin() + nnormals * 2,
    //                  detail::unary_abs<real>());
    // Compute the residual
    cusp::multiply(d_A, d_x, d_r);
    cusp::blas::axpy(d_b, d_r, -1.f);
  }

  printf("Done\n");
  auto d_relative_normals = cusp::make_array2d_view(
    3, nnormals, nnormals, cusp::make_array1d_view(d_x), cusp::row_major{});

  // Now that we have relative normals, we calculate the relative heights

  make_host_image(d_relative_normals, h_image.get());
  stbi::writef("relative_normals.png", h_image);

  make_host_image(d_image, h_image.get());
  stbi::writef("out.png", h_image);
}

