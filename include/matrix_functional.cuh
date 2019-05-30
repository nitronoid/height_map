#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_MATRIX_FUNCTIONAL)
#define SPLIT_DEVICE_INCLUDED_DETAIL_MATRIX_FUNCTIONAL

#include "unary_functional.cuh"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>


namespace detail
{
struct transpose_index : public thrust::unary_function<int, int>
{
  const int M;
  const int N;

  transpose_index(int M, int N) : M(M), N(N)
  {
  }

  __host__ __device__ int operator()(int i) const
  {
    const int x = i / M;
    const int y = i % M;
    return y * N + x;
  }
};
template <typename IndexT>
auto make_transpose_iterator(IndexT i_height, IndexT i_width) -> decltype(
  thrust::make_transform_iterator(thrust::make_counting_iterator<IndexT>(0),
                                  transpose_index(i_height, i_width)))
{
  return thrust::make_transform_iterator(
    thrust::make_counting_iterator<IndexT>(0),
    transpose_index(i_height, i_width));
}

template <typename IndexT>
auto make_row_iterator(IndexT i_width)
  -> decltype(thrust::make_transform_iterator(
    thrust::make_counting_iterator<IndexT>(0), unary_divides<int>(i_width)))
{
  return thrust::make_transform_iterator(
    thrust::make_counting_iterator<IndexT>(0), unary_divides<int>(i_width));
}

template <typename IndexT>
auto make_column_iterator(IndexT i_width)
  -> decltype(thrust::make_transform_iterator(
    thrust::make_counting_iterator<IndexT>(0), unary_modulo<int>(i_width)))
{
  return thrust::make_transform_iterator(
    thrust::make_counting_iterator<IndexT>(0), unary_modulo<int>(i_width));
}

}  // namespace detail


#endif  // SPLIT_DEVICE_INCLUDED_DETAIL_MATRIX_FUNCTIONAL
