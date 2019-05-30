#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_CYCLE_ITERATOR)
#define SPLIT_DEVICE_INCLUDED_DETAIL_CYCLE_ITERATOR

#include "unary_functional.cuh"
#include "matrix_functional.cuh"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace detail
{
template <typename TargetIterator, typename IndexT>
auto make_cycle_iterator(TargetIterator&& iterator, const IndexT length)
  -> decltype(thrust::make_permutation_iterator(
    iterator, detail::make_column_iterator(length)))
{
  return thrust::make_permutation_iterator(
    iterator, detail::make_column_iterator(length));
}
}  // namespace detail


#endif  // SPLIT_DEVICE_INCLUDED_DETAIL_CYCLE_ITERATOR
