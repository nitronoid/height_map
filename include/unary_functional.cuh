#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_UNARY_FUNCTIONAL)
#define SPLIT_DEVICE_INCLUDED_DETAIL_UNARY_FUNCTIONAL

namespace detail
{
template <typename T>
struct reciprocal
{
  __host__ __device__ T operator()(T i_rhs) const
  {
    return T(1) / i_rhs;
  }
};

template <typename T>
struct unary_abs
{
  __host__ __device__ T operator()(T i_x) const
  {
    return std::abs(i_x);
  }
};

template <typename T>
struct unary_modulo
{
  unary_modulo(T i_rhs) : rhs(i_rhs)
  {
  }
  T rhs;

  __host__ __device__ T operator()(T i_lhs) const
  {
    return i_lhs % rhs;
  }
};

template <typename T>
struct unary_exp
{
  __host__ __device__ T operator()(T i_x) const
  {
    return exp(i_x);
  }
};

template <typename T>
struct unary_log
{
  __host__ __device__ T operator()(T i_x) const
  {
    return log(i_x);
  }
};

template <typename T>
struct unary_divides
{
  unary_divides(T i_rhs) : rhs(i_rhs)
  {
  }
  T rhs;

  __host__ __device__ T operator()(T i_lhs) const
  {
    return i_lhs / rhs;
  }
};

template <typename T>
struct unary_multiplies
{
  unary_multiplies(T i_rhs) : rhs(i_rhs)
  {
  }
  T rhs;

  __host__ __device__ T operator()(T i_lhs) const
  {
    return i_lhs * rhs;
  }
};

template <typename T>
struct unary_minus
{
  unary_minus(T i_rhs) : rhs(i_rhs)
  {
  }
  T rhs;

  __host__ __device__ T operator()(T i_lhs) const
  {
    return i_lhs - rhs;
  }
};

template <typename T>
struct unary_plus
{
  unary_plus(T i_rhs) : rhs(i_rhs)
  {
  }
  T rhs;

  __host__ __device__ T operator()(T i_lhs) const
  {
    return i_lhs + rhs;
  }
};
}

#endif // SPLIT_DEVICE_INCLUDED_DETAIL_UNARY_FUNCTIONAL
