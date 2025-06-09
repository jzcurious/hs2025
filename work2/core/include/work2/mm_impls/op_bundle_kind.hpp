#ifndef _OP_BUNDLE_KIND_HPP_
#define _OP_BUNDLE_KIND_HPP_

#include "work2/matrix/matrix_view_kind.hpp"

#include <concepts>  // IWYU pragma: keep

template <class T>
using arg_t = std::add_lvalue_reference_t<std::add_const_t<T>>;

template <class T>
using res_t = std::add_lvalue_reference_t<T>;

#define REQUIRES_BINARY_OP(name)                                                         \
  {                                                                                      \
    U<T>::name(std::declval<res_t<typename U<T>::result_t>>(),                           \
        std::declval<arg_t<typename U<T>::result_t>>(),                                  \
        std::declval<arg_t<typename U<T>::result_t>>())                                  \
  } -> LRefMatrixViewKind

template <template <class> class U, class T>
concept OpBundleKind = requires {
  typename U<T>::op_impl_feature_t;
  typename U<T>::scalar_t;
  typename U<T>::result_t;
} and requires() {
  REQUIRES_BINARY_OP(add);
  REQUIRES_BINARY_OP(sub);
  REQUIRES_BINARY_OP(mul);
  REQUIRES_BINARY_OP(div);
  REQUIRES_BINARY_OP(matmul);
};

#undef REQUIRES_BINARY_OP

#endif  // _OP_BUNDLE_KIND_HPP_
