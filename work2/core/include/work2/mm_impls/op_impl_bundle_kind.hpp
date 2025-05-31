#ifndef _OP_IMPL_BUNDLE_KIND_HPP_
#define _OP_IMPL_BUNDLE_KIND_HPP_

#include <concepts>  // IWYU pragma: keep

template <template <class> class U, class T>
concept OpImplBundleKind = requires {
  typename U<T>::op_impl_feature_t;
  U<T>::multiplies;
  // TODO: Add interface requirement
  // TODO: Add U<T>::plus, U<T>::minus, etc.
};

#endif  // _OP_IMPL_BUNDLE_KIND_HPP_
