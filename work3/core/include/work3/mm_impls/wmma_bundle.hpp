#ifndef _WMMA_BUNDLE_HPP_
#define _WMMA_BUNDLE_HPP_

#include "work3/mm_impls/mm_wmma.hpp"

template <ScalarKind ScalarT>
struct OpImplBundleWmma {
  struct op_impl_feature_t {};

  using scalar_t = ScalarT;

  static MatrixView<ScalarT>& multiplies(const MatrixView<ScalarT>& a,
      const MatrixView<ScalarT>& b,
      MatrixView<ScalarT>& c) {
    return w3::matmul_wmma(a, b, c);
  }

  // NOTE: You can add other operator implementations here, such as plus, minus, etc.
};

#endif  // _WMMA_BUNDLE_HPP_
