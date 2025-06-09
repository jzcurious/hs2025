#ifndef _WMMA_BUNDLE_HPP_
#define _WMMA_BUNDLE_HPP_

#include "work3/mm_impls/mm_wmma.hpp"

template <ScalarKind ScalarT>
struct OpBundleWmma {
  struct op_impl_feature_t {};

  using scalar_t = ScalarT;

  using result_t = MatrixView<ScalarT>;

  static MatrixView<ScalarT>& mul(MatrixView<ScalarT>& c,
      const MatrixView<ScalarT>& a,
      const MatrixView<ScalarT>& b) {
    return w3::matmul_wmma(c, a, b);
  }

  // NOTE: You can add other operator implementations here, such as plus, minus, etc.
};

#endif  // _WMMA_BUNDLE_HPP_
