#ifndef _NAIVE_BUNDLE_HPP_
#define _NAIVE_BUNDLE_HPP_

#include "work2/mm_impls/mm_naive.hpp"

template <ScalarKind ScalarT>
struct OpBundleNaive {
  struct op_impl_feature_t {};

  using scalar_t = ScalarT;

  static MatrixView<ScalarT>& multiplies(MatrixView<ScalarT>& c,
      const MatrixView<ScalarT>& a,
      const MatrixView<ScalarT>& b) {
    return w2::matmul_naive(c, a, b);
  }

  // NOTE: You can add other operator implementations here, such as plus, minus, etc.
};

#endif  // _NAIVE_BUNDLE_HPP_
