#ifndef _FUSED_LINEAR_HPP_
#define _FUSED_LINEAR_HPP_

#include "work2/matrix/matrix_kind.hpp"
#include "work2/matrix/matrix_view.cuh"

namespace internal {

template <ScalarKind ScalarT>
MatrixView<ScalarT>& fused_linear(MatrixView<ScalarT>& y,
    const MatrixView<ScalarT>& x,
    const MatrixView<ScalarT>& w,
    const MatrixView<ScalarT>& b);
}

template <MatrixKind MatrixT>
MatrixT fused_linear(const MatrixT& x, const MatrixT& w, const MatrixT& b) {
  auto y = MatrixT(x.size(0), w.size(1), x.ops().hpad(w.view().hpad()));
  return internal::fused_linear(y.view(), x.view(), w.view(), b.view());
}

#endif  // _FUSED_LINEAR_HPP_
