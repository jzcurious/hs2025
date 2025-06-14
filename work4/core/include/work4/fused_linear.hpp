#ifndef _FUSED_LINEAR_HPP_
#define _FUSED_LINEAR_HPP_

#include "work2/matrix/matrix_kind.hpp"
#include "work2/matrix/matrix_operators.hpp"  // IWYU pragma: keep

template <MatrixKind MatrixT>
MatrixT fused_linear(const MatrixT& x, const MatrixT& w, const MatrixT& b) {
  return x * w + b;  // TODO: replace to kernel call
}

#endif  // _FUSED_LINEAR_HPP_
