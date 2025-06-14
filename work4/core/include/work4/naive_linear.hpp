#ifndef _NAIVE_LINEAR_HPP_
#define _NAIVE_LINEAR_HPP_

#include "work2/matrix/matrix_kind.hpp"
#include "work2/matrix/matrix_operators.hpp"  // IWYU pragma: keep

template <MatrixKind MatrixT>
MatrixT naive_linear(const MatrixT& x, const MatrixT& w, const MatrixT& b) {
  return x * w + b;
}

#endif  // _NAIVE_LINEAR_HPP_
