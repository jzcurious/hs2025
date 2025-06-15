#ifndef _GRAPH_LINEAR_OPERATOR_HPP_
#define _GRAPH_LINEAR_OPERATOR_HPP_

#include "work2/matrix/matrix_kind.hpp"
#include "work4/graph_linear.hpp"

template <MatrixKind MatrixT>
MatrixT graph_linear_operator(const MatrixT& x, const MatrixT& w, const MatrixT& b) {
  auto y = MatrixT(x.size(0), w.size(1), x.ops().hpad(w.view().hpad()));
  graph_linear<typename MatrixT::scalar_t>(y.view(), x.view(), w.view(), b.view());
  return y;
}

#endif  // _GRAPH_LINEAR_OPERATOR_HPP_
