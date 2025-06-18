#ifndef _GRAPH_LINEAR_HPP_
#define _GRAPH_LINEAR_HPP_

#include "work2/matrix/matrix_view.cuh"
#include "work2/matrix/scalar_kind.hpp"
#include "work4/graph_suit.hpp"

struct GraphLinear final : public GraphSuit {
  template <ScalarKind ScalarT>
  MatrixView<ScalarT>& graph_linear(MatrixView<ScalarT>& y,
      const MatrixView<ScalarT>& x,
      const MatrixView<ScalarT>& w,
      const MatrixView<ScalarT>& b);

  template <ScalarKind ScalarT>
  MatrixView<ScalarT>& operator()(MatrixView<ScalarT>& y,
      const MatrixView<ScalarT>& x,
      const MatrixView<ScalarT>& w,
      const MatrixView<ScalarT>& b);
};

extern GraphLinear graph_linear;

#endif  // _GRAPH_LINEAR_HPP_
