#ifndef _GRAPH_LINEAR_HPP_
#define _GRAPH_LINEAR_HPP_

#include "work2/matrix/matrix_view.cuh"

template <ScalarKind ScalarT>
MatrixView<ScalarT>& graph_linear(MatrixView<ScalarT>& y,
    const MatrixView<ScalarT>& x,
    const MatrixView<ScalarT>& w,
    const MatrixView<ScalarT>& b);

#endif  // _GRAPH_LINEAR_HPP_
