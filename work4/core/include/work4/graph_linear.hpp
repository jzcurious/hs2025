#ifndef _GRAPH_LINEAR_HPP_
#define _GRAPH_LINEAR_HPP_

#include "work2/matrix/matrix_view.cuh"
#include "work2/matrix/scalar_kind.hpp"

struct GraphLinear {
 private:
  cudaStream_t _stream;
  cudaGraph_t _graph;
  cudaGraphExec_t _graph_inst;
  bool _graph_created;

  template <ScalarKind ScalarT>
  MatrixView<ScalarT>& graph_linear(MatrixView<ScalarT>& y,
      const MatrixView<ScalarT>& x,
      const MatrixView<ScalarT>& w,
      const MatrixView<ScalarT>& b);

 public:
  GraphLinear();

  template <ScalarKind ScalarT>
  MatrixView<ScalarT>& operator()(MatrixView<ScalarT>& y,
      const MatrixView<ScalarT>& x,
      const MatrixView<ScalarT>& w,
      const MatrixView<ScalarT>& b);

  ~GraphLinear();
};

extern GraphLinear graph_linear;

#endif  // _GRAPH_LINEAR_HPP_
