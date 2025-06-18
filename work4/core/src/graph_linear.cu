#include "work3/mm_impls/kernel_mm_wmma.cuh"
#include "work4/dispatch_ternary.hpp"
#include "work4/graph_linear.hpp"
#include "work4/kernel_broadcast_add.cuh"

#include "cudagh.hpp"

template <ScalarKind ScalarT>
MatrixView<ScalarT>& GraphLinear::graph_linear(MatrixView<ScalarT>& y,
    const MatrixView<ScalarT>& x,
    const MatrixView<ScalarT>& w,
    const MatrixView<ScalarT>& b) {

  constexpr const dim3 block_size = {32, 1};
  constexpr const dim3 wmma_size = {16, 16, 16};

  const dim3 mm_grid_size = {
      cudagh::cover(w.size(1), wmma_size.x),
      cudagh::cover(x.size(0), wmma_size.y),
  };

  const dim3 add_grid_size = {
      cudagh::cover(w.size(1), block_size.x),
      cudagh::cover(x.size(0), block_size.y),
  };

  if (x.colmajor) {
    kernel_mm_wmma<MatrixView<ScalarT>, true, wmma_size.x, wmma_size.y, wmma_size.z>
        <<<mm_grid_size, block_size, 0, _stream>>>(y, x, w);
  } else {
    kernel_mm_wmma<MatrixView<ScalarT>, false, wmma_size.x, wmma_size.y, wmma_size.z>
        <<<mm_grid_size, block_size, 0, _stream>>>(y, x, w);
  }
  kernel_broadcast_add<<<add_grid_size, block_size, 0, _stream>>>(y, y, b);

  return y;
}

template <ScalarKind ScalarT>
MatrixView<ScalarT>& GraphLinear::operator()(MatrixView<ScalarT>& y,
    const MatrixView<ScalarT>& x,
    const MatrixView<ScalarT>& w,
    const MatrixView<ScalarT>& b) {

  if (not _graph_created) {
    cudaStreamBeginCapture(_stream, cudaStreamCaptureModeGlobal);
    graph_linear(y, x, w, b);
    cudaStreamEndCapture(_stream, &_graph);
    cudaGraphInstantiate(&_graph_inst, _graph);
    _graph_created = true;
  }

  cudaGraphLaunch(_graph_inst, _stream);
  cudaStreamSynchronize(_stream);

  return y;
}

GraphLinear graph_linear;

DISPATCH_TERNARY(GraphLinear::operator(), half);

#if __CUDA_ARCH__ >= 800
DISPATCH_TERNARY(GraphLinear::operator(), float);
#endif
