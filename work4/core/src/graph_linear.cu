#include "work3/mm_impls/kernel_mm_wmma.cuh"
#include "work4/dispatch_ternary.hpp"
#include "work4/graph_linear.hpp"
#include "work4/kernel_broadcast_add.cuh"

#include "cudagh.hpp"

template <ScalarKind ScalarT>
MatrixView<ScalarT>& graph_linear(MatrixView<ScalarT>& y,
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

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaGraph_t graph;
  cudaGraphExec_t graph_inst;

  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (x.colmajor) {
    kernel_mm_wmma<MatrixView<ScalarT>, true, wmma_size.x, wmma_size.y, wmma_size.z>
        <<<mm_grid_size, block_size, 0, stream>>>(y, x, w);
  } else {
    kernel_mm_wmma<MatrixView<ScalarT>, false, wmma_size.x, wmma_size.y, wmma_size.z>
        <<<mm_grid_size, block_size, 0, stream>>>(y, x, w);
  }
  kernel_broadcast_add<<<add_grid_size, block_size, 0, stream>>>(y, y, b);
  cudaStreamEndCapture(stream, &graph);

  cudaGraphInstantiate(&graph_inst, graph);

  cudaGraphLaunch(graph_inst, stream);
  cudaStreamSynchronize(stream);

  cudaGraphExecDestroy(graph_inst);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);

  return y;
}

DISPATCH_TERNARY(graph_linear, half);

#if __CUDA_ARCH__ >= 800
DISPATCH_TERNARY(graph_linear, float);
#endif
