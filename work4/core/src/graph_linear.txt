#include "work3/mm_impls/kernel_mm_wmma.cuh"
#include "work4/dispatch_ternary.hpp"
#include "work4/graph_linear.hpp"
#include "work4/kernel_broadcast_add.cuh"

#include "cudagh.hpp"

namespace detail {

struct GraphSuit {
 private:
  cudaStream_t _stream;
  cudaGraph_t _graph;
  cudaGraphExec_t _graph_inst;

 public:
  GraphSuit() {
    cudaStreamCreate(&_stream);
    cudaStreamBeginCapture(_stream, cudaStreamCaptureModeGlobal);
  }

  ~GraphSuit() {
    cudaStreamEndCapture(_stream, &_graph);
    cudaGraphInstantiate(&_graph_inst, _graph);
    cudaGraphLaunch(_graph_inst, _stream);
    cudaStreamSynchronize(_stream);
    cudaGraphExecDestroy(_graph_inst);
    cudaGraphDestroy(_graph);
    cudaStreamDestroy(_stream);
  }

  cudaStream_t stream() const {
    return _stream;
  }
};

}  // namespace detail

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

  {
    auto graph_suit = detail::GraphSuit();

    if (x.colmajor) {
      kernel_mm_wmma<MatrixView<ScalarT>, true, wmma_size.x, wmma_size.y, wmma_size.z>
          <<<mm_grid_size, block_size, 0, graph_suit.stream()>>>(y, x, w);
    } else {
      kernel_mm_wmma<MatrixView<ScalarT>, false, wmma_size.x, wmma_size.y, wmma_size.z>
          <<<mm_grid_size, block_size, 0, graph_suit.stream()>>>(y, x, w);
    }
    kernel_broadcast_add<<<add_grid_size, block_size, 0, graph_suit.stream()>>>(y, y, b);
  }

  return y;
}

DISPATCH_TERNARY(graph_linear, half);

#if __CUDA_ARCH__ >= 800
DISPATCH_TERNARY(graph_linear, float);
#endif
