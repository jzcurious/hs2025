#include "work4/dispatch_ternary.hpp"
#include "work4/fused_linear.hpp"
#include "work4/kernel_linear.cuh"

#include "cudagh.hpp"

template <ScalarKind ScalarT>
MatrixView<ScalarT>& internal::fused_linear(MatrixView<ScalarT>& y,
    const MatrixView<ScalarT>& x,
    const MatrixView<ScalarT>& w,
    const MatrixView<ScalarT>& b) {
  constexpr const dim3 block_size = {32, 1};
  constexpr const dim3 wmma_size = {16, 16, 16};

  const dim3 grid_size = {
      cudagh::cover(w.size(1), wmma_size.x),
      cudagh::cover(x.size(0), wmma_size.y),
  };

  if (x.colmajor) {
    kernel_linear<MatrixView<ScalarT>, true, wmma_size.x, wmma_size.y, wmma_size.z>
        <<<grid_size, block_size>>>(y, x, w, b);
    return y;
  }

  kernel_linear<MatrixView<ScalarT>, false, wmma_size.x, wmma_size.y, wmma_size.z>
      <<<grid_size, block_size>>>(y, x, w, b);
  return y;
}

DISPATCH_TERNARY(internal::fused_linear, half);

#if __CUDA_ARCH__ >= 800
DISPATCH_TERNARY(internal::fused_linear, float);
#endif
