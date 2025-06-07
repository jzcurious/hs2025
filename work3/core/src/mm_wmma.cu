#include "work3/mm_impls/kernel_mm_wmma.cuh"
#include "work3/mm_impls/mm_wmma.hpp"

#include "work2/mm_impls/mm_dispatch.hpp"

#include "cudagh.hpp"

template <ScalarKind ScalarT>  // TODO: add wmma tile size
MatrixView<ScalarT>& w3::matmul_wmma(
    MatrixView<ScalarT>& c, const MatrixView<ScalarT>& a, const MatrixView<ScalarT>& b) {
  constexpr const dim3 block_size = {32, 1};
  constexpr const dim3 wmma_size = {16, 16, 16};

  const dim3 grid_size = {
      cudagh::cover(b.size(1), wmma_size.x),
      cudagh::cover(a.size(0), wmma_size.y),
  };

  if (a.colmajor) {
    kernel_mm_wmma<MatrixView<ScalarT>, true, wmma_size.x, wmma_size.y, wmma_size.z>
        <<<grid_size, block_size>>>(c, a, b);
    return c;
  }

  kernel_mm_wmma<MatrixView<ScalarT>, false, wmma_size.x, wmma_size.y, wmma_size.z>
      <<<grid_size, block_size>>>(c, a, b);
  return c;
}

MM_DISPATCH(w3::matmul_wmma, half);

#if __CUDA_ARCH__ >= 800
MM_DISPATCH(w4::matmul_wmma, float);
#endif
