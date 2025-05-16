#include "cuda_grid_heuristics.hpp"
#include "work2/kernel_mm_naive.cuh"
#include "work2/matrix_view.cuh"
#include "work2/mm_naive.hpp"

template <MatrixKind MatrixT>
void w2::matmul(const MatrixT& a, const MatrixT& b, MatrixT& c) {
  constexpr dim3 block_size = {16, 16};

  const dim3 grid_size = {
      heuristic::cover(b.size(1), block_size.x),
      heuristic::cover(a.size(0), block_size.y),
  };

  kernel_mm_naive<<<grid_size, block_size>>>(a, b, c);
}

template void w2::matmul<MatrixView<float>>(
    const MatrixView<float>& a, const MatrixView<float>& b, MatrixView<float>& c);
