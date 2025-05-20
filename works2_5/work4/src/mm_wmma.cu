#include "cuda_grid_heuristics.hpp"
#include "work2/matrix_kind.hpp"
#include "work2/matrix_view.cuh"

#include "work4/kernel_mm_wmma.cuh"
#include "work4/mm_wmma.hpp"

template <MatrixKind MatrixT>
void w4::matmul(const MatrixT& a, const MatrixT& b, MatrixT& c) {
  constexpr const dim3 block_size = {16, 16};

  const dim3 grid_size = {
      heuristic::cover(b.size(1), block_size.x),
      heuristic::cover(a.size(0), block_size.y),
  };

  kernel_mm_wmma<MatrixT, block_size.x><<<grid_size, block_size>>>(a, b, c);
}

template void w4::matmul<MatrixView<float>>(
    const MatrixView<float>& a, const MatrixView<float>& b, MatrixView<float>& c);
