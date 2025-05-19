#include "cuda_grid_heuristics.hpp"
#include "work2/matrix_kind.hpp"
#include "work2/matrix_view.cuh"

#include "work3/kernel_mm_shmem.cuh"
#include "work3/mm_shmem.hpp"

template <MatrixKind MatrixT>
void w3::matmul(const MatrixT& a, const MatrixT& b, MatrixT& c) {
  constexpr const std::uint32_t tile_side_len = 16;
  constexpr const dim3 block_size = {tile_side_len, tile_side_len};

  const dim3 grid_size = {
      heuristic::cover(b.size(1), block_size.x),
      heuristic::cover(a.size(0), block_size.y),
  };

  kernel_mm_shmem<MatrixT, tile_side_len><<<grid_size, block_size>>>(a, b, c);
}

template void w3::matmul<MatrixView<float>>(
    const MatrixView<float>& a, const MatrixView<float>& b, MatrixView<float>& c);
