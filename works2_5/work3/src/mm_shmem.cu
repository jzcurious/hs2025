#include "cuda_grid_heuristics.hpp"
#include "work2/matrix_kind.hpp"
#include "work2/matrix_view.cuh"
#include "work2/mm_dispatch.hpp"

#include "work3/kernel_mm_shmem.cuh"
#include "work3/mm_shmem.hpp"

template <MatrixKind MatrixT>
void w3::matmul(const MatrixT& a, const MatrixT& b, MatrixT& c) {
  constexpr const dim3 block_size = {16, 16};

  const dim3 grid_size = {
      heuristic::cover(b.size(1), block_size.x),
      heuristic::cover(a.size(0), block_size.y),
  };

  kernel_mm_shmem<MatrixT, block_size.x><<<grid_size, block_size>>>(a, b, c);
}

MM_DISPATCH_FOR_ALL_SUPPORTED_TYPES(w3::matmul);
