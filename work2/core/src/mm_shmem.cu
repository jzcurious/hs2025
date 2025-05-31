#include "work2/mm_impls/kernel_mm_shmem.cuh"
#include "work2/mm_impls/mm_dispatch.hpp"
#include "work2/mm_impls/mm_shmem.hpp"

#include "cudagh.hpp"

template <ScalarKind ScalarT>
MatrixView<ScalarT>& w2::matmul_shmem(
    const MatrixView<ScalarT>& a, const MatrixView<ScalarT>& b, MatrixView<ScalarT>& c) {
  constexpr const dim3 block_size = {16, 16};

  const dim3 grid_size = {
      cudagh::cover(b.size(1), block_size.x),
      cudagh::cover(a.size(0), block_size.y),
  };

  kernel_mm_shmem<block_size.x><<<grid_size, block_size>>>(a, b, c);
  return c;
}

MM_DISPATCH_FOR_ALL_SUPPORTED_TYPES(w2::matmul_shmem);
