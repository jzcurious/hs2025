#ifndef _KERNEL_MM_SHMEM_CUH_
#define _KERNEL_MM_SHMEM_CUH_

#include "work2/matrix_kind.hpp"  // IWYU pragma: keep

#include <cstdint>

template <class MatrixT, std::uint32_t tile_side_len = 16>
__global__ void kernel_mm_shmem(const MatrixT a, const MatrixT b, MatrixT c) {
  using scalar_t = typename MatrixT::scalar_t;

  __shared__ scalar_t tile_a[tile_side_len][tile_side_len];
  __shared__ scalar_t tile_b[tile_side_len][tile_side_len];

  std::uint32_t i = tile_side_len * blockIdx.y + threadIdx.y;
  std::uint32_t j = tile_side_len * blockIdx.x + threadIdx.x;

  double acc = 0;

  for (std::uint32_t v = 0; v < a.size(1); v += tile_side_len) {
    std::uint32_t tx = threadIdx.x;
    std::uint32_t ty = threadIdx.y;

    tile_a[ty][tx] = a(i, v + tx);
    tile_b[ty][tx] = b(v + ty, j);

    __syncthreads();

    for (std::uint32_t t = 0; t < tile_side_len; ++t) {
      acc += tile_a[ty][t] * tile_b[t][tx];
    }

    __syncthreads();
  }

  c(i, j) = acc;
}

#endif  // _KERNEL_MM_SHMEM_CUH_
