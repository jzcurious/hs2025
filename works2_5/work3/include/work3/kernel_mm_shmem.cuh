#ifndef _KERNEL_MM_SHMEM_CUH_
#define _KERNEL_MM_SHMEM_CUH_

#include "work2/matrix_kind.hpp"

#include <cstdint>
#include <cuda_fp16.h>

template <MatrixKind MatrixT, std::uint32_t tile_side_len = 16>
__global__ void kernel_mm_shmem(const MatrixT a, const MatrixT b, MatrixT c) {
  using scalar_t = typename MatrixT::scalar_t;

  __shared__ scalar_t tile_a[tile_side_len][tile_side_len];
  __shared__ scalar_t tile_b[tile_side_len][tile_side_len];

  const std::uint32_t i = tile_side_len * blockIdx.y + threadIdx.y;
  const std::uint32_t j = tile_side_len * blockIdx.x + threadIdx.x;

  std::uint32_t tx = threadIdx.x;
  std::uint32_t ty = threadIdx.y;

  std::uint32_t m = a.size(0);
  std::uint32_t n = b.size(1);
  std::uint32_t k = a.size(1);

  const scalar_t zero = 0;
  scalar_t acc = zero;

  for (std::uint32_t v = 0; v < k; v += tile_side_len) {
    tile_a[ty][tx] = (v + tx < k) and (i < m) ? a(i, v + tx) : zero;
    tile_b[ty][tx] = (v + ty < k) and (j < n) ? b(v + ty, j) : zero;

    __syncthreads();

    for (std::uint32_t t = 0; t < tile_side_len; ++t) {
      acc += tile_a[ty][t] * tile_b[t][tx];
    }

    __syncthreads();
  }

  if (i < m and j < n) c(i, j) = acc;
}

#endif  // _KERNEL_MM_SHMEM_CUH_
