#ifndef _KERNEL_MM_NAIVE_CUH_
#define _KERNEL_MM_NAIVE_CUH_

#include "work2/matrix_kind.hpp"

#include <cstdint>

template <MatrixKind MatrixT>
__global__ void kernel_mm_naive(const MatrixT a, const MatrixT b, MatrixT c) {
  using scalar_t = typename MatrixT::scalar_t;

  std::uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
  std::uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= a.size(0) or j >= b.size(1)) return;

  std::uint32_t k = a.size(1);
  scalar_t acc = 0;
  for (std::uint32_t t = 0; t < k; ++t) acc += a(i, t) * b(t, j);
  c(i, j) = acc;
}

#endif  // _KERNEL_MM_NAIVE_CUH_
