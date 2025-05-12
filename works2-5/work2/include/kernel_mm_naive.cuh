#ifndef _KERNEL_MM_NAIVE_
#define _KERNEL_MM_NAIVE_

#include "matrix_view.cuh"

template <MatrixKind MatrixT>
__global__ void kernel_mm_naive(const MatrixT a, const MatrixT b, MatrixT c) {
  std::size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= a.size(0) and j >= b.size(1)) return;

  std::size_t k = a.size(1);
  double acc = 0;
  for (std::size_t t = 0; t < k; ++t) acc += a(i, t) * b(t, j);
  c(i, j) = acc;
}

#endif  // _KERNEL_MM_NAIVE_
