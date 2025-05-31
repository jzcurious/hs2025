#ifndef _KERNEL_MM_NAIVE_CUH_
#define _KERNEL_MM_NAIVE_CUH_

#include "work2/matrix/matrix_view_kind.hpp"

#include <cstdint>
#include <cuda_fp16.h>
#include <type_traits>

template <MatrixViewKind MatrixT>
__global__ void kernel_mm_naive(const MatrixT a, const MatrixT b, MatrixT c) {
  using scalar_t = typename MatrixT::scalar_t;
  using acc_t = std::conditional_t<std::is_same_v<scalar_t, half>, float, scalar_t>;

  std::uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
  std::uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= a.size(0) or j >= b.size(1)) return;

  std::uint32_t k = a.size(1);
  acc_t acc = 0;
  for (std::uint32_t t = 0; t < k; ++t) acc += static_cast<acc_t>(a(i, t) * b(t, j));
  c(i, j) = static_cast<scalar_t>(acc);
}

#endif  // _KERNEL_MM_NAIVE_CUH_
