#ifndef _KERNEL_ADD_CUH_
#define _KERNEL_ADD_CUH_

#include "work2/matrix/matrix_view_kind.hpp"

#include <cstdint>

template <MatrixViewKind MatrixT>
__global__ void kernel_add(MatrixT& c, const MatrixT& a, const MatrixT& b) {
  const std::uint32_t i = blockDim.y * blockIdx.y + threadIdx.y;
  const std::uint32_t j = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < a.size(0) and j < a.size(1)) c(i, j) = a(i, j) + b(i, j);
}

#endif  // _KERNEL_ADD_CUH_
