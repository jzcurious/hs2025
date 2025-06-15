#ifndef _KERNEL_BROADCAST_ADD_CUH_
#define _KERNEL_BROADCAST_ADD_CUH_

#include "work2/matrix/matrix_view_kind.hpp"

#include <cstdint>

template <MatrixViewKind MatrixT>
__global__ void kernel_broadcast_add(MatrixT c, const MatrixT a, const MatrixT b) {
  const std::uint32_t i = blockDim.y * blockIdx.y + threadIdx.y;
  const std::uint32_t j = blockDim.x * blockIdx.x + threadIdx.x;

  // TODO: add shared memory

  if (i < a.size(0) and j < a.size(1)) c(i, j) = a(i, j) + b(0, j);
}

#endif  // _KERNEL_BROADCAST_ADD_CUH_
