#ifndef _KERNEL_VADD_CUH_
#define _KERNEL_VADD_CUH_

#include <concepts>
#include <cstdint>

template <std::floating_point ScalarT>
__global__ void kernel_vadd(
    const ScalarT* a, const ScalarT* b, ScalarT* c, std::uint32_t len) {
  std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) c[i] = a[i] + b[i];
}

#endif  // _KERNEL_VADD_CUH_
