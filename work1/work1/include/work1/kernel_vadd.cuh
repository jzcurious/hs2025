#include <concepts>

template <std::floating_point ScalarT>
__global__ void kernel_vadd(
    const ScalarT* a, const ScalarT* b, ScalarT* c, std::size_t len) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) c[i] = a[i] + b[i];
}
