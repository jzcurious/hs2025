#include <concepts>

template <std::floating_point T>
__global__ void kernel_vadd(const T *a, const T *b, T *c, std::size_t len) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len)
    c[i] = a[i] + b[i];
}
