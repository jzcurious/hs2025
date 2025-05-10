#include "work1/kernel_vadd.cuh"
#include "work1/wrapper_vadd.hpp"

void w1::vadd_f32(const float *a, const float *b, float *c, std::size_t len) {
  const std::size_t block_size = 128;
  const std::size_t grid_size = (len + block_size - 1) / block_size;
  kernel_vadd<float><<<grid_size, block_size>>>(a, b, c, len);
}
