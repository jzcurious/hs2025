#include "cuda_grid_heuristics.hpp"
#include "work1/kernel_vadd.cuh"
#include "work1/vadd.hpp"

void w1::vadd_f32(const float* a, const float* b, float* c, std::uint32_t len) {
  const std::uint32_t block_size = 128;
  const std::uint32_t grid_size = heuristic::cover(len, block_size);
  kernel_vadd<float><<<grid_size, block_size>>>(a, b, c, len);
}
