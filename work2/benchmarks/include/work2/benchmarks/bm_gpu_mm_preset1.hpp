#ifndef _BM_GPU_MM_PRESET1_HPP_
#define _BM_GPU_MM_PRESET1_HPP_

#include "work2/benchmarks/bm_gpu_mm_template.hpp"

constexpr const int multiplier = 2;
constexpr const auto range = std::make_pair(16, 1 << 10);
constexpr const auto unit = benchmark::kMillisecond;

#define BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(impl_bundle, scalar_type, colmajor)           \
  BENCHMARK_GPU_MM_TEMPLATE(impl_bundle, scalar_type, colmajor)                          \
      ->RangeMultiplier(multiplier)                                                      \
      ->Ranges({range})                                                                  \
      ->Unit(unit)

#endif  // _BM_GPU_MM_PRESET_HPP_
