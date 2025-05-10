#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include "work1/wrapper_vadd.hpp"

static void BM_EigenVectorAddCPU(benchmark::State &state) {
  int len = state.range(0);

  Eigen::VectorXf a = Eigen::VectorXf::Random(len);
  Eigen::VectorXf b = Eigen::VectorXf::Random(len);
  Eigen::VectorXf result(len);

  for (auto _ : state) {
    result = a + b;
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

static void BM_OurVectorAddGPU(benchmark::State &state) {
  int len = state.range(0);

  Eigen::VectorXf a = Eigen::VectorXf::Random(len);
  Eigen::VectorXf b = Eigen::VectorXf::Random(len);

  float *d_a, *d_b, *d_c;
  auto size = len * sizeof(float);

  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, b.data(), size, cudaMemcpyHostToDevice);

  for (auto _ : state) {
    w1::vadd_f32(d_a, d_b, d_c, len);
    benchmark::DoNotOptimize(d_c);
    benchmark::ClobberMemory();
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

BENCHMARK(BM_EigenVectorAddCPU)
    ->RangeMultiplier(8)
    ->Ranges({{8, 1 << 18}}) // 8..262144
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->MeasureProcessCPUTime();

// BENCHMARK(BM_OurVectorAddGPU)
//     ->RangeMultiplier(8)
//     ->Ranges({{8, 1 << 18}}) // 8..262144
//     ->Unit(benchmark::kMicrosecond)
//     ->UseRealTime()
//     ->MeasureProcessCPUTime();

BENCHMARK_MAIN();

// TODO: allocate vectors from pool
