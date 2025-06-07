#include "cuda_timer.hpp"
#include "work1/vadd.hpp"

#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

static void BM_EigenVectorAddCPU(benchmark::State& state) {
  auto len = state.range(0);

  Eigen::VectorXf a = Eigen::VectorXf(len);
  Eigen::VectorXf b = Eigen::VectorXf(len);
  Eigen::VectorXf result(len);

  for (auto _ : state) {
    result = a + b;  // lazy RHS
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

static void BM_OurVectorAddGPU(benchmark::State& state) {
  auto len = state.range(0);
  auto size = state.range(0) * sizeof(float);

  float *d_a, *d_b, *d_c;

  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  for (auto _ : state) {
    float elapsed_time = 0;

    {
      CUDATimer timer(elapsed_time);
      w1::vadd_f32(d_c, d_a, d_b, len);
    }

    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();

    state.SetIterationTime(elapsed_time);
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

static void BM_OurVectorAddGPUCopyOverhead(benchmark::State& state) {
  auto len = state.range(0);
  auto size = len * sizeof(float);

  float *d_a, *d_b, *d_c;

  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  float* h_a = new float[len];
  float* h_b = h_a;
  float* h_c = h_a;

  for (auto _ : state) {
    float elapsed_time = 0;

    {
      CUDATimer timer(elapsed_time);
      cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
      w1::vadd_f32(d_c, d_a, d_b, len);
      cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    }

    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();

    state.SetIterationTime(elapsed_time);
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

constexpr const int multiplier = 8;
constexpr const auto range = std::make_pair(8, 1 << 26);
constexpr const auto unit = benchmark::kMillisecond;

BENCHMARK(BM_EigenVectorAddCPU)
    ->Name("Eigen Vector Addition (CPU)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseRealTime()
    ->MeasureProcessCPUTime();

BENCHMARK(BM_OurVectorAddGPU)
    ->Name("CUDA Vector Addition (GPU)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK(BM_OurVectorAddGPUCopyOverhead)
    ->Name("CUDA Vector Addition (GPU, with copy overhead)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_MAIN();
