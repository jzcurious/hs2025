#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include "cuda_timer.hpp"
#include "work1/wrapper_vadd.hpp"

static void BM_EigenVectorAddCPU(benchmark::State& state) {
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

static void BM_OurVectorAddGPU(benchmark::State& state) {
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
    float elapsed_time = 0;

    {
      CUDATimer timer(elapsed_time);
      w1::vadd_f32(d_a, d_b, d_c, len);
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
  int len = state.range(0);

  Eigen::VectorXf a = Eigen::VectorXf::Random(len);
  Eigen::VectorXf b = Eigen::VectorXf::Random(len);

  float *d_a, *d_b, *d_c;
  auto size = len * sizeof(float);

  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  for (auto _ : state) {
    float elapsed_time = 0;

    {
      CUDATimer timer(elapsed_time);
      cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_a, b.data(), size, cudaMemcpyHostToDevice);
      w1::vadd_f32(d_a, d_b, d_c, len);
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
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseRealTime()
    ->MeasureProcessCPUTime();

BENCHMARK(BM_OurVectorAddGPU)
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK(BM_OurVectorAddGPUCopyOverhead)
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_MAIN();

// TODO: allocate vectors from pool
