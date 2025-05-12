#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

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

class CUDATimer {
 private:
  float& _elapse_time_s;
  cudaEvent_t _start, _stop;

 public:
  CUDATimer(float& elapse_time_s)
      : _elapse_time_s(elapse_time_s) {
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);

    cudaEventRecord(_start);
  }

  ~CUDATimer() {
    cudaEventRecord(_stop);
    cudaEventSynchronize(_stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, _start, _stop);

    _elapse_time_s = milliseconds / 1000;
  }
};

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

BENCHMARK(BM_EigenVectorAddCPU)
    ->RangeMultiplier(8)
    ->Ranges({
        {8, 1 << 26}
})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->MeasureProcessCPUTime();

BENCHMARK(BM_OurVectorAddGPU)
    ->RangeMultiplier(8)
    ->Ranges({
        {8, 1 << 26}
})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK_MAIN();

// TODO: allocate vectors from pool
