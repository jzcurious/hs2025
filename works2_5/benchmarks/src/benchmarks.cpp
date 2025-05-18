#include "cuda_timer.hpp"
#include "work2/matrix_view.cuh"
#include "work2/mm_naive.hpp"

// #define EIGEN_NO_CUDA 1

#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

class MatMulBenchmarkFixtureGPU : public benchmark::Fixture {
 private:
  float* _d_a = nullptr;
  float* _d_b = nullptr;
  float* _d_c = nullptr;

 public:
  // TODO: try to add ctor and refs
  MatrixView<float>* d_a = nullptr;
  MatrixView<float>* d_b = nullptr;
  MatrixView<float>* d_c = nullptr;

  void SetUp(::benchmark::State& state) {
    auto mrows_ncols = state.range(0);
    auto len = mrows_ncols * mrows_ncols;
    auto size = len * sizeof(float);

    cudaMalloc(&_d_a, size);
    cudaMalloc(&_d_b, size);
    cudaMalloc(&_d_c, size);

    d_a = new MatrixView<float>(_d_a, mrows_ncols, mrows_ncols);
    d_b = new MatrixView<float>(_d_b, mrows_ncols, mrows_ncols);
    d_c = new MatrixView<float>(_d_c, mrows_ncols, mrows_ncols);
  }

  void TearDown(::benchmark::State& state) {
    if (_d_a) cudaFree(_d_a);
    if (_d_b) cudaFree(_d_b);
    if (_d_c) cudaFree(_d_c);

    if (d_a) delete d_a;
    if (d_b) delete d_b;
    if (d_c) delete d_c;
  }
};

BENCHMARK_DEFINE_F(MatMulBenchmarkFixtureGPU, BM_MatMulGPU)(benchmark::State& state) {
  for (auto _ : state) {
    float elapsed_time = 0;

    {
      CUDATimer timer(elapsed_time);
      w2::matmul(*d_a, *d_b, *d_c);
    }

    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();

    state.SetIterationTime(elapsed_time);
  }
}

static void BM_MatMulCPU(benchmark::State& state) {
  auto mwors_ncols = state.range(0);

  Eigen::MatrixXf a = Eigen::MatrixXf(mwors_ncols, mwors_ncols);
  Eigen::MatrixXf b = Eigen::MatrixXf(mwors_ncols, mwors_ncols);

  for (auto _ : state) {
    Eigen::MatrixXf c = a * b;
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
}

constexpr const int multiplier = 8;
constexpr const auto range = std::make_pair(8, 1 << 26);
constexpr const auto unit = benchmark::kMillisecond;

BENCHMARK(BM_MatMulCPU)
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseRealTime()
    ->MeasureProcessCPUTime();

BENCHMARK_REGISTER_F(MatMulBenchmarkFixtureGPU, BM_MatMulGPU)
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_MAIN();
