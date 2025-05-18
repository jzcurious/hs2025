#include <cuda_runtime.h>

#include "cuda_timer.hpp"
#include "work2/matrix_view.cuh"
#include "work2/mm_naive.hpp"

#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

using MatrixViewT = MatrixView<float>;

using OpMatMulT
    = std::function<void(const MatrixViewT&, const MatrixViewT&, MatrixViewT&)>;

class MatMulBenchmarkFixtureGPU : public benchmark::Fixture {
 private:
  float* _d_a = nullptr;
  float* _d_b = nullptr;
  float* _d_c = nullptr;

  const MatrixViewT* _view_a = nullptr;
  const MatrixViewT* _view_b = nullptr;
  MatrixViewT* _view_c = nullptr;

 public:
  void SetUp(::benchmark::State& state) {
    auto mrows_ncols = state.range(0);
    auto len = mrows_ncols * mrows_ncols;
    auto size = len * sizeof(float);

    cudaMalloc(&_d_a, size);
    cudaMalloc(&_d_b, size);
    cudaMalloc(&_d_c, size);

    _view_a = new MatrixViewT(_d_a, mrows_ncols, mrows_ncols);
    _view_b = new MatrixViewT(_d_b, mrows_ncols, mrows_ncols);
    _view_c = new MatrixViewT(_d_c, mrows_ncols, mrows_ncols);
  }

  void TearDown(::benchmark::State& state) {
    if (_d_a) cudaFree(_d_a);
    if (_d_b) cudaFree(_d_b);
    if (_d_c) cudaFree(_d_c);

    if (_view_a) delete _view_a;
    if (_view_b) delete _view_b;
    if (_view_c) delete _view_c;
  }

  void run(OpMatMulT mmfunc, benchmark::State& state) {
    for (auto _ : state) {
      float elapsed_time = 0;

      {
        CUDATimer timer(elapsed_time);
        mmfunc(*_view_a, *_view_b, *_view_c);
      }

      benchmark::DoNotOptimize(elapsed_time);
      benchmark::ClobberMemory();

      state.SetIterationTime(elapsed_time);
    }
  }
};

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

constexpr const int multiplier = 2;
constexpr const auto range = std::make_pair(8, 1 << 10);
constexpr const auto unit = benchmark::kMillisecond;

BENCHMARK(BM_MatMulCPU)
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseRealTime()
    ->MeasureProcessCPUTime();

#define BENCHMARK_DEFINE_AND_REGISTER_F_GPU_(bm_name, mm_impl)                           \
  BENCHMARK_DEFINE_F(MatMulBenchmarkFixtureGPU, bm_name)(benchmark::State & state) {     \
    run(mm_impl<MatrixViewT>, state);                                                    \
  }                                                                                      \
  BENCHMARK_REGISTER_F(MatMulBenchmarkFixtureGPU, bm_name)                               \
      ->RangeMultiplier(multiplier)                                                      \
      ->Ranges({range})                                                                  \
      ->Unit(unit)                                                                       \
      ->UseManualTime();

BENCHMARK_DEFINE_AND_REGISTER_F_GPU_(BM_MatMulGPUNaive, w2::matmul);

BENCHMARK_MAIN();
