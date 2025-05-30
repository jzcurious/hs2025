#include "work2/benchmarks/bm_gpu_mm_template.hpp"

#include "work2/mm_impls/naive_bundle.hpp"
#include "work2/mm_impls/shmem_bundle.hpp"

#include <Eigen/Dense>
#include <benchmark/benchmark.h>

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

#define BENCHMARK_GPU_MM_TEMPLATE_(name, impl_bundle, scalar_type, colmajor)             \
  BENCHMARK_GPU_MM_TEMPLATE(name, impl_bundle, scalar_type, colmajor)                    \
      ->RangeMultiplier(multiplier)                                                      \
      ->Ranges({range})                                                                  \
      ->Unit(unit)

BENCHMARK_GPU_MM_TEMPLATE(OpImplBundleNaive, float, false);
BENCHMARK_GPU_MM_TEMPLATE(OpImplBundleNaive, float, true);

BENCHMARK_GPU_MM_TEMPLATE(OpImplBundleShmem, float, false);
BENCHMARK_GPU_MM_TEMPLATE(OpImplBundleShmem, float, true);

BENCHMARK_MAIN();
