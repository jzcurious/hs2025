#include "work2/benchmarks/bm_gpu_mm_preset1.hpp"

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

BENCHMARK(BM_MatMulCPU)
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseRealTime()
    ->MeasureProcessCPUTime();

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpImplBundleNaive, float, false);
BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpImplBundleNaive, float, true);

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpImplBundleShmem, float, false);
BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpImplBundleShmem, float, true);

BENCHMARK_MAIN();
