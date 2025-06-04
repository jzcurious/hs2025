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
    ->Name("Eigen MM (float, col-major)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseRealTime()
    ->MeasureProcessCPUTime();

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpImplBundleNaive, float, false)
    ->Name("CUDA MM (Naive, float, row-major)");

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpImplBundleNaive, float, true)
    ->Name("CUDA MM (Naive, float, col-major)");

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpImplBundleShmem, float, false)
    ->Name("CUDA MM (Shared Memory, float, row-major)");

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpImplBundleShmem, float, true)
    ->Name("CUDA MM (Shared Memory, float, col-major)");

BENCHMARK_MAIN();
