#include "work4/fused_linear_operator.hpp"
#include "work4/graph_linear_operator.hpp"
#include "work4/naive_linear_operator.hpp"

#include "work4/default_bundle.hpp"

#include "work2/benchmarks/bm_gpu_mm_preset1.hpp"

#include <benchmark/benchmark.h>

#define MAKE_LINEAR_LAMBDA(linear_impl)                                                  \
  []<class MatrixT>(const MatrixT& x, const MatrixT& w, const MatrixT& b) {              \
    return linear_impl<MatrixT>(x, w, b);                                                \
  }

template <auto linear_lambda, ScalarKind ScalarT, bool colmajor>
void BM_GPULinearTemplate(benchmark::State& state) {
  using matrix_t = DeviceMatrix<OpBundleDefault, ScalarT>;
  auto mrows_ncols = state.range(0);

  auto x = matrix_t(mrows_ncols, mrows_ncols, {.colmajor_ = colmajor});
  auto w = matrix_t(mrows_ncols, mrows_ncols, {.colmajor_ = colmajor});
  auto b = matrix_t(1, mrows_ncols, {.colmajor_ = colmajor});

  for (auto _ : state) {
    float elapsed_time = 0;

    {
      CUDATimer timer(elapsed_time);
      auto y = linear_lambda.template operator()<matrix_t>(x, w, b);
    }

    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();
    state.SetIterationTime(elapsed_time);
  }
}

#define BENCHMARK_GPU_LINEAR_TEMPLATE(linear_impl, scalar_type, colmajor)                \
  BENCHMARK(BM_GPULinearTemplate<linear_impl, scalar_type, colmajor>)->UseManualTime()

#define BENCHMARK_GPU_LINEAR_TEMPLATE_PRESET_1(linear_impl, scalar_type, colmajor)       \
  BENCHMARK_GPU_LINEAR_TEMPLATE(linear_impl, scalar_type, colmajor)                      \
      ->RangeMultiplier(multiplier)                                                      \
      ->Ranges({range})                                                                  \
      ->Unit(unit)

BENCHMARK_GPU_LINEAR_TEMPLATE_PRESET_1(
    MAKE_LINEAR_LAMBDA(naive_linear_operator), half, false)
    ->Name("CUDA Linear (naive, half, row-major)");

BENCHMARK_GPU_LINEAR_TEMPLATE_PRESET_1(
    MAKE_LINEAR_LAMBDA(fused_linear_operator), half, false)
    ->Name("CUDA Linear (fused, half, row-major)");

BENCHMARK_GPU_LINEAR_TEMPLATE_PRESET_1(
    MAKE_LINEAR_LAMBDA(graph_linear_operator), half, false)
    ->Name("CUDA Linear (graph, half, row-major)");

BENCHMARK_GPU_LINEAR_TEMPLATE_PRESET_1(
    MAKE_LINEAR_LAMBDA(naive_linear_operator), half, true)
    ->Name("CUDA Linear (naive, half, col-major)");

BENCHMARK_GPU_LINEAR_TEMPLATE_PRESET_1(
    MAKE_LINEAR_LAMBDA(fused_linear_operator), half, true)
    ->Name("CUDA Linear (fused, half, col-major)");

BENCHMARK_GPU_LINEAR_TEMPLATE_PRESET_1(
    MAKE_LINEAR_LAMBDA(graph_linear_operator), half, true)
    ->Name("CUDA Linear (graph, half, col-major)");

#if __CUDA_ARCH__ >= 800
BENCHMARK_GPU_LINEAR_TEMPLATE_PRESET_1(
    MAKE_LINEAR_LAMBDA(naive_linear_operator), float, false)
    ->Name("CUDA Linear (naive, float, row-major)");

BENCHMARK_GPU_LINEAR_TEMPLATE_PRESET_1(
    MAKE_LINEAR_LAMBDA(fused_linear_operator), float, false)
    ->Name("CUDA Linear (fused, float, row-major)");

BENCHMARK_GPU_LINEAR_TEMPLATE_PRESET_1(
    MAKE_LINEAR_LAMBDA(graph_linear_operator), float, false)
    ->Name("CUDA Linear (graph, float, row-major)");

BENCHMARK_GPU_LINEAR_TEMPLATE_PRESET_1(
    MAKE_LINEAR_LAMBDA(naive_linear_operator), float, true)
    ->Name("CUDA Linear (naive, float, col-major)");

BENCHMARK_GPU_LINEAR_TEMPLATE_PRESET_1(
    MAKE_LINEAR_LAMBDA(fused_linear_operator), float, true)
    ->Name("CUDA Linear (fused, float, col-major)");

BENCHMARK_GPU_LINEAR_TEMPLATE_PRESET_1(
    MAKE_LINEAR_LAMBDA(graph_linear_operator), float, true)
    ->Name("CUDA Linear (graph, float, col-major)");
#endif

BENCHMARK_MAIN();
