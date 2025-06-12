#ifndef _BM_GPU_MM_TEMPLATE_HPP_
#define _BM_GPU_MM_TEMPLATE_HPP_

#include "work2/matrix/matrix.hpp"
#include "work2/matrix/matrix_operators.hpp"  // IWYU pragma: keep
#include "work2/matrix/scalar_kind.hpp"
#include "work2/mm_impls/op_bundle_kind.hpp"

#include "cuda_timer.hpp"

#include <benchmark/benchmark.h>

template <template <typename> class OpBundleT, ScalarKind ScalarT, bool colmajor>
  requires OpBundleKind<OpBundleT, ScalarT>
void BM_GPUMMTemplate(benchmark::State& state) {
  using matrix_t = DeviceMatrix<OpBundleT, ScalarT>;
  auto mrows_ncols = state.range(0);

  auto a = matrix_t(mrows_ncols, mrows_ncols, {.colmajor_ = colmajor});
  auto b = matrix_t(mrows_ncols, mrows_ncols, {.colmajor_ = colmajor});

  for (auto _ : state) {
    float elapsed_time = 0;

    {
      CUDATimer timer(elapsed_time);
      auto c = a * b;
    }

    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();
    state.SetIterationTime(elapsed_time);
  }
}

#define BENCHMARK_GPU_MM_TEMPLATE(impl_bundle, scalar_type, colmajor)                    \
  BENCHMARK(BM_GPUMMTemplate<impl_bundle, scalar_type, colmajor>)->UseManualTime()

#endif  // _BM_GPU_MM_TEMPLATE_HPP_
