#ifndef _BM_GPU_MM_TEMPLATE_HPP_
#define _BM_GPU_MM_TEMPLATE_HPP_

#include "work2/matrix/matrix.cuh"
#include "work2/matrix/scalar_kind.cuh"

#include "cuda_timer.hpp"

#include <benchmark/benchmark.h>

template <ScalarKind ScalarT>
using OpMatMulT = std::function<void(
    const DeviceMatrix<ScalarT>&, const DeviceMatrix<ScalarT>&, DeviceMatrix<ScalarT>&)>;

template <ScalarKind ScalarT, bool colmajor>
class GPUMMFixture : public benchmark::Fixture {
 public:
  void BM_GPUMMTemplate(OpMatMulT<ScalarT> mmop, benchmark::State& state) {
    auto mrows_ncols = state.range(0);

    auto a = DeviceMatrix<ScalarT>(mrows_ncols, mrows_ncols, colmajor);
    auto b = DeviceMatrix<ScalarT>(mrows_ncols, mrows_ncols, colmajor);
    auto c = DeviceMatrix<ScalarT>(mrows_ncols, mrows_ncols, colmajor);

    for (auto _ : state) {
      float elapsed_time = 0;

      {
        CUDATimer timer(elapsed_time);
        mmop(a, b, c);
      }

      benchmark::DoNotOptimize(elapsed_time);
      benchmark::ClobberMemory();

      state.SetIterationTime(elapsed_time);
    }
  }
};

#define BENCHMARK_GPU_MM_TEMPLATE(name, impl, scalar_type, colmajor)                     \
  BENCHMARK_TEMPLATE_DEFINE_F(GPUMMFixture, name, scalar_type, colmajor)(                \
      benchmark::State & st) {                                                           \
    BM_GPUMMTemplate(impl<scalar_type>, st);                                             \
  }                                                                                      \
  BENCHMARK_REGISTER_F(GPUMMFixture, name)->UseManualTime()

#endif  // _BM_GPU_MM_TEMPLATE_HPP_
