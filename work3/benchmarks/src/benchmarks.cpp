#include "work2/benchmarks/bm_gpu_mm_preset1.hpp"

#include "work2/mm_impls/naive_bundle.hpp"
#include "work2/mm_impls/shmem_bundle.hpp"
#include "work3/mm_impls/wmma_bundle.hpp"

#include <cuda_fp16.h>

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpImplBundleNaive, float, false);
BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpImplBundleNaive, float, true);

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpImplBundleShmem, float, false);
BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpImplBundleShmem, float, true);

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpImplBundleWmma, half, false);
BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpImplBundleWmma, half, true);

#if __CUDA_ARCH__ >= 800
BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpImplBundleWmma, float, true);
BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpImplBundleWmma, float, false);
#endif

BENCHMARK_MAIN();
