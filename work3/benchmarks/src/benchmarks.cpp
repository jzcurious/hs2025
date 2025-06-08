#include "work2/benchmarks/bm_gpu_mm_preset1.hpp"

#include "work2/mm_impls/naive_bundle.hpp"
#include "work2/mm_impls/shmem_bundle.hpp"
#include "work3/mm_impls/wmma_bundle.hpp"

#include <cuda_fp16.h>

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpBundleNaive, float, false)
    ->Name("CUDA MM (Naive, float, row-major)");

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpBundleNaive, float, true)
    ->Name("CUDA MM (Naive, float, col-major)");

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpBundleNaive, half, false)
    ->Name("CUDA MM (Naive, half, row-major)");

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpBundleNaive, half, true)
    ->Name("CUDA MM (Naive, half, col-major)");

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpBundleShmem, float, false)
    ->Name("CUDA MM (Shared Memory, float, row-major)");

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpBundleShmem, float, true)
    ->Name("CUDA MM (Shared Memory, float, col-major)");

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpBundleShmem, half, false)
    ->Name("CUDA MM (Shared Memory, half, row-major)");

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpBundleShmem, half, true)
    ->Name("CUDA MM (Shared Memory, half, col-major)");

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpBundleWmma, half, false)
    ->Name("CUDA MM (WMMA, half, row-major)");

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpBundleWmma, half, true)
    ->Name("CUDA MM (WMMA, half, col-major)");

#if __CUDA_ARCH__ >= 800
BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpBundleWmma, float, false)
    ->Name("CUDA MM (WMMA, float, row-major)");

BENCHMARK_GPU_MM_TEMPLATE_PRESET_1(OpBundleWmma, float, true)
    ->Name("CUDA MM (WMMA, float, col-major)");
#endif

BENCHMARK_MAIN();
