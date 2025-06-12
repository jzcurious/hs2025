#include "work2/tests/test_mm_preset_tiling.hpp"
#include "work3/mm_impls/wmma_bundle.hpp"

#include <cuda_fp16.h>

INSTANTIATE_MM_TEST_SUITE_FOR_TYPE_TILING(OpBundleWmma, half, 16, 1e-2);

#if __CUDA_ARCH__ >= 800
INSTANTIATE_MM_TEST_SUITE_FOR_TYPE_TILING(OpBundleWmma, float, 16, 1e-5);
#endif
