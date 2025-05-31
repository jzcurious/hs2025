#include "work3/mm_impls/wmma_bundle.hpp"

#include "work2/tests/tests_mm_template.hpp"

#include <cuda_fp16.h>

INSTANTIATE_TEST_SUITE_FOR_TYPE(OpImplBundleWmma, half, 1e-2);

#if __CUDA_ARCH__ >= 800
INSTANTIATE_TEST_SUITE_FOR_TYPE(OpImplBundleWmma, float, 1e-5);
#endif
