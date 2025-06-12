#include "work2/mm_impls/naive_bundle.hpp"
#include "work2/mm_impls/shmem_bundle.hpp"
#include "work2/tests/test_mm_preset_no_tiling.hpp"
#include "work2/tests/test_mm_preset_tiling.hpp"

INSTANTIATE_MM_TEST_SUITE_FOR_TYPE_TILING(OpBundleNaive, float, 16, 1e-5);
INSTANTIATE_MM_TEST_SUITE_FOR_TYPE_TILING(OpBundleNaive, double, 16, 1e-5);
INSTANTIATE_MM_TEST_SUITE_FOR_TYPE_TILING(OpBundleNaive, half, 16, 1e-2);

INSTANTIATE_MM_TEST_SUITE_FOR_TYPE_TILING(OpBundleShmem, float, 16, 1e-5);
INSTANTIATE_MM_TEST_SUITE_FOR_TYPE_TILING(OpBundleShmem, double, 16, 1e-5);
INSTANTIATE_MM_TEST_SUITE_FOR_TYPE_TILING(OpBundleShmem, half, 16, 1e-2);

INSTANTIATE_MM_TEST_SUITE_FOR_TYPE_NO_TILING(OpBundleNaive, float, 1e-5);
INSTANTIATE_MM_TEST_SUITE_FOR_TYPE_NO_TILING(OpBundleNaive, double, 1e-5);
INSTANTIATE_MM_TEST_SUITE_FOR_TYPE_NO_TILING(OpBundleNaive, half, 1e-2);

INSTANTIATE_MM_TEST_SUITE_FOR_TYPE_NO_TILING(OpBundleShmem, float, 1e-5);
INSTANTIATE_MM_TEST_SUITE_FOR_TYPE_NO_TILING(OpBundleShmem, double, 1e-5);
INSTANTIATE_MM_TEST_SUITE_FOR_TYPE_NO_TILING(OpBundleShmem, half, 1e-2);
