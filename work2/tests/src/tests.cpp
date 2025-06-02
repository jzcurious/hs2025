#include "work2/mm_impls/naive_bundle.hpp"
#include "work2/mm_impls/shmem_bundle.hpp"
#include "work2/tests/test_mm_preset_no_tiling.hpp"
#include "work2/tests/test_mm_preset_tiling.hpp"

INSTANTIATE_TEST_SUITE_FOR_TYPE_TILING(OpImplBundleNaive, float, 16, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE_TILING(OpImplBundleNaive, double, 16, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE_TILING(OpImplBundleNaive, half, 16, 1e-2);

INSTANTIATE_TEST_SUITE_FOR_TYPE_TILING(OpImplBundleShmem, float, 16, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE_TILING(OpImplBundleShmem, double, 16, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE_TILING(OpImplBundleShmem, half, 16, 1e-2);

INSTANTIATE_TEST_SUITE_FOR_TYPE_NO_TILING(OpImplBundleNaive, float, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE_NO_TILING(OpImplBundleNaive, double, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE_NO_TILING(OpImplBundleNaive, half, 1e-2);

INSTANTIATE_TEST_SUITE_FOR_TYPE_NO_TILING(OpImplBundleShmem, float, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE_NO_TILING(OpImplBundleShmem, double, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE_NO_TILING(OpImplBundleShmem, half, 1e-2);
