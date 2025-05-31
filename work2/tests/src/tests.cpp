#include "work2/mm_impls/naive_bundle.hpp"
#include "work2/mm_impls/shmem_bundle.hpp"
#include "work2/tests/tests_mm_template.hpp"

INSTANTIATE_TEST_SUITE_FOR_TYPE(OpImplBundleNaive, float, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE(OpImplBundleNaive, double, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE(OpImplBundleNaive, half, 1e-2);

INSTANTIATE_TEST_SUITE_FOR_TYPE(OpImplBundleShmem, float, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE(OpImplBundleShmem, double, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE(OpImplBundleShmem, half, 1e-2);
