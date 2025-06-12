#ifndef _TEST_MM_PRESET_NO_TILING_HPP_
#define _TEST_MM_PRESET_NO_TILING_HPP_

#include "work2/tests/tests_mm_template.hpp"

#define INSTANTIATE_MM_TEST_SUITE_FOR_TYPE_NO_TILING(impl_bundle, scalar_type, tol)      \
  INSTANTIATE_MM_TEST_SUITE_FOR_TYPE(impl_bundle, scalar_type, 1, tol)

#endif  // _TEST_MM_PRESET_NO_TILING_HPP_
