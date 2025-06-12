#ifndef _TEST_MM_PRESET_TILING_HPP_
#define _TEST_MM_PRESET_TILING_HPP_

#include "work2/tests/tests_mm_template.hpp"

#define INSTANTIATE_MM_TEST_SUITE_FOR_TYPE_TILING(                                       \
    impl_bundle, scalar_type, tile_size, tol)                                            \
  INSTANTIATE_MM_TEST_SUITE_FOR_TYPE(impl_bundle, scalar_type, tile_size, tol)

#endif  // test_mm_preset_tiling.hpp
