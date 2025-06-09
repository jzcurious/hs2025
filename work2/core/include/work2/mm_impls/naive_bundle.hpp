#ifndef _NAIVE_BUNDLE_HPP_
#define _NAIVE_BUNDLE_HPP_

#include "work2/mm_impls/mm_naive.hpp"
#include "work2/mm_impls/op_bundle.hpp"

// clang-format off
MAKE_BUNDLE(OpBundleNaive) {
  BUNDLE_REGISTER_IMPLEMENTED_BINARY(matmul, w2::matmul_naive);
};

// clang-format on

#endif  // _NAIVE_BUNDLE_HPP_
