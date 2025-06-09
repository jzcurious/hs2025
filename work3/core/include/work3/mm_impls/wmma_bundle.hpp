#ifndef _WMMA_BUNDLE_HPP_
#define _WMMA_BUNDLE_HPP_

#include "work2/mm_impls/op_bundle.hpp"
#include "work3/mm_impls/mm_wmma.hpp"

// clang-format off
MAKE_BUNDLE(OpBundleWmma) {
  BUNDLE_REGISTER_IMPLEMENTED_BINARY(matmul, w3::matmul_wmma);
};

// clang-format on

#endif  // _WMMA_BUNDLE_HPP_
