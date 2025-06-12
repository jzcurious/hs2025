#ifndef _DEFAULT_BUNDLE_HPP_
#define _DEFAULT_BUNDLE_HPP_

#include "work2/mm_impls/op_bundle.hpp"
#include "work3/mm_impls/mm_wmma.hpp"
#include "work4/add.hpp"

// clang-format off
MAKE_BUNDLE(OpBundleDefault) {
  BUNDLE_REGISTER_IMPLEMENTED_BINARY(matmul, w3::matmul_wmma);
  BUNDLE_REGISTER_IMPLEMENTED_BINARY(add, w4::add);
};

// clang-format on

#endif  // _DEFAULT_BUNDLE_HPP_
