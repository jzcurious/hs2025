#ifndef _SHMEM_BUNDLE_HPP_
#define _SHMEM_BUNDLE_HPP_

#include "work2/mm_impls/mm_shmem.hpp"
#include "work2/mm_impls/op_bundle.hpp"

// clang-format off
MAKE_BUNDLE(OpBundleShmem) {
  BUNDLE_REGISTER_IMPLEMENTED_BINARY(matmul, w2::matmul_shmem);
};

// clang-format on

#endif  // _SHMEM_BUNDLE_HPP_
