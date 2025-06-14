#ifndef _DISPATCH_BINARY_HPP_
#define _DISPATCH_BINARY_HPP_

#include <cuda_fp16.h>

#define DISPATCH_BINARY(impl, scalar_type)                                               \
  template MatrixView<scalar_type>& impl<scalar_type>(MatrixView<scalar_type> & c,       \
      const MatrixView<scalar_type>& a,                                                  \
      const MatrixView<scalar_type>& b);

#define DISPATCH_BINARY_FOR_ALL_TYPES(impl)                                              \
  DISPATCH_BINARY(impl, float);                                                          \
  DISPATCH_BINARY(impl, double);                                                         \
  DISPATCH_BINARY(impl, half);

#endif  // _DISPATCH_BINARY_HPP_
