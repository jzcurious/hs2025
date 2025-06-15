#ifndef _DISPATCH_TERNARY_HPP_
#define _DISPATCH_TERNARY_HPP_

#include <cuda_fp16.h>

#define DISPATCH_TERNARY(impl, scalar_type)                                              \
  template MatrixView<scalar_type>& impl<scalar_type>(MatrixView<scalar_type> & d,       \
      const MatrixView<scalar_type>& a,                                                  \
      const MatrixView<scalar_type>& b,                                                  \
      const MatrixView<scalar_type>& c);

#define DISPATCH_TERNARY_FOR_ALL_TYPES(impl)                                             \
  DISPATCH_TERNARY(impl, float);                                                         \
  DISPATCH_TERNARY(impl, double);                                                        \
  DISPATCH_TERNARY(impl, half);

#endif  // _DISPATCH_TERNARY_HPP_
