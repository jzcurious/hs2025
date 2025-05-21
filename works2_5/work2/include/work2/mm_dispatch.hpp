#ifndef _MM_DISPATCH_HPP_
#define _MM_DISPATCH_HPP_

#include <cuda_fp16.h>

#define MM_DISPATCH(impl, scalar_type)                                                   \
  template void impl<MatrixView<scalar_type>>(const MatrixView<scalar_type>&,            \
      const MatrixView<scalar_type>&,                                                    \
      MatrixView<scalar_type>&)

#define MM_DISPATCH_FOR_ALL_SUPPORTED_TYPES(impl)                                        \
  MM_DISPATCH(impl, float);                                                              \
  MM_DISPATCH(impl, double);                                                             \
  MM_DISPATCH(impl, half);

#endif  // _MM_DISPATCH_HPP_
