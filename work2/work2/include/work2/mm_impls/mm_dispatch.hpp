#ifndef _MM_DISPATCH_HPP_
#define _MM_DISPATCH_HPP_

#include <cuda_fp16.h>

#define MM_DISPATCH(impl, scalar_type)                                                   \
  template void impl<scalar_type>(const DeviceMatrix<scalar_type>& a,                    \
      const DeviceMatrix<scalar_type>& b,                                                \
      DeviceMatrix<scalar_type>& c);

#define MM_DISPATCH_FOR_ALL_SUPPORTED_TYPES(impl)                                        \
  MM_DISPATCH(impl, float);                                                              \
  MM_DISPATCH(impl, double);                                                             \
  MM_DISPATCH(impl, half);

#endif  // _MM_DISPATCH_HPP_
