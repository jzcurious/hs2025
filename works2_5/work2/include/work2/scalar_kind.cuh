#ifndef _SCALAR_KIND_CUH_
#define _SCALAR_KIND_CUH_

#include <concepts>  // IWYU pragma: keep
#include <cuda_fp16.h>

template <class T>
concept ScalarKind = std::is_arithmetic_v<T> or std::is_same_v<T, half>;

#endif  // _SCALAR_KIND_CUH_
