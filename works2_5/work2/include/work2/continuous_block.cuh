#ifndef _CONTINUOUS_BLOCK_CUH_
#define _CONTINUOUS_BLOCK_CUH_

#include <cuda_runtime.h>

template <class T>
class ContinuousBlock {
 private:
  T* _ptr;

 public:
  ContinuousBlock(std::size_t numel)
      : _ptr(nullptr) {
    cudaMalloc(&_ptr, numel * sizeof(T));
  }

  operator T*() {
    return _ptr;
  }

  ~ContinuousBlock() {
    if (_ptr) cudaFree(_ptr);
  }
};

#endif  // _CONTINUOUS_BLOCK_CUH_
