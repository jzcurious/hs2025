#ifndef _DEVBLOCK_CUH_
#define _DEVBLOCK_CUH_

#include <cuda_runtime.h>

template <class T>
class DeviceBlock {
 public:
  const std::size_t size;

 private:
  T* _ptr;

 public:
  DeviceBlock(std::size_t numel)
      : size(numel * sizeof(T))
      , _ptr(nullptr) {
    cudaMalloc(&_ptr, size);
  }

  DeviceBlock(DeviceBlock&& block)
      : size(block.size)
      , _ptr(block._ptr) {
    block._ptr = nullptr;
  }

  DeviceBlock(const DeviceBlock&) = delete;
  DeviceBlock& operator=(const DeviceBlock&) = delete;

  void copy_from_host(const void* host_ptr) {
    cudaMemcpy(_ptr, host_ptr, size, cudaMemcpyHostToDevice);
  }

  void copy_from_device(const void* device_ptr) {
    cudaMemcpy(_ptr, device_ptr, size, cudaMemcpyDeviceToDevice);
  }

  void copy_to_host(void* host_ptr) const {
    cudaMemcpy(host_ptr, _ptr, size, cudaMemcpyDeviceToHost);
  }

  void copy_to_device(void* device_ptr) const {
    cudaMemcpy(device_ptr, _ptr, size, cudaMemcpyDeviceToDevice);
  }

  operator T*() {
    return _ptr;
  }

  ~DeviceBlock() {
    if (_ptr) cudaFree(_ptr);
  }
};

#endif  // _DEVBLOCK_CUH_
