#ifndef _DEVBLOCK_HPP_
#define _DEVBLOCK_HPP_

#include <cuda_runtime.h>

template <class T>
class DeviceBlock {
 public:
  const std::size_t size;
  const std::size_t item_size;

 private:
  T* _ptr;

 public:
  DeviceBlock(std::size_t numel)
      : size(numel * sizeof(T))
      , item_size(sizeof(T))
      , _ptr(nullptr) {
    cudaMalloc(&_ptr, size);
  }

  DeviceBlock(DeviceBlock&& block)
      : size(block.size)
      , item_size(block.item_size)
      , _ptr(block._ptr) {
    block._ptr = nullptr;
  }

  DeviceBlock(const DeviceBlock&) = delete;
  DeviceBlock& operator=(const DeviceBlock&) = delete;

  void memset(int value) {
    cudaMemset(_ptr, value, size);
  }

  void memset_2d(std::size_t pitch, int value, std::size_t width, std::size_t height) {
    cudaMemset2D(_ptr, pitch, value, item_size * width, height);
  }

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

  void copy_from_host_2d(const void* host_ptr,
      std::size_t dpitch,
      std::size_t spitch,
      std::size_t width,
      std::size_t height) {
    cudaMemcpy2D(_ptr,
        dpitch,
        host_ptr,
        spitch,
        item_size * width,
        height,
        cudaMemcpyHostToDevice);
  }

  void copy_from_device_2d(const void* device_ptr,
      std::size_t dpitch,
      std::size_t spitch,
      std::size_t width,
      std::size_t height) {
    cudaMemcpy2D(_ptr,
        dpitch,
        device_ptr,
        spitch,
        item_size * width,
        height,
        cudaMemcpyDeviceToDevice);
  }

  void copy_to_host_2d(void* host_ptr,
      std::size_t dpitch,
      std::size_t spitch,
      std::size_t width,
      std::size_t height) {
    cudaMemcpy2D(host_ptr,
        dpitch,
        _ptr,
        spitch,
        item_size * width,
        height,
        cudaMemcpyDeviceToHost);
  }

  void copy_to_device_2d(void* device_ptr,
      std::size_t dpitch,
      std::size_t spitch,
      std::size_t width,
      std::size_t height) {
    cudaMemcpy2D(device_ptr,
        dpitch,
        _ptr,
        spitch,
        item_size * width,
        height,
        cudaMemcpyDeviceToDevice);
  }

  operator T*() {
    return _ptr;
  }

  ~DeviceBlock() {
    if (_ptr) cudaFree(_ptr);
  }
};

#endif  // _DEVBLOCK_HPP_
