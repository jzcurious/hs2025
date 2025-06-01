#ifndef _MATRIX_OPS_HPP_
#define _MATRIX_OPS_HPP_

#include <cstdint>

struct MatrixOps final {
  std::uint32_t vpad_ = 0;
  std::uint32_t hpad_ = 0;
  bool colmajor_ = false;
  void* src_ = nullptr;

  MatrixOps& vpad(std::uint32_t pad);
  MatrixOps& hpad(std::uint32_t pad);
  MatrixOps& colmajor(bool val);
  MatrixOps& colmajor();
  MatrixOps& rowmajor();
  MatrixOps& src(void* host_ptr);
};

#endif  // _MATRIX_OPS_HPP_
