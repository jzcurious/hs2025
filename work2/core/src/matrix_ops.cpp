#include "work2/matrix/matrix_ops.hpp"

MatrixOps& MatrixOps::vpad(std::uint32_t pad) {
  vpad_ = pad;
  return *this;
}

MatrixOps& MatrixOps::hpad(std::uint32_t pad) {
  hpad_ = pad;
  return *this;
}

MatrixOps& MatrixOps::colmajor(bool val) {
  colmajor_ = val;
  return *this;
}

MatrixOps& MatrixOps::colmajor() {
  colmajor_ = true;
  return *this;
}

MatrixOps& MatrixOps::rowmajor() {
  colmajor_ = false;
  return *this;
}

MatrixOps& MatrixOps::src(void* host_ptr) {
  src_ = host_ptr;
  return *this;
}
