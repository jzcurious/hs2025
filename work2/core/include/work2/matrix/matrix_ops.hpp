#ifndef _MATRIX_OPS_HPP_
#define _MATRIX_OPS_HPP_

#include <cstdint>

struct MatrixOps final {
  std::uint32_t vpad_ = 0;
  std::uint32_t hpad_ = 0;
  std::uint32_t tile_mrows_ = 1;
  std::uint32_t tile_ncols_ = 1;
  bool colmajor_ = false;
  void* src_ = nullptr;

  MatrixOps& vpad(std::uint32_t pad);
  MatrixOps& hpad(std::uint32_t pad);
  MatrixOps& colmajor(bool val);
  MatrixOps& colmajor();
  MatrixOps& rowmajor();
  MatrixOps& src(void* host_ptr);

  MatrixOps& tile(std::uint32_t tile_mrows,
      std::uint32_t tile_ncols,
      std::uint32_t mrows,
      std::uint32_t ncols);

  MatrixOps copy() const;
};

#endif  // _MATRIX_OPS_HPP_
