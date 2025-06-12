#ifndef _MATRIX_OPTIONS_HPP_
#define _MATRIX_OPTIONS_HPP_

#include <cstdint>

struct MatrixOptions final {
  std::uint32_t vpad_ = 0;
  std::uint32_t hpad_ = 0;
  std::uint32_t tile_mrows_ = 1;
  std::uint32_t tile_ncols_ = 1;
  bool colmajor_ = false;
  void* src_ = nullptr;

  MatrixOptions& vpad(std::uint32_t pad);
  MatrixOptions& hpad(std::uint32_t pad);
  MatrixOptions& colmajor(bool val);
  MatrixOptions& colmajor();
  MatrixOptions& rowmajor();
  MatrixOptions& src(void* host_ptr);

  MatrixOptions& tile(std::uint32_t tile_mrows,
      std::uint32_t tile_ncols,
      std::uint32_t mrows,
      std::uint32_t ncols);

  MatrixOptions like() const;
};

#endif  // _MATRIX_OPTIONS_HPP_
