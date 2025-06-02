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

MatrixOps& MatrixOps::tile(std::uint32_t tile_mrows,
    std::uint32_t tile_ncols,
    std::uint32_t mrows,
    std::uint32_t ncols) {

  tile_mrows_ = tile_mrows;
  tile_ncols_ = tile_ncols;

  auto calc_padding = [](std::uint32_t matrix_size, std::uint32_t tile_size) {
    return (matrix_size + tile_size - 1) / tile_size * tile_size - matrix_size;
  };

  hpad_ = calc_padding(ncols, tile_ncols);
  vpad_ = calc_padding(mrows, tile_mrows);

  return *this;
}

MatrixOps MatrixOps::like() const {
  return MatrixOps{*this}.src(nullptr);
}
