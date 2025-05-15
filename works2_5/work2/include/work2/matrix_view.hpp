#ifndef _MATRIX_VIEW_HPP_
#define _MATRIX_VIEW_HPP_

#include <concepts>
#include <cstdint>  // IWYU pragma: keep
#include <memory>

template <std::floating_point ScalarT>
class MatrixView final {
 private:
  class Impl;
  std::unique_ptr<Impl> _pimpl;

 public:
  struct matrix_feature {};

  MatrixView(ScalarT* data, std::uint32_t mrows, std::uint32_t ncols);
  MatrixView(const MatrixView& matrix_view);
  ~MatrixView();
  ScalarT& operator()(std::uint32_t row, std::uint32_t col);
  ScalarT operator()(std::uint32_t row, std::uint32_t col) const;
  void transpose();
  std::uint32_t size(std::uint8_t axis) const;
};

#endif  // _MATRIX_VIEW_HPP_
