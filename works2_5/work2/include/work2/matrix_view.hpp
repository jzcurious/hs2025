#ifndef _MATRIX_VIEW_HPP_
#define _MATRIX_VIEW_HPP_

#include <concepts>
#include <cstdint>  // IWYU pragma: keep

template <std::floating_point ScalarT>
class MatrixView final {
 private:
  class Impl;
  Impl* _pimpl;

 public:
  struct matrix_feature {};

  MatrixView(ScalarT* data, std::size_t mrows, std::size_t ncols) {
    _pimpl = new Impl(data, mrows, ncols);
  }

  ~MatrixView() {
    delete _pimpl;
  }

  ScalarT& operator()(std::size_t row, std::size_t col) {
    return (*_pimpl)(row, col);
  }

  ScalarT operator()(std::size_t row, std::size_t col) const {
    return (*_pimpl)(row, col);
  }

  void transpose() {
    _pimpl->transpose();
  }

  std::size_t size(std::uint8_t axis) const {
    return _pimpl->size(axis);
  }
};

#endif  // _MATRIX_VIEW_HPP_
