#include "work2/matrix_view.hpp"

#include <concepts>
#include <cstdint>

template <std::floating_point ScalarT>
class MatrixView<ScalarT>::Impl final {
 private:
  ScalarT* _data;
  std::size_t _mrows;
  std::size_t _ncols;
  std::size_t _ldim;

 public:
  struct matrix_feature {};

  Impl(ScalarT* data, std::size_t mrows, std::size_t ncols)
      : _data(data)
      , _mrows(mrows)
      , _ncols(ncols)
      , _ldim(ncols) {}

  ScalarT& operator()(std::size_t row, std::size_t col) {
    return *(_data + row * _ldim + col);
  }

  ScalarT operator()(std::size_t row, std::size_t col) const {
    return *(_data + row * _ldim + col);
  }

  void transpose() {
    std::swap(_mrows, _ncols);
  }

  std::size_t size(std::uint8_t axis) const {
    return axis ? _ncols : _mrows;
  }
};

template class MatrixView<float>;
