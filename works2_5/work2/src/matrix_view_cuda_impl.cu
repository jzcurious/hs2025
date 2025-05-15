#include "work2/matrix_view.hpp"

#include <concepts>
#include <cstdint>
#include <cuda_runtime.h>

template <std::floating_point ScalarT>
class MatrixView<ScalarT>::Impl final {
 private:
  ScalarT* _data;
  std::uint32_t _mrows;
  std::uint32_t _ncols;
  std::uint32_t _ldim;

 public:
  struct matrix_feature {};

  __host__ __device__ Impl(ScalarT* data, std::uint32_t mrows, std::uint32_t ncols)
      : _data(data)
      , _mrows(mrows)
      , _ncols(ncols)
      , _ldim(ncols) {}

  __host__ __device__ Impl(const Impl& impl)
      : _data(impl._data)
      , _mrows(impl._mrows)
      , _ncols(impl._ncols)
      , _ldim(impl._ncols) {}

  __host__ __device__ ScalarT& operator()(std::uint32_t row, std::uint32_t col) {
    return *(_data + row * _ldim + col);
  }

  __host__ __device__ ScalarT operator()(std::uint32_t row, std::uint32_t col) const {
    return *(_data + row * _ldim + col);
  }

  __host__ __device__ void transpose() {
    std::swap(_mrows, _ncols);
  }

  __host__ __device__ std::uint32_t size(std::uint8_t axis) const {
    return axis ? _ncols : _mrows;
  }
};

template <std::floating_point ScalarT>
MatrixView<ScalarT>::MatrixView(ScalarT* data, std::uint32_t mrows, std::uint32_t ncols)
    : _pimpl(std::make_unique<Impl>(data, mrows, ncols)) {}

template <std::floating_point ScalarT>
MatrixView<ScalarT>::MatrixView(const MatrixView& matrix_view)
    : _pimpl(std::make_unique<Impl>(*matrix_view._pimpl)) {}

template <std::floating_point ScalarT>
MatrixView<ScalarT>::~MatrixView() = default;

template <std::floating_point ScalarT>
ScalarT& MatrixView<ScalarT>::operator()(std::uint32_t row, std::uint32_t col) {
  return (*_pimpl)(row, col);
}

template <std::floating_point ScalarT>
ScalarT MatrixView<ScalarT>::operator()(std::uint32_t row, std::uint32_t col) const {
  return (*_pimpl)(row, col);
}

template <std::floating_point ScalarT>
void MatrixView<ScalarT>::transpose() {
  _pimpl->transpose();
}

template <std::floating_point ScalarT>
std::uint32_t MatrixView<ScalarT>::size(std::uint8_t axis) const {
  return _pimpl->size(axis);
}

template class MatrixView<float>;
