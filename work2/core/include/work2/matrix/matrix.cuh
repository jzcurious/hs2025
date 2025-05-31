#ifndef _MATRIX_CUH_
#define _MATRIX_CUH_

#include "work2/matrix/devblock.cuh"
#include "work2/matrix/matrix_view.cuh"

#include "work2/mm_impls/op_impl_bundle_kind.hpp"

template <template <class> class OpImplBundleT, ScalarKind ScalarT>
  requires OpImplBundleKind<OpImplBundleT, ScalarT>
class DeviceMatrix final {
 private:
  DeviceBlock<ScalarT> _block;
  MatrixView<ScalarT> _view;

 public:
  DeviceMatrix(std::uint32_t mrows,
      std::uint32_t ncols,
      bool colmajor = false,
      std::uint32_t row_pad = 0,
      std::uint32_t col_pad = 0)
      : _block((ncols + col_pad) * (mrows + row_pad))
      , _view(_block, mrows, ncols, colmajor, row_pad, col_pad) {}

  DeviceMatrix(DeviceMatrix&& matrix)
      : _block(std::move(matrix._block))
      , _view(matrix._view) {}

  DeviceMatrix(const DeviceMatrix& matrix) = delete;
  DeviceMatrix& operator=(const DeviceMatrix&) = delete;

  std::uint32_t size(std::uint8_t axis) const {
    return _view.size(axis);
  }

  std::uint32_t numel() const {
    return _view.numel();
  }

  ScalarT& operator()(std::uint32_t i, std::uint32_t j) {
    return _view(i, j);
  }

  ScalarT operator()(std::uint32_t i, std::uint32_t j) const {
    return _view(i, j);
  }

  const MatrixView<ScalarT>& view() const {
    return _view;
  }

  MatrixView<ScalarT>& view() {
    return _view;
  }

  const DeviceBlock<ScalarT>& block() const {
    return _block;
  }

  DeviceBlock<ScalarT>& block() {
    return _block;
  }

  DeviceMatrix operator*(const DeviceMatrix& matrix) const {
    // NOTE: You could add a check for commutativity of matrices.
    auto result = DeviceMatrix(size(0), matrix.size(1), _view.colmajor);
    OpImplBundleT<ScalarT>::multiplies(_view, matrix._view, result._view);
    return result;
  }
};

#endif  // _MATRIX_CUH_
