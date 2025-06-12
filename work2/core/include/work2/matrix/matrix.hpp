#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include "work2/matrix/devblock.hpp"
#include "work2/matrix/matrix_options.hpp"
#include "work2/matrix/matrix_view.cuh"
#include "work2/mm_impls/op_bundle_kind.hpp"

template <template <class> class OpBundleT, ScalarKind ScalarT>
  requires OpBundleKind<OpBundleT, ScalarT>
class DeviceMatrix final {
 private:
  DeviceBlock<ScalarT> _block;
  MatrixView<ScalarT> _view;
  MatrixOptions _ops;

 public:
  struct matrix_f {};

  using scalar_t = ScalarT;

  DeviceMatrix(
      std::uint32_t mrows, std::uint32_t ncols, MatrixOptions ops = MatrixOptions{})
      : _block((ncols + ops.hpad_) * (mrows + ops.vpad_))
      , _view(_block, mrows, ncols, ops.colmajor_, ops.vpad_, ops.hpad_)
      , _ops(ops) {
    if (ops.vpad_ or ops.hpad_) {
      _block.memset(0);  // NOTE: It's stupid, but I'm too lazy to do it any other way.
                         // You can fix it.
    }
    if (ops.src_) copy_data_from_host(ops.src_);
  }

  DeviceMatrix(DeviceMatrix&& matrix)
      : _block(std::move(matrix._block))
      , _view(matrix._view)
      , _ops(matrix._ops) {}

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

  MatrixOptions ops() const {
    return _ops.like();
  }

  void copy_data_from_host(const void* host_ptr) {
    if (_view.hpad() and not _view.colmajor) {
      _block.copy_from_host_2d(
          host_ptr, _view.ldim(), _view.size(1), _view.size(1), _view.size(0));
      return;
    }

    if (_view.vpad() and _view.colmajor) {
      _block.copy_from_host_2d(
          host_ptr, _view.ldim(), _view.size(0), _view.size(0), _view.size(1));
      return;
    }

    _block.copy_from_host(host_ptr, _view.numel());
  }

  void copy_data_to_host(void* host_ptr) const {
    if (_view.hpad() and not _view.colmajor) {
      _block.copy_to_host_2d(
          host_ptr, _view.size(1), _view.ldim(), _view.size(1), _view.size(0));
      return;
    }

    if (_view.vpad() and _view.colmajor) {
      _block.copy_to_host_2d(
          host_ptr, _view.size(0), _view.ldim(), _view.size(0), _view.size(1));
      return;
    }

    _block.copy_to_host(host_ptr, _view.numel());
  }
};

#endif  // _MATRIX_HPP_
