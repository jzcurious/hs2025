#ifndef _MATRIX_CUH_
#define _MATRIX_CUH_

#include "work2/matrix/devblock.hpp"
#include "work2/matrix/matrix_ops.hpp"
#include "work2/matrix/matrix_view.cuh"
#include "work2/mm_impls/op_impl_bundle_kind.hpp"

template <template <class> class OpImplBundleT, ScalarKind ScalarT>
  requires OpImplBundleKind<OpImplBundleT, ScalarT>
class DeviceMatrix final {
 private:
  DeviceBlock<ScalarT> _block;
  MatrixView<ScalarT> _view;
  MatrixOps _ops;

 public:
  DeviceMatrix(std::uint32_t mrows, std::uint32_t ncols, MatrixOps ops = MatrixOps{})
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

  void copy_data_to_host(void* host_ptr) {
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

  DeviceMatrix operator*(const DeviceMatrix& matrix) const {
    /* NOTE: You can add a check for matrix commutativity (`size(1) == matrix.size(0)`),
     * but I'm too lazy to do it. */

    auto result_mrows = size(0);
    auto result_ncols = matrix.size(1);
    auto result_ops = _ops.like().vpad(_view.vpad()).hpad(matrix._view.hpad());

    auto result = DeviceMatrix(result_mrows, result_ncols, result_ops);

    OpImplBundleT<ScalarT>::multiplies(_view, matrix._view, result._view);
    return result;
  }
};

#endif  // _MATRIX_CUH_
