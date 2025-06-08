#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include "work2/matrix/devblock.hpp"
#include "work2/matrix/matrix_expr.hpp"
#include "work2/matrix/matrix_expr_kind.hpp"
#include "work2/matrix/matrix_ops.hpp"
#include "work2/matrix/matrix_view.cuh"
#include "work2/mm_impls/op_bundle_kind.hpp"

template <template <class> class OpBundleT, ScalarKind ScalarT>
  requires OpBundleKind<OpBundleT, ScalarT>
class DeviceMatrix final {
 private:
  DeviceBlock<ScalarT> _block;
  MatrixView<ScalarT> _view;

  template <class FuncT>
  using MatrixExprBinary
      = MatrixExpr<FuncT, MatrixView<ScalarT>, MatrixView<ScalarT>, MatrixView<ScalarT>>;

  DeviceMatrix(const MatrixView<ScalarT>& view)
      : _block((view.mrows() + view.vpad()) * (view.ncols() + view.hpad()))
      , _view(view) {
    if (_view.vpad() or _view.hpad()) _block.memset(0);
  }

 public:
  DeviceMatrix(std::uint32_t mrows, std::uint32_t ncols, MatrixOps ops = MatrixOps{})
      : _block((mrows + ops.vpad_) * (ncols + ops.hpad_))
      , _view(_block, mrows, ncols, ops.colmajor_, ops.vpad_, ops.hpad_) {
    if (ops.vpad_ or ops.hpad_) {
      _block.memset(0);  // NOTE: It's stupid, but I'm too lazy to do it any other way.
                         // You can fix it.
    }
    if (ops.src_) copy_data_from_host(ops.src_);
  }

  DeviceMatrix(DeviceMatrix&& matrix)
      : _block(std::move(matrix._block))
      , _view(matrix._view) {}

  DeviceMatrix(const DeviceMatrix& matrix) = delete;

  template <MatrixExprKind ExprT>
  DeviceMatrix(ExprT&& expr)
      : DeviceMatrix(expr.required_result_view()) {
    expr.eval(_view);
  }

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

  auto operator*(const DeviceMatrix& matrix) const {
    MatrixView<ScalarT> result_view(nullptr,
        size(0),
        matrix.size(1),
        _view.colmajor,
        _view.vpad(),
        matrix._view.hpad());

    using multiplies_expr_t
        = MatrixExprBinary<std::decay_t<decltype(OpBundleT<ScalarT>::multiplies)>>;

    return multiplies_expr_t(
        OpBundleT<ScalarT>::multiplies, result_view, _view, matrix._view);
  }
};

#endif  // _MATRIX_HPP_
