#ifndef _MATRIX_VIEW_CUH_
#define _MATRIX_VIEW_CUH_

#include "work2/matrix/hd_attr.cuh"
#include "work2/matrix/scalar_kind.hpp"

#include <cstdint>

// clang-format off
struct layout {
  struct rowmajor {};
  struct colmajor {};
};  // clang-format on

template <ScalarKind ScalarT>
class MatrixView final {
 private:
  ScalarT* _data;
  std::uint32_t _mrows;
  std::uint32_t _ncols;
  std::uint32_t _ldim;
  std::uint32_t _vpad;
  std::uint32_t _hpad;

 public:
  using scalar_t = ScalarT;

  const bool colmajor;

  __hd__ MatrixView(ScalarT* data,
      std::uint32_t mrows,
      std::uint32_t ncols,
      bool colmajor = false,
      std::uint32_t vpad = 0,
      std::uint32_t hpad = 0)
      : _data(data)
      , _mrows(mrows)
      , _ncols(ncols)
      , _ldim(colmajor ? mrows + vpad : ncols + hpad)
      , _vpad(vpad)
      , _hpad(hpad)
      , colmajor(colmajor) {}

  __hd__ MatrixView(ScalarT* data,
      std::uint32_t mrows,
      std::uint32_t ncols,
      layout::rowmajor,
      std::uint32_t vpad = 0,
      std::uint32_t hpad = 0)
      : MatrixView(data, mrows, ncols, false, vpad, hpad) {}

  __hd__ MatrixView(ScalarT* data,
      std::uint32_t mrows,
      std::uint32_t ncols,
      layout::colmajor,
      std::uint32_t vpad = 0,
      std::uint32_t hpad = 0)
      : MatrixView(data, mrows, ncols, true, vpad, hpad) {}

  __hd__ MatrixView(const MatrixView& view)
      : _data(view._data)
      , _mrows(view._mrows)
      , _ncols(view._ncols)
      , _ldim(view._ldim)
      , _vpad(view._vpad)
      , _hpad(view._hpad)
      , colmajor(view.colmajor) {}

  __hd__ std::uint32_t size(std::uint8_t axis) const {
    return axis ? _ncols : _mrows;
  }

  __hd__ std::uint32_t numel() const {
    return _ncols * _mrows;
  }

  __hd__ std::uint32_t pad(std::uint8_t axis) const {
    return axis ? _hpad : _vpad;
  }

  __hd__ std::uint32_t vpad() const {
    return _vpad;
  }

  __hd__ std::uint32_t hpad() const {
    return _hpad;
  }

  __hd__ ScalarT* data() const {
    return _data;
  }

  __hd__ ScalarT* data(std::uint32_t i, std::uint32_t j) const {
    return _data + (colmajor ? j * _ldim + i : i * _ldim + j);
  }

  __hd__ ScalarT& operator()(std::uint32_t i, std::uint32_t j) {
    return *(data(i, j));
  }

  __hd__ ScalarT operator()(std::uint32_t i, std::uint32_t j) const {
    return *(data(i, j));
  }

  __hd__ std::uint32_t ldim() const {
    return _ldim;
  }
};

#endif  // _MATRIX_VIEW_CUH_
