#ifndef _MATRIX_VIEW_CUH_
#define _MATRIX_VIEW_CUH_

#include "work2/matrix/hd_attr.cuh"
#include "work2/matrix/scalar_kind.cuh"

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
  std::uint32_t _row_pad;
  std::uint32_t _col_pad;

 public:
  using scalar_t = ScalarT;

  const bool colmajor;

  __hd__ MatrixView(ScalarT* data,
      std::uint32_t mrows,
      std::uint32_t ncols,
      bool colmajor = false,
      std::uint32_t row_pad = 0,
      std::uint32_t col_pad = 0)
      : _data(data)
      , _mrows(mrows)
      , _ncols(ncols)
      , _ldim(colmajor ? mrows + row_pad : ncols + col_pad)
      , _row_pad(row_pad)
      , _col_pad(col_pad)
      , colmajor(colmajor) {}

  __hd__ MatrixView(ScalarT* data,
      std::uint32_t mrows,
      std::uint32_t ncols,
      layout::rowmajor,
      std::uint32_t row_pad = 0,
      std::uint32_t col_pad = 0)
      : MatrixView(data, mrows, ncols, false, row_pad, col_pad) {}

  __hd__ MatrixView(ScalarT* data,
      std::uint32_t mrows,
      std::uint32_t ncols,
      layout::colmajor,
      std::uint32_t row_pad = 0,
      std::uint32_t col_pad = 0)
      : MatrixView(data, mrows, ncols, true, row_pad, col_pad) {}

  __hd__ MatrixView(const MatrixView& view)
      : _data(view._data)
      , _mrows(view._mrows)
      , _ncols(view._ncols)
      , _ldim(view._ldim)
      , _row_pad(view._row_pad)
      , _col_pad(view._col_pad)
      , colmajor(view.colmajor) {}

  __hd__ std::uint32_t size(std::uint8_t axis) const {
    return axis ? _ncols : _mrows;
  }

  __hd__ std::uint32_t numel() const {
    return _ncols * _mrows;
  }

  __hd__ std::uint32_t pad(std::uint8_t axis) const {
    return axis ? _col_pad : _row_pad;
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
