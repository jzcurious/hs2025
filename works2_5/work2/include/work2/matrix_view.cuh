#ifndef _MATRIX_VIEW_CUH_
#define _MATRIX_VIEW_CUH_

#include "work2/scalar_kind.cuh"

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

  struct matrix_feature {};

  const bool colmajor;

  __host__ __device__ MatrixView(ScalarT* data,
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

  __host__ __device__ MatrixView(ScalarT* data,
      std::uint32_t mrows,
      std::uint32_t ncols,
      layout::rowmajor,
      std::uint32_t row_pad = 0,
      std::uint32_t col_pad = 0)
      : MatrixView(data, mrows, ncols, false, row_pad, col_pad) {}

  __host__ __device__ MatrixView(ScalarT* data,
      std::uint32_t mrows,
      std::uint32_t ncols,
      layout::colmajor,
      std::uint32_t row_pad = 0,
      std::uint32_t col_pad = 0)
      : MatrixView(data, mrows, ncols, true, row_pad, col_pad) {}

  __host__ __device__ MatrixView(const MatrixView& view)
      : _data(view._data)
      , _mrows(view._mrows)
      , _ncols(view._ncols)
      , _ldim(view._ldim)
      , _row_pad(view._row_pad)
      , _col_pad(view._col_pad)
      , colmajor(view.colmajor) {}

  __host__ __device__ void transpose() {
    std::swap(_mrows, _ncols);
    std::swap(_row_pad, _col_pad);
  }

  __host__ __device__ std::uint32_t size(std::uint8_t axis) const {
    return axis ? _ncols : _mrows;
  }

  __host__ __device__ std::uint32_t size() const {
    return _ncols * _mrows;
  }

  __host__ __device__ std::uint32_t pad(std::uint8_t axis) const {
    return axis ? _col_pad : _row_pad;
  }

  __host__ __device__ ScalarT* data() const {
    return _data;
  }

  __host__ __device__ ScalarT* data(std::uint32_t row, std::uint32_t col) const {
    return _data + (colmajor ? col * _ldim + row : row * _ldim + col);
  }

  __host__ __device__ ScalarT& operator()(std::uint32_t row, std::uint32_t col) {
    return *(data(row, col));
  }

  __host__ __device__ ScalarT operator()(std::uint32_t row, std::uint32_t col) const {
    return *(data(row, col));
  }

  __host__ __device__ std::uint32_t ldim() const {
    return _ldim;
  }
};

#endif  // _MATRIX_VIEW_CUH_
