#ifndef _MATRIX_VIEW_CUH_
#define _MATRIX_VIEW_CUH_

#include <concepts>
#include <cstdint>

template <std::floating_point ScalarT>
class MatrixView final {
 private:
  ScalarT* _data;
  std::uint32_t _mrows;
  std::uint32_t _ncols;
  std::uint32_t _ldim;

 public:
  struct matrix_feature {};

  __host__ __device__ MatrixView(ScalarT* data, std::uint32_t mrows, std::uint32_t ncols)
      : _data(data)
      , _mrows(mrows)
      , _ncols(ncols)
      , _ldim(ncols) {}

  __host__ __device__ MatrixView(const MatrixView& view)
      : _data(view._data)
      , _mrows(view._mrows)
      , _ncols(view._ncols)
      , _ldim(view._ncols) {}

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

#endif  // _MATRIX_VIEW_CUH_
