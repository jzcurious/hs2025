#ifndef _MATRIX_VIEW_CUH_
#define _MATRIX_VIEW_CUH_

#include <concepts>
#include <cstdint>

// clang-format off
struct layout {
  struct rowmajor {};
  struct colmajor {};
};  // clang-format on

template <std::floating_point ScalarT>
class MatrixView final {
 private:
  ScalarT* _data;
  std::uint32_t _mrows;
  std::uint32_t _ncols;
  std::uint32_t _ldim;

 public:
  using scalar_t = ScalarT;

  struct matrix_feature {};

  const bool colmajor;

  __host__ __device__ MatrixView(
      ScalarT* data, std::uint32_t mrows, std::uint32_t ncols, bool colmajor = false)
      : _data(data)
      , _mrows(mrows)
      , _ncols(ncols)
      , _ldim(colmajor ? mrows : ncols)
      , colmajor(colmajor) {}

  __host__ __device__ MatrixView(
      ScalarT* data, std::uint32_t mrows, std::uint32_t ncols, layout::rowmajor)
      : MatrixView(data, mrows, ncols, false) {}

  __host__ __device__ MatrixView(
      ScalarT* data, std::uint32_t mrows, std::uint32_t ncols, layout::colmajor)
      : MatrixView(data, mrows, ncols, true) {}

  __host__ __device__ MatrixView(const MatrixView& view)
      : _data(view._data)
      , _mrows(view._mrows)
      , _ncols(view._ncols)
      , _ldim(view._ldim)
      , colmajor(view.colmajor) {}

  __host__ __device__ ScalarT& operator()(std::uint32_t row, std::uint32_t col) {
    return *(_data + (colmajor ? col * _ldim + row : row * _ldim + col));
  }

  __host__ __device__ ScalarT operator()(std::uint32_t row, std::uint32_t col) const {
    return *(_data + (colmajor ? col * _ldim + row : row * _ldim + col));
  }

  __host__ __device__ void transpose() {
    std::swap(_mrows, _ncols);
  }

  __host__ __device__ std::uint32_t size(std::uint8_t axis) const {
    return axis ? _ncols : _mrows;
  }

  __host__ __device__ std::uint32_t size() const {
    return _ncols * _mrows;
  }

  __host__ __device__ ScalarT* data() const {
    return _data;
  }

  __host__ __device__ std::uint32_t ldim() const {
    return _ldim;
  }
};

#endif  // _MATRIX_VIEW_CUH_
