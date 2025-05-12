#ifndef _MATRIX_VIEW_CUH_
#define _MATRIX_VIEW_CUH_

#include <cstdint>

template <class T>
concept MatrixKind = requires() { typename T::matrix_feature(); };

template <std::floating_point ScalarT>
class MatrixView final {
 private:
  ScalarT* _data;
  std::size_t _mrows;
  std::size_t _ncols;
  std::size_t _ldim;

 public:
  struct matrix_feature {};

  __host__ __device__ MatrixView(ScalarT* data, std::size_t mrows, std::size_t ncols)
      : _data(data)
      , _mrows(mrows)
      , _ncols(ncols)
      , _ldim(ncols) {}

  __host__ __device__ ScalarT& operator()(std::size_t row, std::size_t col) {
    return *(_data + row * _ldim + col);
  }

  __host__ __device__ ScalarT operator()(std::size_t row, std::size_t col) const {
    return *(_data + row * _ldim + col);
  }

  __host__ __device__ void transpose() {
    std::swap(_mrows, _ncols);
  }

  __host__ __device__ std::size_t size(std::uint8_t axis) const {
    return axis ? _ncols : _mrows;
  }
};

#endif  // _MATRIX_VIEW_CUH_
