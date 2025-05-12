#include <concepts>

// TODO: quantized floating point types

template <std::floating_point ScalarT>
class MatrixView final {
 private:
  ScalarT* _data;

 public:
  const std::size_t mrows;
  const std::size_t ncols;

 public:
  MatrixView(ScalarT* data, std::size_t mrows, std::size_t ncols)
      : _data(data)
      , mrows(mrows)
      , ncols(ncols) {}

  // ScalarT& operator[](std::size_t row, std::size_t col) {
  //   return *(_data +);
  // }
};

template <bool transpose_a,
    bool transpose_b,
    bool transpose_c,
    std::floating_point ScalarT>
__global__ void kernel_mm_naive(const ScalarT* a,
    const ScalarT* b,
    ScalarT* c,
    const std::size_t m,
    const std::size_t k,
    const std::size_t n) {
  std::size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;

  double acc = 0;

  for (std::size_t t = 0; t < k; ++t) acc += a[i][t] * b[t][j];
  c[i][j] = acc;
}
