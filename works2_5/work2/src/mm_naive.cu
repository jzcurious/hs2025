#include "cuda_grid_heuristics.hpp"
#include "work2/kernel_mm_naive.cuh"
#include "work2/matrix_view.hpp"  // IWYU pragma: keep
#include "work2/mm_naive.hpp"

template <MatrixKind MatrixT>
void w2::matmul(const MatrixT a, const MatrixT b, MatrixT c) {
  constexpr dim3 block_size = {16, 16};

  auto grid_size_by_axis
      = [](std::uint32_t matrix_size_by_axis, std::uint32_t block_size_by_axis) {
          return (matrix_size_by_axis + block_size_by_axis - 1) / block_size_by_axis;
        };

  const dim3 grid_size = {
      grid_size_by_axis(b.size(1), block_size.x),
      grid_size_by_axis(a.size(0), block_size.y),
  };

  // MatrixT d_a, d_b, d_c;
  // cudaMalloc();

  // kernel_mm_naive<MatrixT><<<grid_size, block_size>>>(a, b, c);
}

template void w2::matmul<MatrixView<float>>(
    const MatrixView<float> a, const MatrixView<float> b, MatrixView<float> c);
