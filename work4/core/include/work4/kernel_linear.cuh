#ifndef _KERNEL_LINEAR_CUH_
#define _KERNEL_LINEAR_CUH_

#include "work2/matrix/matrix_view_kind.hpp"

#include <cstdint>
#include <mma.h>

using namespace nvcuda;

template <MatrixViewKind MatrixT,
    bool colmajor = false,
    std::uint32_t wmma_m = 16,
    std::uint32_t wmma_n = 16,
    std::uint32_t wmma_k = 16>
__global__ void kernel_linear(
    MatrixT y, const MatrixT x, const MatrixT w, const MatrixT b) {
  using scalar_t = typename MatrixT::scalar_t;

  using layout_t_ = std::conditional_t<colmajor, wmma::col_major, wmma::row_major>;

  wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, scalar_t, layout_t_> fx;
  wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, scalar_t, layout_t_> fw;
  wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, scalar_t> fy;

  wmma::fill_fragment(fy, 0.0f);

  std::uint32_t warp_x = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  std::uint32_t warp_y = (blockIdx.y * blockDim.y + threadIdx.y);

  std::uint32_t i = warp_y * wmma_m;
  std::uint32_t j = warp_x * wmma_n;

  std::uint32_t k = x.size(1);

  for (std::uint32_t v = 0; v < k; v += wmma_k) {
    wmma::load_matrix_sync(fx, x.data(i, v), x.ldim());
    wmma::load_matrix_sync(fw, w.data(v, j), w.ldim());
    wmma::mma_sync(fy, fx, fw, fy);
  }

  wmma::store_matrix_sync(
      y.data(i, j), fy, y.ldim(), colmajor ? wmma::mem_col_major : wmma::mem_row_major);

  if ((threadIdx.x < wmma_n) and (j + threadIdx.x < y.size(1)))
    for (std::uint32_t t = i; (t < y.size(0)) and (t < i + wmma_m); ++t)
      y(t, j + threadIdx.x) += b(0, j + threadIdx.x);
}

#endif  // _KERNEL_LINEAR_CUH_
