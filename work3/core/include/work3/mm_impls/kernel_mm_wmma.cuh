#ifndef _KERNEL_MM_WMMA_CUH_
#define _KERNEL_MM_WMMA_CUH_

#include "work2/matrix/matrix_view_kind.hpp"

#include <cstdint>
#include <mma.h>

using namespace nvcuda;

template <MatrixViewKind MatrixT,
    bool colmajor = false,
    std::uint32_t wmma_m = 16,
    std::uint32_t wmma_n = 16,
    std::uint32_t wmma_k = 16>
__global__ void kernel_mm_wmma(MatrixT c, const MatrixT a, const MatrixT b) {
  using scalar_t = typename MatrixT::scalar_t;

  using layout_t_ = std::conditional_t<colmajor, wmma::col_major, wmma::row_major>;

  wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, scalar_t, layout_t_> fa;
  wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, scalar_t, layout_t_> fb;
  wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, scalar_t> fc;

  wmma::fill_fragment(fc, 0.0f);

  std::uint32_t warp_x = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  std::uint32_t warp_y = (blockIdx.y * blockDim.y + threadIdx.y);

  std::uint32_t i = warp_y * wmma_m;
  std::uint32_t j = warp_x * wmma_n;

  std::uint32_t k = a.size(1);

  for (std::uint32_t v = 0; v < k; v += wmma_k) {
    wmma::load_matrix_sync(fa, a.data(i, v), a.ldim());
    wmma::load_matrix_sync(fb, b.data(v, j), b.ldim());
    wmma::mma_sync(fc, fa, fb, fc);
  }

  wmma::store_matrix_sync(
      c.data(i, j), fc, c.ldim(), colmajor ? wmma::mem_col_major : wmma::mem_row_major);
}

#endif  // _KERNEL_MM_WMMA_CUH_
