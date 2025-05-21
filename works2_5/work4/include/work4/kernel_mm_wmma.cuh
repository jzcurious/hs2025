#ifndef _KERNEL_MM_WMMA_CUH_
#define _KERNEL_MM_WMMA_CUH_

#include "work2/matrix_kind.hpp"

#include <cstdint>
#include <mma.h>

using namespace nvcuda;

template <MatrixKind MatrixT,
    std::uint32_t wmma_m = 16,
    std::uint32_t wmma_n = 16,
    std::uint32_t wmma_k = 16>
__global__ void kernel_mm_wmma(const MatrixT a, const MatrixT b, MatrixT c) {
  using scalar_t = typename MatrixT::scalar_t;

  wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, scalar_t, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, scalar_t, wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, scalar_t> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  std::uint32_t warp_x = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  std::uint32_t warp_y = (blockIdx.y * blockDim.y + threadIdx.y);

  std::uint32_t i = warp_y * wmma_m;
  std::uint32_t j = warp_x * wmma_n;

  for (int v = 0; v < a.size(1); v += wmma_k) {
    scalar_t* ptr_a = a.data() + a.ldim() * i + v;
    scalar_t* ptr_b = b.data() + b.ldim() * v + j;

    wmma::load_matrix_sync(a_frag, ptr_a, a.ldim());
    wmma::load_matrix_sync(b_frag, ptr_b, b.ldim());

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  scalar_t* ptr_c = c.data() + c.ldim() * i + j;
  wmma::store_matrix_sync(ptr_c, c_frag, c.ldim(), wmma::mem_row_major);
}

#endif  // _KERNEL_MM_WMMA_CUH_
