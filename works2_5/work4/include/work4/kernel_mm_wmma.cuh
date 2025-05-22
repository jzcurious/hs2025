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

  wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, scalar_t, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, scalar_t, wmma::row_major> fb;
  wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, scalar_t> fc;

  wmma::fill_fragment(fc, 0.0f);

  std::uint32_t warp_x = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  std::uint32_t warp_y = (blockIdx.y * blockDim.y + threadIdx.y);

  std::uint32_t i = warp_y * wmma_m;
  std::uint32_t j = warp_x * wmma_n;

  std::uint32_t m = a.size(0);
  std::uint32_t n = b.size(1);
  std::uint32_t k = a.size(1);

  std::uint32_t laneid_x = threadIdx.x & 0x1F;

  for (std::uint32_t v = 0; v < k; v += wmma_k) {
    if (i + wmma_m < m and v + wmma_n < n) {
      scalar_t* ptr_a = a.data() + a.ldim() * i + v;
      wmma::load_matrix_sync(fa, ptr_a, a.ldim());
    } else {
      wmma::fill_fragment(fa, 0.0f);
      if (laneid_x < wmma_n) {
        for (std::uint32_t r = 0; r < wmma_m * wmma_n; r += wmma_n) {
          fa.x[laneid_x + r] = a(i + r, laneid_x + v);
        }
      }
    }

    if (j + wmma_n < n and v + wmma_m < m) {
      scalar_t* ptr_b = b.data() + b.ldim() * v + j;
      wmma::load_matrix_sync(fb, ptr_b, b.ldim());
    } else {
      wmma::fill_fragment(fb, 0.0f);
      if (laneid_x < wmma_n) {
        for (std::uint32_t r = 0; r < wmma_m * wmma_n; r += wmma_n) {
          fb.x[laneid_x + r] = b(laneid_x + v, j + r);
        }
      }
    }

    wmma::mma_sync(fc, fa, fb, fc);
  }

  scalar_t* ptr_c = c.data() + c.ldim() * i + j;
  wmma::store_matrix_sync(ptr_c, fc, c.ldim(), wmma::mem_row_major);
}

#endif  // _KERNEL_MM_WMMA_CUH_
