#include "work4/add.hpp"
#include "work4/kernel_add.cuh"
#include "work4/kernel_broadcast_add.cuh"

#include "cudagh.hpp"
#include <cuda_fp16.h>

template <ScalarKind ScalarT>
MatrixView<ScalarT>& w4::add(
    MatrixView<ScalarT>& c, const MatrixView<ScalarT>& a, const MatrixView<ScalarT>& b) {
  constexpr dim3 block_size = {16, 16};

  const dim3 grid_size = {
      cudagh::cover(a.size(1), block_size.x),
      cudagh::cover(a.size(0), block_size.y),
  };

  if (a.size(0) > b.size(0))
    kernel_broadcast_add<<<grid_size, block_size>>>(c, a, b);
  else
    kernel_add<<<grid_size, block_size>>>(c, a, b);

  return c;
}

#define ADD_DISPATCH(scalar_type)                                                        \
  template MatrixView<scalar_type>& w4::add(MatrixView<scalar_type>& c,                  \
      const MatrixView<scalar_type>& a,                                                  \
      const MatrixView<scalar_type>& b)

ADD_DISPATCH(float);
ADD_DISPATCH(double);
ADD_DISPATCH(half);
