#ifndef _MM_NAIVE_HPP_
#define _MM_NAIVE_HPP_

#include "work2/matrix/matrix.cuh"

namespace w2 {

template <ScalarKind ScalarT>
void matmul_naive(const DeviceMatrix<ScalarT>& a,
    const DeviceMatrix<ScalarT>& b,
    DeviceMatrix<ScalarT>& c);

}  // namespace w2

namespace mm_naive_impl {

template <ScalarKind ScalarT>
DeviceMatrix<ScalarT> operator*(
    const DeviceMatrix<ScalarT>& a, const DeviceMatrix<ScalarT>& b) {
  DeviceMatrix<ScalarT> c(a.size(0), b.size(1), a.colmajor);
  w2::matmul_naive(a, b, c);
  return c;
}

}  // namespace mm_naive_impl

#endif  // _MM_NAIVE_HPP_
