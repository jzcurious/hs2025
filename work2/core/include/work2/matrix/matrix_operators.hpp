#ifndef _MATRIX_OPERATORS_HPP_
#define _MATRIX_OPERATORS_HPP_

#include "work2/matrix/matrix.hpp"

template <template <class> class OpBundleT, ScalarKind ScalarT>
DeviceMatrix<OpBundleT, ScalarT> operator*(const DeviceMatrix<OpBundleT, ScalarT>& a,
    const DeviceMatrix<OpBundleT, ScalarT>& b) {
  /* NOTE: You can add a check for matrix commutativity (`size(1) == matrix.size(0)`),
   * but I'm too lazy to do it. */

  auto c = DeviceMatrix<OpBundleT, ScalarT>(
      a.size(0), b.size(1), a.ops().hpad(b.view().hpad()));
  OpBundleT<ScalarT>::matmul(c.view(), a.view(), b.view());
  return c;
}

template <template <class> class OpBundleT, ScalarKind ScalarT>
DeviceMatrix<OpBundleT, ScalarT> operator+(const DeviceMatrix<OpBundleT, ScalarT>& a,
    const DeviceMatrix<OpBundleT, ScalarT>& b) {

  if (a.size(0) == 1) return b + a;

  auto c = DeviceMatrix<OpBundleT, ScalarT>(a.size(0), a.size(1));
  OpBundleT<ScalarT>::add(c.view(), a.view(), b.view());
  return c;
}

#endif  // _MATRIX_OPERATORS_HPP_
