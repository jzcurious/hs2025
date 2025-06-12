#include "work4/naive_linear.hpp"

template <MatrixKind MatrixT>
MatrixT w4::naive_linear(const MatrixT& x, const MatrixT& w, const MatrixT& b) {
  return x * w + b;
}
