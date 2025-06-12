#include "work4/fused_linear.hpp"

template <MatrixKind MatrixT>
MatrixT w4::fused_linear(const MatrixT& x, const MatrixT& w, const MatrixT& b) {
  return x * w + b;  // TODO: replace to kernel call
}
