#ifndef _FUSED_LINEAR_HPP_
#define _FUSED_LINEAR_HPP_

#include "work2/matrix/matrix_kind.hpp"

namespace w4 {

template <MatrixKind MatrixT>
MatrixT fused_linear(const MatrixT& x, const MatrixT& w, const MatrixT& b);

}  // namespace w4

#endif  // _FUSED_LINEAR_HPP_
