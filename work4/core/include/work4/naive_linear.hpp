#ifndef _NAIVE_LINEAR_HPP_
#define _NAIVE_LINEAR_HPP_

#include "work2/matrix/matrix_kind.hpp"

namespace w4 {

template <MatrixKind MatrixT>
MatrixT naive_linear(const MatrixT& x, const MatrixT& w, const MatrixT& b);

}  // namespace w4

#endif  // _NAIVE_LINEAR_HPP_
