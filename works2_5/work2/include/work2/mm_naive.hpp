#ifndef _MM_NAIVE_HPP_
#define _MM_NAIVE_HPP_

#include "work2/matrix_kind.hpp"

namespace w2 {

template <MatrixKind MatrixT>
void matmul(const MatrixT a, const MatrixT b, MatrixT c);

}  // namespace w2

#endif  // _MM_NAIVE_HPP_
