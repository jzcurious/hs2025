#ifndef _MM_WMMA_HPP_
#define _MM_WMMA_HPP_

#include "work2/matrix_kind.hpp"

namespace w4 {

template <MatrixKind MatrixT>
void matmul(const MatrixT& a, const MatrixT& b, MatrixT& c);

}  // namespace w4

#endif  // _MM_WMMA_HPP_
