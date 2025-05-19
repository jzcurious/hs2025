#ifndef _MM_SHMEM_HPP_
#define _MM_SHMEM_HPP_

#include "work2/matrix_kind.hpp"

namespace w3 {

template <MatrixKind MatrixT>
void matmul(const MatrixT& a, const MatrixT& b, MatrixT& c);

}  // namespace w3

#endif  // _MM_SHMEM_HPP_
