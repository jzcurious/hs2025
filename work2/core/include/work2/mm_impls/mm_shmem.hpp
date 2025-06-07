#ifndef _MM_SHMEM_HPP_
#define _MM_SHMEM_HPP_

#include "work2/matrix/matrix_view.cuh"

namespace w2 {

template <ScalarKind ScalarT>
MatrixView<ScalarT>& matmul_shmem(
    MatrixView<ScalarT>& c, const MatrixView<ScalarT>& a, const MatrixView<ScalarT>& b);

}  // namespace w2

#endif  // _MM_SHMEM_HPP_
