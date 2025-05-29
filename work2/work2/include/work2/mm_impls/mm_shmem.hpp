#ifndef _MM_SHMEM_HPP_
#define _MM_SHMEM_HPP_

#include "work2/matrix/matrix.cuh"

namespace w2 {

template <ScalarKind ScalarT>
void matmul_shmem(const DeviceMatrix<ScalarT>& a,
    const DeviceMatrix<ScalarT>& b,
    DeviceMatrix<ScalarT>& c);

}  // namespace w2

#endif  // _MM_SHMEM_HPP_
