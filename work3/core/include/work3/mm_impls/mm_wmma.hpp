#ifndef _MM_WMMA_HPP_
#define _MM_WMMA_HPP_

#include "work2/matrix/matrix_view.cuh"

namespace w3 {

template <ScalarKind ScalarT>
MatrixView<ScalarT>& matmul_wmma(
    MatrixView<ScalarT>& c, const MatrixView<ScalarT>& a, const MatrixView<ScalarT>& b);

}  // namespace w3

#endif  // _MM_WMMA_HPP_
