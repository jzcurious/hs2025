#ifndef _ADD_HPP_
#define _ADD_HPP_

#include "work2/matrix/matrix_view.cuh"

namespace w4 {

template <ScalarKind ScalarT>
MatrixView<ScalarT>& add(
    MatrixView<ScalarT>& c, const MatrixView<ScalarT>& a, const MatrixView<ScalarT>& b);

}  // namespace w4

#endif  // _ADD_HPP_
