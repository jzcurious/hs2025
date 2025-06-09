#ifndef _OP_BUNDLE_HPP_
#define _OP_BUNDLE_HPP_

#include "work2/matrix/matrix_view.cuh"

#define BUNDLE_REGISTER_NOT_IMPLEMENTED_BINARY(name)                                     \
  static MatrixView<ScalarT>& name(MatrixView<ScalarT>& c,                               \
      const MatrixView<ScalarT>& a,                                                      \
      const MatrixView<ScalarT>& b) {                                                    \
    static_assert(sizeof(ScalarT) == 0, "This method is not implemented");               \
    return c;                                                                            \
  }

template <class ScalarT>
struct OpBundle {
  struct op_impl_feature_t {};

  using scalar_t = ScalarT;

  using result_t = MatrixView<ScalarT>;

  BUNDLE_REGISTER_NOT_IMPLEMENTED_BINARY(add);
  BUNDLE_REGISTER_NOT_IMPLEMENTED_BINARY(sub);
  BUNDLE_REGISTER_NOT_IMPLEMENTED_BINARY(mul);
  BUNDLE_REGISTER_NOT_IMPLEMENTED_BINARY(div);
  BUNDLE_REGISTER_NOT_IMPLEMENTED_BINARY(matmul);
};

#undef BUNDLE_REGISTER_NOT_IMPLEMENTED_BINARY

#define BUNDLE_REGISTER_IMPLEMENTED_BINARY(name, impl)                                   \
  static MatrixView<ScalarT>& name(MatrixView<ScalarT>& c,                               \
      const MatrixView<ScalarT>& a,                                                      \
      const MatrixView<ScalarT>& b) {                                                    \
    return impl(c, a, b);                                                                \
  }

#define MAKE_BUNDLE(name)                                                                \
  template <ScalarKind ScalarT>                                                          \
  struct name final : OpBundle<ScalarT>

#endif  // _OP_BUNDLE_HPP_
