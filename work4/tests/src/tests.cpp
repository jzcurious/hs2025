#include "work2/matrix/matrix.hpp"
#include "work2/matrix/matrix_operators.hpp"  // IWYU pragma: keep
#include "work2/mm_impls/op_bundle_kind.hpp"
#include "work2/tests/matrix_test_helpers.hpp"
#include "work4/default_bundle.hpp"
#include "work4/fused_linear_operator.hpp"
#include "work4/graph_linear_operator.hpp"
#include "work4/naive_linear_operator.hpp"

using LinearTestParams = MatrixTestParams;

template <template <typename> class OpBundleT, ScalarKind ScalarT>
  requires OpBundleKind<OpBundleT, ScalarT>
class LinearTest : public ::testing::TestWithParam<LinearTestParams> {
 private:
  using matrix_t = DeviceMatrix<OpBundleT, ScalarT>;
  using func_ptr_t = matrix_t (*)(const matrix_t&, const matrix_t&, const matrix_t&);

  template <func_ptr_t func, class EigenMatrixT>
  bool linear_test_template_(const LinearTestParams& params) {
    auto [_0, _1, colmajor, m, n, k, tm, tn, tol] = params;

    EigenMatrixT h_x = EigenMatrixT::Random(m, k);
    EigenMatrixT h_w = EigenMatrixT::Random(k, n);
    EigenMatrixT h_b = EigenMatrixT::Random(1, n);
    EigenMatrixT h_y = h_x * h_w + h_b.replicate(m, 1);

    auto d_x = matrix_t(
        m, k, MatrixOptions{}.colmajor(colmajor).tile(tm, tn, m, k).src(h_x.data()));

    auto d_w = matrix_t(
        k, n, MatrixOptions{}.colmajor(colmajor).tile(tm, tn, k, n).src(h_w.data()));

    auto d_b = matrix_t(
        1, n, MatrixOptions{}.colmajor(colmajor).tile(tm, tn, 1, n).src(h_b.data()));

    auto d_y = func(d_x, d_w, d_b);

    return is_approx_eigen(d_y, h_y, tol);
  }

  template <class EigenMatrixT>
  bool linear_test_template_naive_(const LinearTestParams& params) {
    return linear_test_template_<naive_linear_operator, EigenMatrixT>(params);
  }

  template <class EigenMatrixT>
  bool linear_test_template_fused_(const LinearTestParams& params) {
    return linear_test_template_<fused_linear_operator, EigenMatrixT>(params);
  }

  template <class EigenMatrixT>
  bool linear_test_template_grpah_(const LinearTestParams& params) {
    return linear_test_template_<graph_linear_operator, EigenMatrixT>(params);
  }

 protected:
  DISPATCH_MATRIX_TEST_CASE(
      linear_test_template_naive_, naive_linear_test_, LinearTestParams, 2);

  DISPATCH_MATRIX_TEST_CASE(
      linear_test_template_fused_, fused_linear_test_, LinearTestParams, 2);

  DISPATCH_MATRIX_TEST_CASE(
      linear_test_template_grpah_, graph_linear_test_, LinearTestParams, 2);
};

#define INSTANTIATE_LINEAR_TEST_SUITE_FOR_TYPE(impl_bundle, scalar_type, tile_size, tol) \
  INSTANTIATE_MATRIX_TEST_SUITE_FOR_TYPE_WITH_DEFAULT_VALUES(LinearTestsNaive,           \
      LinearTest,                                                                        \
      naive_linear_test_,                                                                \
      impl_bundle,                                                                       \
      scalar_type,                                                                       \
      tile_size,                                                                         \
      tol);                                                                              \
  INSTANTIATE_MATRIX_TEST_SUITE_FOR_TYPE_WITH_DEFAULT_VALUES(LinearTestsFused,           \
      LinearTest,                                                                        \
      fused_linear_test_,                                                                \
      impl_bundle,                                                                       \
      scalar_type,                                                                       \
      tile_size,                                                                         \
      tol);                                                                              \
  INSTANTIATE_MATRIX_TEST_SUITE_FOR_TYPE_WITH_DEFAULT_VALUES(LinearTestsGrpah,           \
      LinearTest,                                                                        \
      graph_linear_test_,                                                                \
      impl_bundle,                                                                       \
      scalar_type,                                                                       \
      tile_size,                                                                         \
      tol);

INSTANTIATE_LINEAR_TEST_SUITE_FOR_TYPE(OpBundleDefault, half, 16, 1e-2);

#if __CUDA_ARCH__ >= 800
INSTANTIATE_LINEAR_TEST_SUITE_FOR_TYPE(OpBundleDefault, float, 16, 1e-5);
#endif
