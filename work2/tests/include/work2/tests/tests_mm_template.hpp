#ifndef TESTS_MM_TEMPLATE_HPP
#define TESTS_MM_TEMPLATE_HPP

#include "work2/matrix/matrix.hpp"
#include "work2/matrix/matrix_operators.hpp"  // IWYU pragma: keep
#include "work2/matrix/scalar_kind.hpp"
#include "work2/mm_impls/op_bundle_kind.hpp"
#include "work2/tests/matrix_test_helpers.hpp"

using MatMulTestParams = MatrixTestParams;

template <template <typename> class OpBundleT, ScalarKind ScalarT>
  requires OpBundleKind<OpBundleT, ScalarT>
class MatMulTest : public ::testing::TestWithParam<MatMulTestParams> {
 public:
  using params_tag_t = MatrixTestParamsTag;

 private:
  using matrix_t = DeviceMatrix<OpBundleT, ScalarT>;

  template <class EigenMatrixT>
  bool matmul_test_template_(const MatMulTestParams& params) {
    auto [_0, _1, colmajor, m, n, k, tm, tn, tol] = params;

    EigenMatrixT h_a = EigenMatrixT::Random(m, k);
    EigenMatrixT h_b = EigenMatrixT::Random(k, n);
    EigenMatrixT h_c = h_a * h_b;

    auto d_a = matrix_t(
        m, k, MatrixOptions{}.colmajor(colmajor).tile(tm, tn, m, k).src(h_a.data()));
    auto d_b = matrix_t(
        k, n, MatrixOptions{}.colmajor(colmajor).tile(tm, tn, k, n).src(h_b.data()));
    auto d_c = d_a * d_b;

    return is_approx_eigen(d_c, h_c, tol);
  }

 protected:
  DISPATCH_MATRIX_TEST_CASE(matmul_test_template_, matmul_test_, MatMulTestParams, 2);
};

#define INSTANTIATE_MM_TEST_SUITE_FOR_TYPE(impl_bundle, scalar_type, tile_size, tol)     \
  INSTANTIATE_MATRIX_TEST_SUITE_FOR_TYPE_WITH_DEFAULT_VALUES(                            \
      MMTests, MatMulTest, matmul_test_, impl_bundle, scalar_type, tile_size, tol)

#endif  // TESTS_MM_TEMPLATE_HPP
