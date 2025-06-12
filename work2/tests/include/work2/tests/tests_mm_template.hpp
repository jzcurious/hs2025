#ifndef TESTS_MM_TEMPLATE_HPP
#define TESTS_MM_TEMPLATE_HPP

#include <cuda_runtime.h>

#include "work2/matrix/matrix.hpp"
#include "work2/matrix/matrix_kind.hpp"
#include "work2/matrix/matrix_operators.hpp"  // IWYU pragma: keep
#include "work2/mm_impls/op_bundle_kind.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>

struct MatMulTestParamsTag {
  // NOTE: Oh, dumb gtest )
};

// clang-format off
using MatMulTestParams = std::tuple<
  MatMulTestParamsTag,  // tag
  std::string,          // blame
  bool,                 // colmajor
  std::uint32_t,        // n
  std::uint32_t,        // m
  std::uint32_t,        // k
  std::uint32_t,        // tn
  std::uint32_t,        // tm
  float                 // tol
>;
// clang-format on

inline void PrintTo(const MatMulTestParams& params, std::ostream* os) {
  auto [_, blame, colmajor, m, n, k, tm, tn, tol] = params;
  *os << blame << (colmajor ? "_colmajor" : "_rowmajor") << "_m" + std::to_string(m)
      << "_n" << std::to_string(n) << "_k" << std::to_string(k);

  if (tm > 1 or tn > 1) *os << "_tile" << tm << "x" << tn;
}

template <MatrixKind MatrixT, class EigenMatrixT>
bool is_approx_eigen(
    const MatrixT& device_target, const EigenMatrixT& eigen_reference, float tol) {
  EigenMatrixT eigen_target = EigenMatrixT(device_target.size(0), device_target.size(1));
  device_target.copy_data_to_host(eigen_target.data());

  if constexpr (std::is_same_v<typename MatrixT::scalar_t, half>) {
    auto eigen_reference_f = eigen_reference.template cast<float>();
    return eigen_reference_f.isApprox(eigen_target.template cast<float>(), tol);
  } else {
    return eigen_reference.isApprox(eigen_target, tol);
  }
}

#define DISPATCH_MATRIX_TEST_CASE(test_template, test_name, params_type, colmajor_idx)   \
  bool test_name(const params_type& params) {                                            \
    using eigen_scalar_t                                                                 \
        = std::conditional_t<std::is_same_v<ScalarT, half>, Eigen::half, ScalarT>;       \
                                                                                         \
    if (std::get<colmajor_idx>(params))                                                  \
      return test_template<Eigen::                                                       \
              Matrix<eigen_scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(  \
          params);                                                                       \
                                                                                         \
    return test_template<                                                                \
        Eigen::Matrix<eigen_scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>( \
        params);                                                                         \
  }

template <template <typename> class OpBundleT, ScalarKind ScalarT>
  requires OpBundleKind<OpBundleT, ScalarT>
class MatMulTest : public ::testing::TestWithParam<MatMulTestParams> {
 public:
  using params_tag_t = MatMulTestParamsTag;

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

#define INSTANTIATE_MATRIX_TEST_SUITE_FOR_TYPE(test_suite_name,                             \
    test_class_template,                                                                    \
    test_function,                                                                          \
    impl_bundle,                                                                            \
    scalar_type,                                                                            \
    tile_size,                                                                              \
    tol,                                                                                    \
    values)                                                                                 \
                                                                                            \
  using test_class_template##_##impl_bundle##_##scalar_type##_tile##tile_size##x##tile_size \
      = test_class_template<impl_bundle, scalar_type>;                                      \
                                                                                            \
  TEST_P(                                                                                   \
      test_class_template##_##impl_bundle##_##scalar_type##_tile##tile_size##x##tile_size,  \
      test_function##scalar_type) {                                                         \
    EXPECT_TRUE(test_function(this->GetParam()));                                           \
  }                                                                                         \
                                                                                            \
  INSTANTIATE_TEST_SUITE_P(test_suite_name,                                                 \
      test_class_template##_##impl_bundle##_##scalar_type##_tile##tile_size##x##tile_size,  \
      values);

#define INSTANTIATE_MM_TEST_SUITE_FOR_TYPE(impl_bundle, scalar_type, tile_size, tol)     \
  INSTANTIATE_MATRIX_TEST_SUITE_FOR_TYPE(MMTests,                                        \
      MatMulTest,                                                                        \
      matmul_test_,                                                                      \
      impl_bundle,                                                                       \
      scalar_type,                                                                       \
      tile_size,                                                                         \
      tol,                                                                               \
      ::testing::Combine(::testing::Values(MatMulTestParamsTag{}),                       \
          ::testing::Values(#impl_bundle),                                               \
          ::testing::Bool(),                                                             \
          ::testing::Values(1, 2, 24, 128, 263),                                         \
          ::testing::Values(1, 3, 37, 120, 124),                                         \
          ::testing::Values(1, 4, 35, 121, 257),                                         \
          ::testing::Values(tile_size),                                                  \
          ::testing::Values(tile_size),                                                  \
          ::testing::Values(tol)))

#endif  // TESTS_MM_TEMPLATE_HPP
