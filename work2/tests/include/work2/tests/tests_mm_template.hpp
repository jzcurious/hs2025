#ifndef TESTS_MM_TEMPLATE_HPP
#define TESTS_MM_TEMPLATE_HPP

#include <cuda_runtime.h>

#include "work2/matrix/matrix.hpp"
#include "work2/mm_impls/op_bundle_kind.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>

struct Blame {
  // NOTE: Oh, dumb gtest )
  const std::string str;
};

using MatMulTestParams = std::tuple<Blame,
    bool,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    float>;  // TODO:
             // Replace to
             // struct.

inline void PrintTo(const MatMulTestParams& params, std::ostream* os) {
  auto [blame, colmajor, m, n, k, tm, tn, tol] = params;
  *os << blame.str << (colmajor ? "_colmajor" : "_rowmajor") << "_m" + std::to_string(m)
      << "_n" << std::to_string(n) << "_k" << std::to_string(k);

  if (tm > 1 or tn > 1) *os << "_tile" << tm << "x" << tn;
}

template <template <typename> class OpBundleT, ScalarKind ScalarT>
  requires OpBundleKind<OpBundleT, ScalarT>
class MatMulTest : public ::testing::TestWithParam<MatMulTestParams> {
 private:
  using matrix_t = DeviceMatrix<OpBundleT, ScalarT>;

  template <class EigenMatrix>
  bool matmul_test_template_(const MatMulTestParams& params) {
    auto [_, colmajor, m, n, k, tm, tn, tol] = params;

    EigenMatrix h_a = EigenMatrix::Random(m, k);
    EigenMatrix h_b = EigenMatrix::Random(k, n);
    EigenMatrix h_c = h_a * h_b;

    auto d_a = matrix_t(
        m, k, MatrixOps{}.colmajor(colmajor).tile(tm, tn, m, k).src(h_a.data()));
    auto d_b = matrix_t(
        k, n, MatrixOps{}.colmajor(colmajor).tile(tm, tn, k, n).src(h_b.data()));
    auto d_c = d_a * d_b;

    EigenMatrix hd_c = EigenMatrix(m, n);
    d_c.copy_data_to_host(hd_c.data());

    if constexpr (std::is_same_v<ScalarT, half>) {
      auto hd_c_float = hd_c.template cast<float>();
      auto h_c_float = h_c.template cast<float>();
      return h_c_float.isApprox(hd_c_float, tol);
    } else {
      return h_c.isApprox(hd_c, tol);
    }
  }

 protected:
  bool matmul_test_(const MatMulTestParams& params) {
    using eigen_scalar_t
        = std::conditional_t<std::is_same_v<ScalarT, half>, Eigen::half, ScalarT>;

    auto colmajor = std::get<1>(params);

    if (colmajor)
      return matmul_test_template_<
          Eigen::Matrix<eigen_scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(
          params);

    return matmul_test_template_<
        Eigen::Matrix<eigen_scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        params);
  }
};

#define INSTANTIATE_TEST_SUITE_FOR_TYPE(impl_bundle, scalar_type, tile_size, tol)        \
  using MatmulTest_##impl_bundle##_##scalar_type##_tile##tile_size##x##tile_size         \
      = MatMulTest<impl_bundle, scalar_type>;                                            \
                                                                                         \
  TEST_P(MatmulTest_##impl_bundle##_##scalar_type##_tile##tile_size##x##tile_size,       \
      matmul_test_##scalar_type) {                                                       \
    EXPECT_TRUE(matmul_test_(this->GetParam()));                                         \
  }                                                                                      \
                                                                                         \
  INSTANTIATE_TEST_SUITE_P(MMTests,                                                      \
      MatmulTest_##impl_bundle##_##scalar_type##_tile##tile_size##x##tile_size,          \
      ::testing::Combine(::testing::Values(Blame(#impl_bundle)),                         \
          ::testing::Bool(),                                                             \
          ::testing::Values(1, 2, 24, 128, 263),                                         \
          ::testing::Values(1, 3, 37, 120, 124),                                         \
          ::testing::Values(1, 4, 35, 121, 257),                                         \
          ::testing::Values(tile_size),                                                  \
          ::testing::Values(tile_size),                                                  \
          ::testing::Values(tol)));

#endif  // TESTS_MM_TEMPLATE_HPP
