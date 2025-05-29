#include <cuda_runtime.h>

#include "work2/matrix/matrix.cuh"
#include "work2/mm_impls/mm_naive.hpp"
#include "work2/mm_impls/mm_shmem.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <functional>
#include <gtest/gtest.h>
#include <string>

template <ScalarKind ScalarT>
class MMFunctor {
 public:
  using opmm_t = std::function<void(const DeviceMatrix<ScalarT>&,
      const DeviceMatrix<ScalarT>&,
      DeviceMatrix<ScalarT>&)>;

  const std::string label;

 private:
  opmm_t _opmm;

 public:
  MMFunctor(const char* label, MMFunctor::opmm_t opmm)
      : label(label)
      , _opmm(opmm) {}

  DeviceMatrix<ScalarT>& operator()(const DeviceMatrix<ScalarT>& a,
      const DeviceMatrix<ScalarT>& b,
      DeviceMatrix<ScalarT>& c) const {
    _opmm(a, b, c);
    return c;
  }

  operator std::string() const {
    return label;
  }

  friend std::ostream& operator<<(std::ostream& os, const MMFunctor& func) {
    return os << static_cast<std::string>(func);
  }
};

template <ScalarKind ScalarT>
using MatMulTestParams = std::
    tuple<MMFunctor<ScalarT>, bool, std::uint32_t, std::uint32_t, std::uint32_t, float>;

template <ScalarKind ScalarT>
class MatMulTest : public ::testing::TestWithParam<MatMulTestParams<ScalarT>> {
 private:
  using matrix_t = DeviceMatrix<ScalarT>;

  template <class EigenMatrix>
  bool matmul_test_template_(const MatMulTestParams<ScalarT>& params) {
    auto [mmfunc, colmajor, m, n, k, tol] = params;

    EigenMatrix h_a = EigenMatrix::Random(m, k);
    EigenMatrix h_b = EigenMatrix::Random(k, n);
    EigenMatrix h_c = h_a * h_b;

    auto d_a = matrix_t(m, k, colmajor);
    auto d_b = matrix_t(k, n, colmajor);
    auto d_c = matrix_t(m, n, colmajor);

    d_a.block().copy_from_host(h_a.data());
    d_b.block().copy_from_host(h_b.data());
    mmfunc(d_a, d_b, d_c);

    EigenMatrix hd_c = EigenMatrix(m, n);
    d_c.block().copy_to_host(hd_c.data());

    if constexpr (std::is_same_v<ScalarT, half>) {
      auto hd_c_float = hd_c.template cast<float>();
      auto h_c_float = h_c.template cast<float>();
      return h_c_float.isApprox(hd_c_float, tol);
    } else {
      return h_c.isApprox(hd_c, tol);
    }
  }

 protected:
  bool matmul_test_(const MatMulTestParams<ScalarT>& params) {
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

template <ScalarKind ScalarT>
void PrintTo(const MatMulTestParams<ScalarT>& params, std::ostream* os) {
  auto [mmfunc, colmajor, m, n, k, tol] = params;
  *os << mmfunc << (colmajor ? "_colmajor" : "_rowmajor") << "_m" + std::to_string(m)
      << "_n" << std::to_string(n) << "_k" << std::to_string(k);
}

#define INSTANTIATE_TEST_SUITE_FOR_TYPE(impl_label, impl_template, scalar_type, tol)     \
  using MatmulTest_##impl_label##_##scalar_type = MatMulTest<scalar_type>;               \
  TEST_P(MatmulTest_##impl_label##_##scalar_type, matmul_test_##scalar_type) {           \
    EXPECT_TRUE(matmul_test_(this->GetParam()));                                         \
  }                                                                                      \
  const MMFunctor<scalar_type> impl_label##_##func_##scalar_type(                        \
      #impl_label, impl_template<scalar_type>);                                          \
  INSTANTIATE_TEST_SUITE_P(MMTests,                                                      \
      MatmulTest_##impl_label##_##scalar_type,                                           \
      ::testing::Combine(::testing::Values(impl_label##_##func_##scalar_type),           \
          ::testing::Bool(),                                                             \
          ::testing::Values(1, 2, 24, 128, 263),                                         \
          ::testing::Values(1, 3, 37, 120, 124),                                         \
          ::testing::Values(1, 4, 35, 121, 257),                                         \
          ::testing::Values(tol)));

INSTANTIATE_TEST_SUITE_FOR_TYPE(naive, w2::matmul_naive, float, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE(naive, w2::matmul_naive, double, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE(naive, w2::matmul_naive, half, 1e-2);

INSTANTIATE_TEST_SUITE_FOR_TYPE(shmem, w2::matmul_shmem, float, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE(shmem, w2::matmul_shmem, double, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE(shmem, w2::matmul_shmem, half, 1e-2);
