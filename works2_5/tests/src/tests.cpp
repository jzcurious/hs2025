#include <cuda_runtime.h>

#include "work2/matrix_view.cuh"
#include "work2/mm_naive.hpp"
#include "work3/mm_shmem.hpp"
#include "work4/mm_wmma.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <gtest/gtest.h>

#include <functional>
#include <string>

template <ScalarKind ScalarT>
class MMFunctor {
 public:
  using opmm_t = std::function<void(
      const MatrixView<ScalarT>&, const MatrixView<ScalarT>&, MatrixView<ScalarT>&)>;

  const std::string label;

 private:
  opmm_t _opmm;

 public:
  MMFunctor(const char* label, MMFunctor::opmm_t opmm)
      : label(label)
      , _opmm(opmm) {}

  MatrixView<ScalarT>& operator()(const MatrixView<ScalarT>& a,
      const MatrixView<ScalarT>& b,
      MatrixView<ScalarT>& c) const {
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
  using mview_t = MatrixView<ScalarT>;

  template <class EigenMatrix>
  bool matmul_test_template_(const MatMulTestParams<ScalarT>& params) {
    auto [mmfunc, colmajor, m, n, k, tol] = params;

    EigenMatrix h_a = EigenMatrix::Random(m, k);
    EigenMatrix h_b = EigenMatrix::Random(k, n);
    EigenMatrix h_c = h_a * h_b;

    auto d_a = mview_t(_d_a, m, k, colmajor);
    auto d_b = mview_t(_d_b, k, n, colmajor);
    auto d_c = mview_t(_d_c, m, n, colmajor);

    cudaMemcpy(_d_a, h_a.data(), h_a.size() * sizeof(ScalarT), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_b, h_b.data(), h_b.size() * sizeof(ScalarT), cudaMemcpyHostToDevice);

    mmfunc(d_a, d_b, d_c);

    EigenMatrix hd_c = EigenMatrix(m, n);
    cudaMemcpy(
        hd_c.data(), d_c.data(), hd_c.size() * sizeof(ScalarT), cudaMemcpyDeviceToHost);

    if constexpr (std::is_same_v<ScalarT, half>) {
      auto hd_c_float = hd_c.template cast<float>();
      auto h_c_float = h_c.template cast<float>();
      return h_c_float.isApprox(hd_c_float, tol);
    } else {
      return h_c.isApprox(hd_c, tol);
    }
  }

 protected:
  ScalarT* _d_a = nullptr;
  ScalarT* _d_b = nullptr;
  ScalarT* _d_c = nullptr;

  void SetUp() override {
    auto [_1, _2, m, n, k, _3] = this->GetParam();
    cudaMalloc(&_d_a, sizeof(ScalarT) * m * k);
    cudaMalloc(&_d_b, sizeof(ScalarT) * k * n);
    cudaMalloc(&_d_c, sizeof(ScalarT) * m * n);
  }

  void TearDown() override {
    if (_d_a) cudaFree(_d_a);
    if (_d_b) cudaFree(_d_b);
    if (_d_c) cudaFree(_d_c);
  }

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

#define INSTANTIATE_TEST_SUITE_FOR_TYPE(impl_label, impl_template, scalar_type, tol)     \
  using MatmulTest_##impl_label##_##scalar_type = MatMulTest<scalar_type>;               \
  TEST_P(MatmulTest_##impl_label##_##scalar_type, matmul_test_##scalar_type) {           \
    EXPECT_TRUE(matmul_test_(this->GetParam()));                                         \
  }                                                                                      \
  const MMFunctor<scalar_type> impl_label##_##func_##scalar_type(                        \
      #impl_label, impl_template<MatrixView<scalar_type>>);                              \
  INSTANTIATE_TEST_SUITE_P(MMTests,                                                      \
      MatmulTest_##impl_label##_##scalar_type,                                           \
      ::testing::Combine(::testing::Values(impl_label##_##func_##scalar_type),           \
          ::testing::Bool(),                                                             \
          ::testing::Values(1, 2, 24, 128, 263),                                         \
          ::testing::Values(1, 3, 37, 120, 124),                                         \
          ::testing::Values(1, 4, 35, 121, 257),                                         \
          ::testing::Values(tol)));

INSTANTIATE_TEST_SUITE_FOR_TYPE(naive, w2::matmul, float, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE(naive, w2::matmul, double, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE(naive, w2::matmul, half, 1e-2);

INSTANTIATE_TEST_SUITE_FOR_TYPE(shmem, w3::matmul, float, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE(shmem, w3::matmul, double, 1e-5);
INSTANTIATE_TEST_SUITE_FOR_TYPE(shmem, w3::matmul, half, 1e-2);

INSTANTIATE_TEST_SUITE_FOR_TYPE(wmma, w4::matmul, half, 1e-2);

// auto f = [](const
// testing::TestParamInfo<MatmulTest_##impl_label##_##scalar_type::ParamType>& info) {
// };

// TODO: add human readable test names
