#include <cuda_runtime.h>

#include "work2/matrix_view.cuh"
#include "work2/mm_naive.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <gtest/gtest.h>

#include <format>
#include <string>

struct MatMulTestParams {
  bool colmajor;
  std::uint32_t m;
  std::uint32_t n;
  std::uint32_t k;
  float tol;

  operator std::string() const {
    return std::format("{}, m = {}, n = {}, k = {}, tolerance = {}",
        colmajor ? "col-major" : "row-major",
        m,
        n,
        k,
        tol);
  }
};

class MatMulTest : public ::testing::TestWithParam<MatMulTestParams> {
 private:
  template <class EigenMatrix>
  bool matmul_test_template_(const MatMulTestParams& params) {
    auto [colmajor, m, n, k, tol] = params;

    EigenMatrix h_a = EigenMatrix::Random(m, k);
    EigenMatrix h_b = EigenMatrix::Random(k, n);
    EigenMatrix h_c = h_a * h_b;

    auto d_a = MatrixView<float>(_d_a, m, k, colmajor);
    auto d_b = MatrixView<float>(_d_b, k, n, colmajor);
    auto d_c = MatrixView<float>(_d_c, m, n, colmajor);

    cudaMemcpy(_d_a, h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_b, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice);

    w2::matmul(d_a, d_b, d_c);

    EigenMatrix hd_c = EigenMatrix(m, n);
    cudaMemcpy(
        hd_c.data(), d_c.data(), hd_c.size() * sizeof(float), cudaMemcpyDeviceToHost);

    return h_c.isApprox(hd_c, tol);
  }

 protected:
  float* _d_a = nullptr;
  float* _d_b = nullptr;
  float* _d_c = nullptr;

  void SetUp() override {
    auto [_, m, n, k, __] = GetParam();
    cudaMalloc(&_d_a, sizeof(float) * m * k);
    cudaMalloc(&_d_b, sizeof(float) * k * n);
    cudaMalloc(&_d_c, sizeof(float) * m * n);
  }

  void TearDown() override {
    if (_d_a) cudaFree(_d_a);
    if (_d_b) cudaFree(_d_b);
    if (_d_c) cudaFree(_d_c);
  }

  bool matmul_test_(const MatMulTestParams& params) {
    if (params.colmajor) return matmul_test_template_<Eigen::MatrixXf>(params);
    return matmul_test_template_<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(params);
  }
};

TEST_P(MatMulTest, matmul_test) {
  EXPECT_TRUE(matmul_test_(GetParam()));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    MatMulTests,
    MatMulTest,
    ::testing::Values(
      MatMulTestParams{.colmajor = false, .m = 1, .n = 1, .k = 1, .tol = 1e-5},
      MatMulTestParams{.colmajor = false, .m = 2, .n = 5, .k = 4, .tol = 1e-5},
      MatMulTestParams{.colmajor = false, .m = 24, .n = 54, .k = 44, .tol = 1e-4},
      MatMulTestParams{.colmajor = false, .m = 128, .n = 54, .k = 127, .tol = 1e-4},
      MatMulTestParams{.colmajor = false, .m = 512, .n = 124, .k = 32, .tol = 1e-4},
      MatMulTestParams{.colmajor = false, .m = 12, .n = 124, .k = 257, .tol = 1e-4},
      MatMulTestParams{.colmajor = true, .m = 1, .n = 1, .k = 1, .tol = 1e-5},
      MatMulTestParams{.colmajor = true, .m = 2, .n = 5, .k = 4, .tol = 1e-5},
      MatMulTestParams{.colmajor = true, .m = 24, .n = 54, .k = 44, .tol = 1e-4},
      MatMulTestParams{.colmajor = true, .m = 128, .n = 54, .k = 127, .tol = 1e-4},
      MatMulTestParams{.colmajor = true, .m = 512, .n = 124, .k = 32, .tol = 1e-4},
      MatMulTestParams{.colmajor = true, .m = 12, .n = 124, .k = 257, .tol = 1e-4},
      [](const testing::TestParamInfo<MatMulTestParams>& info) {
          return static_cast<std::string>(info.param);
      }
    )
);
// clang-format on
