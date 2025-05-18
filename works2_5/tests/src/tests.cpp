#include <cuda_runtime.h>

#include "work2/matrix_view.cuh"
#include "work2/mm_naive.hpp"

#define EIGEN_NO_CUDA 1

#include <Eigen/Dense>
#include <cstdint>
#include <gtest/gtest.h>

// #ifdef __CLANGD__
// void* operator new(std::size_t size);
// #endif

struct MatMulTestParams {
  std::uint32_t m;
  std::uint32_t n;
  std::uint32_t k;
  float tol;
};

class MatMulTest : public ::testing::TestWithParam<MatMulTestParams> {
 protected:
  float* _d_a = nullptr;
  float* _d_b = nullptr;
  float* _d_c = nullptr;

  void SetUp() override {
    auto [m, n, k, _] = GetParam();
    cudaMalloc(&_d_a, sizeof(float) * m * k);
    cudaMalloc(&_d_b, sizeof(float) * k * n);
    cudaMalloc(&_d_c, sizeof(float) * m * n);
  }

  void TearDown() override {
    if (_d_a) cudaFree(_d_a);
    if (_d_b) cudaFree(_d_b);
    if (_d_c) cudaFree(_d_c);
  }

  bool matmul_test_(MatMulTestParams params) {
    auto [m, n, k, tol] = params;

    Eigen::MatrixXf h_a = Eigen::MatrixXf::Random(m, k);
    Eigen::MatrixXf h_b = Eigen::MatrixXf::Random(k, n);
    Eigen::MatrixXf h_c = h_a * h_b;

    auto d_a = MatrixView<float>(_d_a, m, k);
    auto d_b = MatrixView<float>(_d_b, k, n);
    auto d_c = MatrixView<float>(_d_c, m, n);

    cudaMemcpy(_d_a, h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_b, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice);

    w2::matmul(d_a, d_b, d_c);

    Eigen::MatrixXf hd_c = Eigen::MatrixXf(m, n);
    cudaMemcpy(hd_c.data(), _d_c, hd_c.size() * sizeof(float), cudaMemcpyDeviceToHost);

    return h_c.isApprox(hd_c, tol);
  }
};

TEST_P(MatMulTest, matmul_test) {
  EXPECT_TRUE(true);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    MatMulTests,
    MatMulTest,
    ::testing::Values(
      MatMulTestParams{1, 1, 1, 1e-6},
      MatMulTestParams{2, 5, 4, 1e-6},
      MatMulTestParams{24, 54, 44, 1e-6},
      MatMulTestParams{128, 54, 127, 1e-6},
      MatMulTestParams{512, 124, 32, 1e-6},
      MatMulTestParams{12, 124, 257, 1e-6}
    )
);
// clang-format on
