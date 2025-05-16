#include "work2/mm_naive.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

struct MatMulTestParams {
  std::uint32_t m;
  std::uint32_t n;
  std::uint32_t k;
  float tol;
};

class MatMulTest : public ::testing::TestWithParam<MatMulTestParams> {
 protected:
  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;

  void SetUp() override {
    auto params = GetParam();

    cudaMalloc(&d_a, sizeof(float) * params.m * params.k);
    cudaMalloc(&d_b, sizeof(float) * params.k * params.n);
    cudaMalloc(&d_c, sizeof(float) * params.m * params.n);
  }

  void TearDown() override {
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_c) cudaFree(d_c);
  }

  bool matmul_test_(MatMulTestParams params) {}
};
