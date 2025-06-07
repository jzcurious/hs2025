#include "work1/vadd.hpp"

#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

class VAddTest : public ::testing::TestWithParam<std::pair<std::uint32_t, float>> {
 protected:
  bool vadd_test_(std::uint32_t len, float tol) {
    Eigen::VectorXf a = Eigen::VectorXf::Random(len);
    Eigen::VectorXf b = Eigen::VectorXf::Random(len);
    Eigen::VectorXf c = a + b;

    float *d_a, *d_b, *d_c;
    auto size = len * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);

    w1::vadd_f32(d_c, d_a, d_b, len);

    auto hd_c = Eigen::VectorXf(len);

    cudaMemcpy(hd_c.data(), d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c.isApprox(hd_c, tol);
  }
};

TEST_P(VAddTest, vadd_test) {
  auto [len, tol] = GetParam();
  EXPECT_TRUE(vadd_test_(len, tol));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    VAddTests,
    VAddTest,
    ::testing::Values(
        std::make_pair(1, 1e-6),
        std::make_pair(2, 1e-6),
        std::make_pair(3, 1e-6),
        std::make_pair(127, 1e-6),
        std::make_pair(128, 1e-6),
        std::make_pair(129, 1e-6),
        std::make_pair(512, 1e-6),
        std::make_pair(513, 1e-6),
        std::make_pair(1023, 1e-6),
        std::make_pair(1024, 1e-6)
    )
);
// clang-format on
