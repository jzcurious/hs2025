#include <cuda_runtime.h>

#include "work2/matrix_view.cuh"
#include "work2/mm_naive.hpp"
#include "work3/mm_shmem.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <gtest/gtest.h>

#include <functional>
#include <string>

using MatrixViewT = MatrixView<float>;

using OpMatMulT
    = std::function<void(const MatrixViewT&, const MatrixViewT&, MatrixViewT&)>;

class MMFunctor {
 public:
  const std::string label;

 private:
  OpMatMulT _opmm;

 public:
  MMFunctor(const char* label, OpMatMulT opmm)
      : label(label)
      , _opmm(opmm) {}

  MatrixViewT& operator()(
      const MatrixViewT& a, const MatrixViewT& b, MatrixViewT& c) const {
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

using MatMulTestParams
    = std::tuple<MMFunctor, bool, std::uint32_t, std::uint32_t, std::uint32_t, float>;

class MatMulTest : public ::testing::TestWithParam<MatMulTestParams> {
 private:
  template <class EigenMatrix>
  bool matmul_test_template_(const MatMulTestParams& params) {
    auto [mmfunc, colmajor, m, n, k, tol] = params;

    EigenMatrix h_a = EigenMatrix::Random(m, k);
    EigenMatrix h_b = EigenMatrix::Random(k, n);
    EigenMatrix h_c = h_a * h_b;

    auto d_a = MatrixViewT(_d_a, m, k, colmajor);
    auto d_b = MatrixViewT(_d_b, k, n, colmajor);
    auto d_c = MatrixViewT(_d_c, m, n, colmajor);

    cudaMemcpy(_d_a, h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(_d_b, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice);

    mmfunc(d_a, d_b, d_c);

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
    auto [_1, _2, m, n, k, _3] = GetParam();
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
    auto colmajor = std::get<1>(params);
    if (colmajor) return matmul_test_template_<Eigen::MatrixXf>(params);
    return matmul_test_template_<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(params);
  }
};

TEST_P(MatMulTest, matmul_test) {
  EXPECT_TRUE(matmul_test_(GetParam()));
}

const MMFunctor naive_mmfunc("naive", w2::matmul<MatrixViewT>);
const MMFunctor shmem_mmfunc("shmem", w3::matmul<MatrixViewT>);

INSTANTIATE_TEST_SUITE_P(MMTests,
    MatMulTest,
    ::testing::Combine(::testing::Values(naive_mmfunc, shmem_mmfunc),
        ::testing::Bool(),
        ::testing::Values(1, 2, 24, 128, 512),
        ::testing::Values(1, 3, 37, 120, 124),
        ::testing::Values(1, 4, 35, 121, 257),
        ::testing::Values(1e-5)));
