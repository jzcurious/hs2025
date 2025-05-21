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
};

struct MatMulTestParams {
  const MMFunctor& mmfunctor;

  bool colmajor;
  std::uint32_t m;
  std::uint32_t n;
  std::uint32_t k;
  float tol;

  operator std::string() const {
    std::stringstream ss;
    ss << mmfunctor.label << (colmajor ? "_colmajor_m" : "_rowmajor_m")
       << std::to_string(m) << "_n" << std::to_string(n) << "_k" << std::to_string(k);
    return ss.str();
  }

  friend std::ostream& operator<<(std::ostream& os, const MatMulTestParams& params) {
    return os << static_cast<std::string>(params);
  }
};

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
    if (params.colmajor) return matmul_test_template_<Eigen::MatrixXf>(params);
    return matmul_test_template_<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(params);
  }
};

TEST_P(MatMulTest, matmul_test) {
  EXPECT_TRUE(matmul_test_(GetParam()));
}

const MMFunctor naive_mmfunc("naive", w2::matmul<MatrixViewT>);
const MMFunctor shmem_mmfunc("shmem", w3::matmul<MatrixViewT>);

auto make_params_cases(const MMFunctor& func, float tol) {
  return std::vector<MatMulTestParams>{
      {.mmfunctor = func, .colmajor = false,   .m = 1,   .n = 1,   .k = 1, .tol = tol},
      {.mmfunctor = func, .colmajor = false,   .m = 2,   .n = 5,   .k = 4, .tol = tol},
      {.mmfunctor = func, .colmajor = false,  .m = 24,  .n = 54,  .k = 44, .tol = tol},
      {.mmfunctor = func, .colmajor = false, .m = 128,  .n = 54, .k = 127, .tol = tol},
      {.mmfunctor = func, .colmajor = false, .m = 512, .n = 124,  .k = 32, .tol = tol},
      {.mmfunctor = func, .colmajor = false,  .m = 12, .n = 124, .k = 257, .tol = tol},
      {.mmfunctor = func,  .colmajor = true,   .m = 1,   .n = 1,   .k = 1, .tol = tol},
      {.mmfunctor = func,  .colmajor = true,   .m = 2,   .n = 5,   .k = 4, .tol = tol},
      {.mmfunctor = func,  .colmajor = true,  .m = 24,  .n = 54,  .k = 44, .tol = tol},
      {.mmfunctor = func,  .colmajor = true, .m = 128,  .n = 54, .k = 127, .tol = tol},
      {.mmfunctor = func,  .colmajor = true, .m = 512, .n = 124,  .k = 32, .tol = tol},
      {.mmfunctor = func,  .colmajor = true,  .m = 12, .n = 124, .k = 257, .tol = tol}
  };
}

#define INSTANTIATE_TEST_SUITE_MATMUL_(suite_name, func, atol)                           \
  INSTANTIATE_TEST_SUITE_P(                                                              \
      suite_name, MatMulTest, ::testing::ValuesIn(make_params_cases(func, 1e-5)));

INSTANTIATE_TEST_SUITE_MATMUL_(MatMulTestsNaive, naive_mmfunc, 1e-5);
INSTANTIATE_TEST_SUITE_MATMUL_(MatMulTestsShmem, shmem_mmfunc, 1e-5);
