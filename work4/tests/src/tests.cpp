#include "work2/tests/tests_mm_template.hpp"
#include "work4/default_bundle.hpp"

#include <gtest/gtest.h>

struct LinearTestParamsTag {
  // NOTE: Oh, dumb gtest )
};

// clang-format off
using LinearTestParams = std::tuple<
  LinearTestParamsTag,  // tag
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

inline void PrintTo(const LinearTestParams& params, std::ostream* os) {
  auto [_, blame, colmajor, m, n, k, tm, tn, tol] = params;
  *os << blame << (colmajor ? "_colmajor" : "_rowmajor") << "_m" + std::to_string(m)
      << "_n" << std::to_string(n) << "_k" << std::to_string(k);

  if (tm > 1 or tn > 1) *os << "_tile" << tm << "x" << tn;
}

template <template <typename> class OpBundleT, ScalarKind ScalarT>
  requires OpBundleKind<OpBundleT, ScalarT>
class LinearTest : public ::testing::TestWithParam<LinearTestParams> {
 private:
  using matrix_t = DeviceMatrix<OpBundleT, ScalarT>;

  template <class EigenMatrix>
  bool linear_test_template_(const LinearTestParams& params) {
    auto [_0, _1, colmajor, m, n, k, tm, tn, tol] = params;

    EigenMatrix h_x = EigenMatrix::Random(m, k);
    EigenMatrix h_w = EigenMatrix::Random(k, n);
    EigenMatrix h_b = EigenMatrix::Random(1, n);
    EigenMatrix h_y = h_x * h_w;
    // EigenMatrix h_y = h_x * h_w + h_b;

    auto d_x = matrix_t(
        m, k, MatrixOptions{}.colmajor(colmajor).tile(tm, tn, m, k).src(h_x.data()));

    auto d_w = matrix_t(
        k, n, MatrixOptions{}.colmajor(colmajor).tile(tm, tn, k, n).src(h_w.data()));

    auto d_b = matrix_t(
        1, n, MatrixOptions{}.colmajor(colmajor).tile(tm, tn, 1, n).src(h_b.data()));

    // auto d_y = d_x * d_w + d_b;
    auto d_y = d_x * d_w;

    return is_approx_eigen(d_y, h_y, tol);
  }

 protected:
  DISPATCH_MATRIX_TEST_CASE(linear_test_template_, linear_test_, LinearTestParams, 2);
};

#define INSTANTIATE_LINEAR_TEST_SUITE_FOR_TYPE(impl_bundle, scalar_type, tile_size, tol) \
  INSTANTIATE_MATRIX_TEST_SUITE_FOR_TYPE(LinearTests,                                    \
      LinearTest,                                                                        \
      linear_test_,                                                                      \
      impl_bundle,                                                                       \
      scalar_type,                                                                       \
      tile_size,                                                                         \
      tol,                                                                               \
      ::testing::Combine(::testing::Values(LinearTestParamsTag{}),                       \
          ::testing::Values(#impl_bundle),                                               \
          ::testing::Bool(),                                                             \
          ::testing::Values(1, 2, 24, 128, 263),                                         \
          ::testing::Values(1, 3, 37, 120, 124),                                         \
          ::testing::Values(1, 4, 35, 121, 257),                                         \
          ::testing::Values(tile_size),                                                  \
          ::testing::Values(tile_size),                                                  \
          ::testing::Values(tol)))

INSTANTIATE_LINEAR_TEST_SUITE_FOR_TYPE(OpBundleDefault, half, 16, 1e-2);
