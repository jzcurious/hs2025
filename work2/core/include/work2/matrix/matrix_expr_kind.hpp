#ifndef _MATRIX_EXPR_KIND_HPP_
#define _MATRIX_EXPR_KIND_HPP_

#include <concepts>  // IWYU pragma: keep

#include "work2/matrix/matrix_view_kind.hpp"

template <class T>
concept MatrixExprKind = requires {
  typename T::result_t;
  typename T::matrix_expr_f;
} and requires(T x) {
  { x.required_result_view() } -> LRefToMatrixKind;
  {
    x.eval(std::declval<std::add_lvalue_reference_t<typename T::result_t>>())
  } -> LRefToMatrixKind;
};

#endif  // _MATRIX_EXPR_KIND_HPP_
