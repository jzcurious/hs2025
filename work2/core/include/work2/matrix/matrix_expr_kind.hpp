#ifndef _MATRIX_EXPR_KIND_HPP_
#define _MATRIX_EXPR_KIND_HPP_

#include <concepts>  // IWYU pragma: keep

template <class T>
concept MatrixExprKind = requires {
  typename T::scalar_t;
  typename T::matrixexpr_f;
};  // TODO: add more traits

#endif  // _MATRIX_EXPR_KIND_HPP_
