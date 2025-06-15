#ifndef _MATRIX_KIND_HPP_
#define _MATRIX_KIND_HPP_

#include <concepts>  // IWYU pragma: keep

template <class T>
concept MatrixKind = requires { typename T::matrix_f; };  // TODO: add more traits

#endif  // _MATRIX_KIND_HPP_
