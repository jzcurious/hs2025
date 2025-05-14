#ifndef _MATRIX_KIND_HPP_
#define _MATRIX_KIND_HPP_

#include <concepts>

template <class T>
concept MatrixKind = requires { (typename T::matrix_feature{}); };

#endif  // _MATRIX_KIND_HPP_
