#ifndef _MATRIX_VIEW_KIND_HPP_
#define _MATRIX_VIEW_KIND_HPP_

#include <concepts>  // IWYU pragma: keep
#include <cstdint>

template <class T>
concept MatrixViewKind = requires { typename T::scalar_t; }
                         and requires(T x, std::uint32_t i, std::uint32_t j) {
                               { x(i, j) } -> std::same_as<typename T::scalar_t&>;
                               { x.data(i, j) } -> std::same_as<typename T::scalar_t*>;
                               { x.ldim() } -> std::same_as<std::uint32_t>;
                               { x.size(i) } -> std::same_as<std::uint32_t>;
                             };

#endif  // _MATRIX_VIEW_KIND_HPP_
