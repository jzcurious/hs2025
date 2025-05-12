#ifndef _WRAPPER_MM_HPP_
#define _WRAPPER_MM_HPP_

#include <cstddef>

namespace w2 {

void mm_f32(const float* a, const float* b, float* c, std::size_t len);

}  // namespace w2

#endif  // _WRAPPER_MM_HPP_
