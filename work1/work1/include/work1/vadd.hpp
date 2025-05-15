#ifndef _WRAPPER_VADD_HPP_
#define _WRAPPER_VADD_HPP_

#include <cstdint>

namespace w1 {

void vadd_f32(const float* a, const float* b, float* c, std::uint32_t len);

}  // namespace w1

#endif  // _WRAPPER_VADD_HPP_
