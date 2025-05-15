#ifndef _CUDA_GRID_HEURISTICS_HPP_
#define _CUDA_GRID_HEURISTICS_HPP_

#include <cstdint>
#include <cuda_runtime.h>

namespace heuristic {

inline std::uint32_t cover(std::uint32_t work_size, std::uint32_t block_size) {
  return (work_size + block_size - 1) / block_size;
}

}  // namespace heuristic

#endif  // _CUDA_GRID_HEURISTICS_HPP_
