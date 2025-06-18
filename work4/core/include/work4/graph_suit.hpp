#ifndef _GRAPH_SUIT_HPP_
#define _GRAPH_SUIT_HPP_

#include <cuda_runtime.h>

struct GraphSuit {
 protected:
  cudaStream_t _stream;
  cudaGraph_t _graph;
  cudaGraphExec_t _graph_inst;
  bool _graph_created;

 public:
  GraphSuit()
      : _graph_created(false) {
    cudaStreamCreate(&_stream);
  }

  ~GraphSuit() {
    cudaGraphExecDestroy(_graph_inst);
    cudaGraphDestroy(_graph);
    cudaStreamDestroy(_stream);
  }
};

#endif  // _GRAPH_SUIT_HPP_
