#ifndef TENSINE_OPS_DISPATCH_POOLING_SEMANTIC_H
#define TENSINE_OPS_DISPATCH_POOLING_SEMANTIC_H


#ifdef __cplusplus
extern "C" {
#endif


#include "tensine/core/tensor.h"


TsTensor* ts_maxpool2d(
    const TsTensor* input,
    const size_t* kernel_size,
    const size_t* stride,
    const size_t* padding
);


#ifdef __cplusplus
}
#endif


#endif /* TENSINE_OPS_DISPATCH_POOLING_SEMANTIC_H */