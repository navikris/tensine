#ifndef TENSINE_OPS_POOLING_KERNEL_DISPATCH_H
#define TENSINE_OPS_POOLING_KERNEL_DISPATCH_H


#include "tensine/core/tensor.h"


void ts_maxpool2d_dispatch(
    const TsTensor* input_1,
    TsTensor* output,
    const size_t* kernel_size,
    const size_t* stride,
    const size_t* padding
);

#endif /* TENSINE_OPS_POOLING_KERNEL_DISPATCH_H */