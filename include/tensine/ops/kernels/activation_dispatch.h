#ifndef TENSINE_OPS_ACTIVATION_KERNEL_DISPATCH_H
#define TENSINE_OPS_ACTIVATION_KERNEL_DISPATCH_H


#include "tensine/core/tensor.h"


void ts_softmax_dispatch(
    const TsTensor* input,
    TsTensor* output
);

#endif /* TENSINE_OPS_ACTIVATION_KERNEL_DISPATCH_H */