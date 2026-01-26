#ifndef TENSINE_OPS_MATMUL_KERNEL_DISPATCH_H
#define TENSINE_OPS_MATMUL_KERNEL_DISPATCH_H


#include "tensine/core/tensor.h"


void ts_matmul_dispatch(const TsTensor* input_1, const TsTensor* input_2, TsTensor* output);


#endif /* TENSINE_OPS_MATMUL_KERNEL_DISPATCH_H */