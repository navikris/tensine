#ifndef TENSINE_OPS_ELEMENTWISE_KERNEL_DISPATCH_H
#define TENSINE_OPS_ELEMENTWISE_KERNEL_DISPATCH_H


#include "tensine/core/tensor.h"


void ts_elementwise_add_dispatch(const TsTensor* input_1, const TsTensor* input_2, TsTensor* output);
void ts_elementwise_mul_dispatch(const TsTensor* input_1, const TsTensor* input_2, TsTensor* output);


#endif /* TENSINE_OPS_ELEMENTWISE_KERNEL_DISPATCH_H */