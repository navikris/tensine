#ifndef TENSINE_OPS_DISPATCH_ELTWISE_SEMANTIC_H
#define TENSINE_OPS_DISPATCH_ELTWISE_SEMANTIC_H


#ifdef __cplusplus
extern "C" {
#endif


#include "tensine/core/tensor.h"


TsTensor* ts_add(const TsTensor* input_1, const TsTensor* input_2);
TsTensor* ts_mul(const TsTensor* input_1, const TsTensor* input_2);


#ifdef __cplusplus
}
#endif


#endif /* TENSINE_OPS_DISPATCH_ELTWISE_SEMANTIC_H */