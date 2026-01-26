#ifndef TENSINE_OPS_DISPATCH_MATMUL_SEMANTIC_H
#define TENSINE_OPS_DISPATCH_MATMUL_SEMANTIC_H


#ifdef __cplusplus
extern "C" {
#endif


#include "tensine/core/tensor.h"


TsTensor* ts_matmul(const TsTensor* input_1, const TsTensor* input_2);


#ifdef __cplusplus
}
#endif


#endif /* TENSINE_OPS_DISPATCH_MATMUL_SEMANTIC_H */