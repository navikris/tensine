#ifndef TENSINE_OPS_DISPATCH_ACTIVATION_SEMANTIC_H
#define TENSINE_OPS_DISPATCH_ACTIVATION_SEMANTIC_H


#ifdef __cplusplus
extern "C" {
#endif


#include "tensine/core/tensor.h"


TsTensor* ts_softmax(const TsTensor* input, size_t dim);


#ifdef __cplusplus
}
#endif


#endif /* TENSINE_OPS_DISPATCH_ACTIVATION_SEMANTIC_H */