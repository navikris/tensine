#ifndef TENSINE_TENSOR_H
#define TENSINE_TENSOR_H


#include <stddef.h>

#include "dtype.h"


/*
Ownership rules:
- TsTensor owns shape and strides
- TsTensor owns data only if owns_data == 1
*/
typedef struct {
    void* data;         // raw data buffer
    int owns_data;      // ownership flag
    TsDType dtype;      // element type

    size_t ndim;        // number of dimensions
    size_t* shape;      // length ndim
    size_t* strides;    // length ndim in bytes
    size_t numel;       // total num elements, must match shape

    // TODO: replace void with TsGradFn struct
    int requires_grad;  // autograd flag
    void* grad_fn;      // backward node
} TsTensor;


/* Public API */
TsTensor* ts_tensor_create(TsDType dtype, size_t ndim, const size_t* shape, int requires_grad, void* grad_fn);
TsTensor* ts_tensor_from_buffer(void* data, TsDType dtype, size_t ndim, const size_t* shape, int requires_grad, void* grad_fn);
void ts_tensor_free(TsTensor* tensor);


#endif /* TENSINE_TENSOR_H */
