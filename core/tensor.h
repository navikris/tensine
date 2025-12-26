#ifndef TENSINE_TENSOR_H
#define TENSINE_TENSOR_H


#include <stddef.h>
#include <dtype.h>


// TODO: replace void with TsGradFn struct
typedef struct {
    void* data;         // raw data buffer
    TsDType dtype;      // element type

    size_t ndim;        // number of dimensions
    size_t* shape;      // length ndim
    size_t* strides;    // length ndim in bytes
    size_t numel;       // total num elements, must match shape

    int requires_grad;  // autograd flag
    void* grad_fn;      // backward node
} TsTensor;


/* Public API */
TsTensor* ts_tensor_create(TsDType dtype, size_t ndim, const size_t* shape, int requires_grad, void* grad_fn);


#endif /* TENSINE_TENSOR_H */
