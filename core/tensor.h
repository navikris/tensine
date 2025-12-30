#ifndef TENSINE_TENSOR_H
#define TENSINE_TENSOR_H


#include <stddef.h>

#include "dtype.h"


#define TS_MAX_DIMS 4


/*
Ownership rules:
- TsTensor owns shape and strides
- TsTensorStorage owns data buffer and deallocation based on ref_count
*/


typedef struct {
    void* data;         // storage pointer
    int owns_data;      // ownership flag
    size_t nbytes;      // total num bytes
    size_t ref_count;   // number of reference
} TsTensorStorage;


typedef struct {
    TsTensorStorage* storage;   // data storage
    TsDType dtype;              // element type

    size_t ndim;                // number of dimensions
    size_t* shape;              // length ndim
    size_t* strides;            // length ndim in bytes
    size_t numel;               // total num elements, must match shape

    // TODO: replace void with TsGradFn struct
    int requires_grad;          // autograd flag
    void* grad_fn;              // backward node
} TsTensor;


/* Public API */
TsTensor* ts_tensor_create(TsDType dtype, size_t ndim, const size_t* shape, int requires_grad, void* grad_fn);
TsTensor* ts_tensor_empty_like(const TsTensor* src);
TsTensor* ts_tensor_from_buffer(void* data, TsDType dtype, size_t ndim, const size_t* shape, int requires_grad, void* grad_fn);
TsTensor* ts_tensor_from_storage(TsTensorStorage* storage, TsDType dtype, size_t ndim, const size_t* shape, int requires_grad, void* grad_fn);
void ts_tensor_free(TsTensor* tensor);
TsDType ts_tensor_dtype(const TsTensor* t);
size_t ts_tensor_ndim(const TsTensor* t);
size_t ts_tensor_numel(const TsTensor* t);
const size_t* ts_tensor_shape(const TsTensor* t);
int ts_tensor_is_contiguous(const TsTensor* t);
TsTensor* ts_tensor_to_contiguous(TsTensor* src);
TsTensor* ts_tensor_shallow_copy(const TsTensor* src);
TsTensor* ts_tensor_clone(const TsTensor* src);
void ts_tensor_print(const TsTensor* t);


#endif /* TENSINE_TENSOR_H */
