#ifndef TENSINE_TENSOR_H
#define TENSINE_TENSOR_H


#ifdef __cplusplus
extern "C" {
#endif


#include "tensine/core/dtype.h"

#include <stddef.h>


#define TS_MAX_DIMS 4


/*
Ownership rules:
- TsTensor owns shape, strides and offsets
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
    size_t* offsets;            // stride offsets lenfth ndim in bytes (unused for now)
    size_t numel;               // total num elements, must match shape

    // TODO: replace void with TsGradFn struct
    int requires_grad;          // autograd flag
    void* grad_fn;              // backward node
} TsTensor;


/* Public APIs */

/* Creation/Destruction */
TsTensor* ts_tensor_create(TsDType dtype, size_t ndim, const size_t* shape, int requires_grad, void* grad_fn);
TsTensor* ts_tensor_empty_like(const TsTensor* src);
TsTensor* ts_tensor_from_buffer(void* data, TsDType dtype, size_t ndim, const size_t* shape, int requires_grad, void* grad_fn);
TsTensor* ts_tensor_from_storage(TsTensorStorage* storage, TsDType dtype, size_t ndim, const size_t* shape, int requires_grad, void* grad_fn);
void ts_tensor_free(TsTensor* tensor);

/* Metadata view */
TsDType ts_tensor_dtype(const TsTensor* t);
size_t ts_tensor_ndim(const TsTensor* t);
size_t ts_tensor_numel(const TsTensor* t);
const size_t* ts_tensor_shape(const TsTensor* t);

/* Alias ops */
TsTensor* ts_tensor_shallow_copy(const TsTensor* src);

/* View ops */
TsTensor* ts_tensor_reshape(const TsTensor* src, const size_t* new_shape, size_t new_ndim);
TsTensor* ts_tensor_permute(const TsTensor* src, const size_t* permute_order);
TsTensor* ts_tensor_transpose(const TsTensor* src, size_t dim_1, size_t dim_2);
TsTensor* ts_tensor_slice(const TsTensor* src, const size_t* start_idxs, const size_t* end_idxs);
TsTensor* ts_tensor_broadcast(const TsTensor* src, const size_t* new_shape, size_t new_ndim);

/* Copy/Layout */
int ts_tensor_is_contiguous(const TsTensor* t);
TsTensor* ts_tensor_to_contiguous(const TsTensor* src);
TsTensor* ts_tensor_clone(const TsTensor* src);

/* Print */
void ts_tensor_print(const TsTensor* t);


#ifdef __cplusplus
}
#endif


#endif /* TENSINE_TENSOR_H */
