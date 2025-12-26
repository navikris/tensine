#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "tensor.h"
#include "dtype.h"


static void* allocate_data(TsDType dtype, size_t numel) {
    size_t itemsize = ts_dtype_itemsize(dtype);
    if (itemsize == 0) return NULL;

    if (numel != 0 && ((SIZE_MAX / numel) < itemsize))
        return NULL;  // overflow

    return malloc(numel * itemsize);
}


static void calculate_strides(
    TsDType dtype,
    size_t ndim,
    const size_t* shape,
    size_t* strides
) {
    size_t itemsize = ts_dtype_itemsize(dtype);

    if (ndim == 0) return;

    strides[ndim - 1] = itemsize;

    for (size_t i = ndim - 1; i-- > 0;) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}


TsTensor* ts_tensor_create(
    TsDType dtype,
    size_t ndim,
    const size_t* shape,
    int requires_grad,
    void* grad_fn
) {
    if (ndim > 0 && !shape) return NULL;

    TsTensor* tensor = malloc(sizeof(TsTensor));
    if (!tensor) return NULL;

    size_t numel = 1;
    for (size_t i = 0; i < ndim; ++i) {
        if (shape[i] != 0 && numel > SIZE_MAX / shape[i]) {
            free(tensor);
            return NULL;
        }
        numel *= shape[i];
    }

    tensor->data = allocate_data(dtype, numel);
    if (!tensor->data) {
        free(tensor);
        return NULL;
    }
    tensor->owns_data = 1;

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->numel = numel;
    tensor->requires_grad = requires_grad;
    tensor->grad_fn = grad_fn;

    tensor->shape = malloc(ndim * sizeof(size_t));
    tensor->strides = malloc(ndim * sizeof(size_t));

    if (!tensor->shape || !tensor->strides) {
        free(tensor->data);
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }

    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    calculate_strides(dtype, ndim, tensor->shape, tensor->strides);

    return tensor;
}


TsTensor* ts_tensor_from_buffer(
    void* data,
    TsDType dtype,
    size_t ndim,
    const size_t* shape,
    int requires_grad,
    void* grad_fn
) {
    if (!data) return NULL;
    if (ndim > 0 && !shape) return NULL;

    TsTensor* tensor = malloc(sizeof(TsTensor));
    if (!tensor) return NULL;

    size_t numel = 1;
    for (size_t i = 0; i < ndim; ++i) {
        if (shape[i] != 0 && numel > SIZE_MAX / shape[i]) {
            free(tensor);
            return NULL;
        }
        numel *= shape[i];
    }

    tensor->data = data;
    tensor->owns_data = 0;

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->numel = numel;
    tensor->requires_grad = requires_grad;
    tensor->grad_fn = grad_fn;

    tensor->shape = malloc(ndim * sizeof(size_t));
    tensor->strides = malloc(ndim * sizeof(size_t));

    if (!tensor->shape || !tensor->strides) {
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }

    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    calculate_strides(dtype, ndim, tensor->shape, tensor->strides);

    return tensor;
}


void ts_tensor_free(TsTensor* tensor) {
    if (!tensor) return;

    if (tensor->owns_data && tensor->data) {
        free(tensor->data);
        tensor->data = NULL;
    }

    free(tensor->shape);
    free(tensor->strides);

    free(tensor);
}
