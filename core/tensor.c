#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "tensor.h"
#include "dtype.h"


static size_t _get_num_bytes(TsDType dtype, size_t numel) {
    size_t itemsize = ts_dtype_itemsize(dtype);
    if (itemsize == 0) return 0;

    if (numel != 0 && ((SIZE_MAX / numel) < itemsize))
        return 0;  // overflow

    return numel * itemsize;
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


static void ts_storage_retain(TsTensorStorage* s) {
    assert(s->ref_count > 0);
    s->ref_count++;
}


static void ts_storage_release(TsTensorStorage* s) {
    assert(s->ref_count > 0);

    s->ref_count--;
    if (s->ref_count == 0) {
        if (s->owns_data) {
            free(s->data);
        }
        free(s);
    }
}


static TsTensorStorage* ts_storage_create(size_t nbytes) {
    TsTensorStorage* s = malloc(sizeof(TsTensorStorage));
    if (!s) return NULL;

    s->data = malloc(nbytes);
    if (!s->data) {
        free(s);
        return NULL;
    }

    s->nbytes = nbytes;
    s->ref_count = 1;
    s->owns_data = 1;
    return s;
}


static TsTensorStorage* ts_storage_from_buffer(void* data, size_t nbytes) {
    TsTensorStorage* s = malloc(sizeof(TsTensorStorage));
    if (!s) return NULL;

    s->data = data;
    s->nbytes = nbytes;
    s->ref_count = 1;
    s->owns_data = 0;
    return s;
}


size_t ts_tensor_numel(const TsTensor* t) {
    assert(t);
    return t->numel;
}


size_t ts_tensor_ndim(const TsTensor* t) {
    assert(t);
    return t->ndim;
}


TsDType ts_tensor_dtype(const TsTensor* t) {
    assert(t);
    return t->dtype;
}


const size_t* ts_tensor_shape(const TsTensor* t) {
    assert(t);
    return t->shape;
}


// Creates contiguous tensors
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

    tensor->storage = ts_storage_create(_get_num_bytes(dtype, numel));
    if (!tensor->storage) {
        free(tensor);
        return NULL;
    }

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->numel = numel;
    tensor->requires_grad = requires_grad;
    tensor->grad_fn = grad_fn;

    tensor->shape = malloc(ndim * sizeof(size_t));
    tensor->strides = malloc(ndim * sizeof(size_t));

    if (!tensor->shape || !tensor->strides) {
        ts_storage_release(tensor->storage);
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

    tensor->storage = ts_storage_from_buffer(
        data, 
        _get_num_bytes(dtype, numel)
    );
    if (!tensor->storage) {
        free(tensor);
        return NULL;
    }

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->numel = numel;
    tensor->requires_grad = requires_grad;
    tensor->grad_fn = grad_fn;

    tensor->shape = malloc(ndim * sizeof(size_t));
    tensor->strides = malloc(ndim * sizeof(size_t));

    if (!tensor->shape || !tensor->strides) {
        ts_storage_release(tensor->storage);
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }

    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    calculate_strides(dtype, ndim, tensor->shape, tensor->strides);

    return tensor;
}


TsTensor* ts_tensor_from_storage(
    TsTensorStorage* storage,
    TsDType dtype,
    size_t ndim,
    const size_t* shape,
    int requires_grad,
    void* grad_fn
) {
    if (!storage) return NULL;
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

    ts_storage_retain(storage);
    tensor->storage = storage;
    if (!tensor->storage) {
        free(tensor);
        return NULL;
    }

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->numel = numel;
    tensor->requires_grad = requires_grad;
    tensor->grad_fn = grad_fn;

    tensor->shape = malloc(ndim * sizeof(size_t));
    tensor->strides = malloc(ndim * sizeof(size_t));

    if (!tensor->shape || !tensor->strides) {
        ts_storage_release(tensor->storage);
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

    if (tensor->storage) {
        ts_storage_release(tensor->storage);
        tensor->storage = NULL;
    }

    free(tensor->shape);
    free(tensor->strides);

    free(tensor);
}


int ts_tensor_is_contiguous(const TsTensor* t) {
    if (!t || t->ndim == 0) return 0;

    size_t expected = ts_dtype_itemsize(t->dtype);

    for (size_t i = t->ndim; i-- > 0; ) {
        if (t->strides[i] != expected)
            return 0;
        expected *= t->shape[i];
    }

    return 1;
}


TsTensor* ts_tensor_shallow_copy(const TsTensor* src) {
    if (!src) return NULL;
    
    return ts_tensor_from_storage(
        src->storage,
        src->dtype,
        src->ndim,
        src->shape,
        src->requires_grad,
        src->grad_fn
    );
}


TsTensor* ts_tensor_clone(const TsTensor* src) {
    if (!src || !src->storage || !src->storage->data)
        return NULL;
    
    // TODO: create ts_tensor_create_like
    TsTensor* tensor = ts_tensor_create(
        src->dtype,
        src->ndim,
        src->shape,
        src->requires_grad,
        NULL
    );
    if (!tensor) return NULL;

    memcpy(tensor->storage->data, src->storage->data, src->storage->nbytes);
    return tensor;
}
