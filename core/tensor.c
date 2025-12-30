#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "dtype.h"
#include "tensor.h"
#include "tensor_print.h"


static size_t _get_numel(const size_t* shape, size_t ndim) {
    size_t numel = 1;
    for (size_t i = 0; i < ndim; ++i) {
        assert(shape[i] != 0);
        assert(numel < SIZE_MAX / shape[i]);
        numel *= shape[i];
    }

    return numel;
}


static size_t _get_num_bytes(TsDType dtype, size_t numel) {
    size_t itemsize = ts_dtype_itemsize(dtype);
    if (itemsize == 0) return 0;

    if (numel != 0 && ((SIZE_MAX / numel) < itemsize))
        return 0;  // overflow

    return numel * itemsize;
}


static void set_strides_contiguous(
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


static void ts_copy_strided(
    void* dst,
    const size_t* dst_stride,
    const void* src,
    const size_t* src_stride,
    const size_t* shape,
    int ndim,
    size_t itemsize
) {
    size_t idx[TS_MAX_DIMS] = {0};

    while (1) {
        size_t src_off = 0;
        size_t dst_off = 0;

        for (int d = 0; d < ndim; ++d) {
            src_off += idx[d] * src_stride[d];
            dst_off += idx[d] * dst_stride[d];
        }

        memcpy(
            (char*)dst + dst_off,
            (const char*)src + src_off,
            itemsize
        );

        // increment multi-dimensional index
        int d = ndim - 1;
        while (d >= 0) {
            idx[d]++;
            if (idx[d] < shape[d])
                break;
            idx[d] = 0;
            d--;
        }
        if (d < 0)
            break;
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

    size_t numel = _get_numel(shape, ndim);

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
    set_strides_contiguous(dtype, ndim, tensor->shape, tensor->strides);

    return tensor;
}


TsTensor* ts_tensor_empty_like(const TsTensor* src) {
    if (!src) return NULL;

    TsTensor* tensor = malloc(sizeof(TsTensor));
    if (!tensor) return NULL;

    tensor->storage = ts_storage_create(_get_num_bytes(src->dtype, src->numel));
    if (!tensor->storage) {
        free(tensor);
        return NULL;
    }

    tensor->dtype = src->dtype;
    tensor->ndim = src->ndim;
    tensor->numel = src->numel;
    tensor->requires_grad = src->requires_grad;
    tensor->grad_fn = NULL;

    tensor->shape = malloc(tensor->ndim * sizeof(size_t));
    tensor->strides = malloc(tensor->ndim * sizeof(size_t));

    if (!tensor->shape || !tensor->strides) {
        ts_storage_release(tensor->storage);
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }

    // logical shape will be copied over, strides are recalculated for contiguous tensor
    memcpy(tensor->shape, src->shape, tensor->ndim * sizeof(size_t));
    set_strides_contiguous(tensor->dtype, tensor->ndim, tensor->shape, tensor->strides);

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

    size_t numel = _get_numel(shape, ndim);

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
    set_strides_contiguous(dtype, ndim, tensor->shape, tensor->strides);

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

    size_t numel = _get_numel(shape, ndim);

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
    set_strides_contiguous(dtype, ndim, tensor->shape, tensor->strides);

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


TsTensor* ts_tensor_to_contiguous(TsTensor* src) {
    if (ts_tensor_is_contiguous(src)) {
        return ts_tensor_shallow_copy(src);
    };

    TsTensor* tensor = ts_tensor_empty_like(src);

    ts_copy_strided(
        tensor->storage->data,
        tensor->strides,
        src->storage->data,
        src->strides,
        tensor->shape,
        tensor->ndim,
        ts_dtype_itemsize(tensor->dtype)
    );

    return tensor;
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
    
    TsTensor* tensor = ts_tensor_empty_like(src);
    if (!tensor) return NULL;

    memcpy(tensor->storage->data, src->storage->data, src->storage->nbytes);
    return tensor;
}


void ts_tensor_print(const TsTensor* t) {
    if (!t) return;
    ts_tensor_print_impl(t);
}