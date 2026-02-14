#include "tensine/core/dtype.h"
#include "tensine/core/tensor.h"
#include "tensine/core/tensor_print.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>


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
    const size_t* src_offset,
    const size_t* shape,
    int ndim,
    size_t itemsize
) {
    size_t idx[TS_MAX_DIMS] = {0};

    while (1) {
        size_t src_off = 0;
        size_t dst_off = 0;

        for (int d = 0; d < ndim; ++d) {
            src_off += (idx[d] + src_offset[d]) * src_stride[d];
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
    set_strides_contiguous(
        tensor->dtype,
        tensor->ndim,
        tensor->shape,
        tensor->strides
    );

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


TsTensor* ts_tensor_to_contiguous(const TsTensor* src) {
    if (ts_tensor_is_contiguous(src)) {
        return ts_tensor_shallow_copy(src);
    };

    TsTensor* tensor = ts_tensor_empty_like(src);
    size_t src_offsets[TS_MAX_DIMS] = {0}; // HACK: till src->offsets gets used

    ts_copy_strided(
        tensor->storage->data,
        tensor->strides,
        src->storage->data,
        src->strides,
        src_offsets,
        tensor->shape,
        tensor->ndim,
        ts_dtype_itemsize(tensor->dtype)
    );

    return tensor;
}


static void _broadcast_seed_impl(
    const TsTensor* src,
    const TsTensor* dst,
    size_t src_dim,
    size_t dst_dim,
    size_t src_ptr_offset,
    size_t dst_ptr_offset,
    const size_t* num_copies
) {
    if (src_dim == (src->ndim - 1)) {
        char* src_ptr = (char*)src->storage->data + src_ptr_offset;
        char* dst_ptr = (char*)dst->storage->data + dst_ptr_offset;
        size_t nbytes = src->shape[src_dim] * src->strides[src_dim];
        for (size_t i = 0; i < num_copies[dst_dim]; ++i) {
            memcpy(dst_ptr + i * nbytes, src_ptr, nbytes);
        }
    } else {
        for (size_t i = 0; i < src->shape[src_dim]; ++i) {
            size_t new_src_offset = src_ptr_offset + src->strides[src_dim] * i;
            size_t new_dst_offset = dst_ptr_offset + dst->strides[dst_dim] * i;
            _broadcast_seed_impl(
                src,
                dst,
                src_dim + 1,
                dst_dim + 1,
                new_src_offset,
                new_dst_offset,
                num_copies
            );
        }
        char* dst_ptr = (char*)dst->storage->data + dst_ptr_offset;
        size_t nbytes = src->shape[src_dim] * dst->strides[dst_dim];
        for (size_t i = 1; i < num_copies[dst_dim]; ++i) {
            memcpy(dst_ptr + i * nbytes, dst_ptr, nbytes);
        }
    }
}


TsTensor* ts_tensor_broadcast(
    const TsTensor* src,
    const size_t* new_shape,
    size_t new_ndim
) {
    if (!src || !new_shape || (new_ndim==0)) return NULL;

    // Validating the op inputs
    assert (new_ndim <= src->ndim);
    for (size_t i = 1; i < src->ndim + 1; ++i) {
        assert (src->shape[src->ndim - i] <= new_shape[new_ndim - i]);
        assert (new_shape[new_ndim - i] % src->shape[src->ndim - i] == 0);
    }

    if (ts_tensor_is_contiguous(src)) {
        TsTensor* broadcasted_tensor = ts_tensor_create(
            src->dtype,
            new_ndim,
            new_shape,
            0,
            NULL
        );
        if (!broadcasted_tensor) return NULL;

        size_t* num_copies = malloc(new_ndim * sizeof(size_t));
        size_t unsqueezed_ndims = new_ndim - src->ndim;
        size_t leftover_copies = 0;
        for (size_t dim = 0; dim < new_ndim; dim++) {
            if (dim < unsqueezed_ndims) {
                num_copies[dim] = new_shape[dim];
                leftover_copies += new_shape[dim];
            } else {
                num_copies[dim] = new_shape[dim] / src->shape[dim - unsqueezed_ndims];
            }
        }

        _broadcast_seed_impl(
            src,
            broadcasted_tensor,
            0,
            broadcasted_tensor->ndim - src->ndim,
            0,
            0,
            num_copies
        );

        char* dst_ptr = (char*)broadcasted_tensor->storage->data;
        size_t nbytes = broadcasted_tensor->shape[unsqueezed_ndims] \
                         * broadcasted_tensor->strides[unsqueezed_ndims];
        for (size_t i = 0; i < leftover_copies; ++i) {
            memcpy(dst_ptr + nbytes * i , dst_ptr, nbytes);
        }

        return broadcasted_tensor;
    }

    // TODO: Add support for non contiguous src tensor
    // Fallback to contiguous for now
    TsTensor* src_contiguous = ts_tensor_to_contiguous(src);
    TsTensor* broadcasted_tensor = ts_tensor_broadcast(src_contiguous, new_shape, new_ndim);
    ts_tensor_free(src_contiguous);

    return broadcasted_tensor;
}


TsTensor* ts_tensor_reshape(
    const TsTensor* src,
    const size_t* new_shape,
    size_t new_ndim
) {
    if (!src || !new_shape || (new_ndim==0)) return NULL;

    if (ts_tensor_is_contiguous(src)) {
        TsTensor* viewed_tensor = ts_tensor_from_storage(
            src->storage,
            src->dtype,
            new_ndim,
            new_shape,
            src->requires_grad,
            src->grad_fn
        );
        if (!viewed_tensor) return NULL;

        viewed_tensor->ndim = new_ndim;        
        memcpy(viewed_tensor->shape, new_shape, new_ndim * sizeof(size_t));
        set_strides_contiguous(
            viewed_tensor->dtype,
            viewed_tensor->ndim,
            viewed_tensor->shape,
            viewed_tensor->strides
        );
        return viewed_tensor;
    }

    // TODO: Add support for non contiguous src tensor
    // Fallback to contiguous for now
    TsTensor* src_contiguous = ts_tensor_to_contiguous(src);
    TsTensor* viewed_tensor = ts_tensor_reshape(src_contiguous, new_shape, new_ndim);
    ts_tensor_free(src_contiguous);

    return viewed_tensor;
}


TsTensor* ts_tensor_permute(const TsTensor* src, const size_t* permute_order) {
    if (!src || !permute_order) return NULL;

    TsTensor* viewed_tensor = ts_tensor_shallow_copy(src);
    if (!viewed_tensor) return NULL;

    // assuming permute order matches ndim properly
    for (size_t dim = 0; dim < src->ndim; ++dim) {
        viewed_tensor->shape[dim] = src->shape[permute_order[dim]];
        viewed_tensor->strides[dim] = src->strides[permute_order[dim]];
    }

    return ts_tensor_to_contiguous(viewed_tensor);
}


TsTensor* ts_tensor_transpose(const TsTensor* src, size_t dim_1, size_t dim_2) {
    if (!src) return NULL;
    assert (dim_1 < src->ndim && dim_2 < src->ndim);

    TsTensor* viewed_tensor = ts_tensor_shallow_copy(src);
    if (!viewed_tensor) return NULL;

    viewed_tensor->shape[dim_1] = src->shape[dim_2];
    viewed_tensor->strides[dim_1] = src->strides[dim_2];
    viewed_tensor->shape[dim_2] = src->shape[dim_1];
    viewed_tensor->strides[dim_2] = src->strides[dim_1];

    return ts_tensor_to_contiguous(viewed_tensor);
}


TsTensor* ts_tensor_slice(const TsTensor* src, const size_t* start_idxs, const size_t* end_idxs) {
    if (!src || !start_idxs || (end_idxs==0)) return NULL;

    if (ts_tensor_is_contiguous(src)) {
        TsTensor* viewed_tensor = ts_tensor_empty_like(src);
        if (!viewed_tensor) return NULL;

        // For future reference, shallow copy instead of clone
        // TsTensor* viewed_tensor = ts_tensor_shallow_copy(src);
        // memcpy(viewed_tensor->offsets, start_idxs, viewed_tensor->ndim * sizeof(size_t));

        // Update shape and strides to match slice
        for (size_t dim = 0; dim < src->ndim; ++dim) {
            viewed_tensor->shape[dim] = end_idxs[dim] - start_idxs[dim];
        }
        set_strides_contiguous(
            viewed_tensor->dtype,
            viewed_tensor->ndim,
            viewed_tensor->shape,
            viewed_tensor->strides
        );

        // Clone the tensor data in contiguous layout
        ts_copy_strided(
            viewed_tensor->storage->data,
            viewed_tensor->strides,
            src->storage->data,
            src->strides,
            start_idxs,
            viewed_tensor->shape,
            viewed_tensor->ndim,
            ts_dtype_itemsize(viewed_tensor->dtype)
        );

        return viewed_tensor; // will return a cloned tensor for all cases currently
    }

    // TODO: Add support for non contiguous src tensor
    // Fallback to contiguous for now
    TsTensor* src_contiguous = ts_tensor_to_contiguous(src);
    TsTensor* viewed_tensor = ts_tensor_slice(src_contiguous, start_idxs, end_idxs);
    ts_tensor_free(src_contiguous);

    return viewed_tensor;
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