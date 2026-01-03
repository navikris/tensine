#include "tensine/core/dtype.h"
#include "tensine/core/tensor.h"
#include "tensine/core/tensor_print.h"

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>


static size_t _get_numel_stride(const size_t byte_stride, TsDType dtype) {
    return byte_stride / ts_dtype_itemsize(dtype);
}


static void print_element(const void* data, TsDType dtype, size_t index) {
    switch (dtype) {
        case TS_DTYPE_FLOAT32:
            printf("%f ", ((float*)data)[index]);
            break;
        case TS_DTYPE_FLOAT64:
            printf("%e ", ((double*)data)[index]);
            break;
        case TS_DTYPE_INT32:
            printf("%d ", ((int32_t*)data)[index]);
            break;
        case TS_DTYPE_INT64:
            printf("%ld ", ((int64_t*)data)[index]);
            break;
        case TS_DTYPE_BOOL:
            printf("%d ", ((uint8_t*)data)[index]);
            break;
        case TS_DTYPE_COUNT:
            break;
    }
}


static void print_recursive(
    const TsTensor* t,
    size_t dim,
    size_t offset
) {
    if (dim == t->ndim - 1) {
        printf("[");
        for (size_t i = 0; i < t->shape[dim]; ++i) {
            size_t idx = offset + i * _get_numel_stride(t->strides[dim], t->dtype);
            print_element(t->storage->data, t->dtype, idx);
        }
        printf("]");
        return;
    }

    printf("[");
    for (size_t i = 0; i < t->shape[dim]; ++i) {
        print_recursive(t, dim + 1, offset + i * _get_numel_stride(t->strides[dim], t->dtype));
        if (i + 1 < t->shape[dim]) printf(",\n");
    }
    printf("]");
}


void ts_tensor_print_impl(const TsTensor* t) {
    printf("Tensor(shape=(");
    for (size_t i = 0; i < t->ndim; ++i) {
        printf("%zu", t->shape[i]);
        if (i + 1 < t->ndim) printf(", ");
    }
    printf("), dtype=%s)\n", ts_dtype_name(t->dtype));

    print_recursive(t, 0, 0);
    printf("\n");
}
