#include "tensine/core/tensor.h"
#include "tensine/ops/dispatch/pooling_ops.h"
#include "tensine/ops/kernels/pooling_dispatch.h"

#include <assert.h>
#include <string.h>
#include <stdlib.h>


size_t get_out_dim_numel(size_t numel, size_t kernel_size, size_t stride, size_t padding) {
    return ((numel + 2 * padding - kernel_size) / stride + 1);
}


TsTensor* ts_maxpool2d(
    const TsTensor* input,
    const size_t* kernel_size,
    const size_t* stride,
    const size_t* padding
) {
    assert(input);
    assert(kernel_size != 0);
    assert(stride != 0);

    size_t ndim = input->ndim;
    size_t* output_shape = (size_t*)malloc(ndim * sizeof(size_t));
    if (!output_shape) {
        return NULL;
    }
    memcpy(output_shape, input->shape, ndim * sizeof(size_t));
    output_shape[ndim - 2] = get_out_dim_numel(
        input->shape[ndim - 2],
        kernel_size[1],
        stride[1],
        padding[1]
    );
    output_shape[ndim - 3] = get_out_dim_numel(
        input->shape[ndim - 3],
        kernel_size[0],
        stride[0],
        padding[0]
    );

    TsTensor* output = ts_tensor_create(
        input->dtype,
        input->ndim,
        output_shape,
        0,
        NULL
    );
    if (!output) return NULL;
    ts_maxpool2d_dispatch(input, output, kernel_size, stride, padding);

    return output;
}
