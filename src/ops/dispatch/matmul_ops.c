#include "tensine/core/tensor.h"
#include "tensine/ops/dispatch/matmul_ops.h"
#include "tensine/ops/kernels/matmul_dispatch.h"

#include <assert.h>
#include <string.h>
#include <stdlib.h>


TsTensor* ts_matmul(
    const TsTensor* input_1,
    const TsTensor* input_2
) {
    assert(input_1 && input_2);
    assert(input_1->dtype == input_2->dtype);
    assert(input_1->ndim == input_2->ndim);
    assert(input_1->ndim == 2);
    assert(input_1->shape[input_1->ndim - 1] == input_2->shape[input_2->ndim - 2]);

    size_t* output_shape = (size_t*)malloc(sizeof(size_t) * (input_1->ndim));
    memcpy(output_shape, input_1->shape, sizeof(size_t) * (input_1->ndim));
    memcpy(output_shape + (input_1->ndim - 1), input_1->shape + (input_1->ndim - 1), sizeof(size_t));
    TsTensor* output = ts_tensor_create(
        input_1->dtype,
        input_1->ndim,
        output_shape,
        0,
        NULL
    );
    if (!output) return NULL;
    ts_matmul_dispatch(input_1, input_2, output);

    return output;
}
