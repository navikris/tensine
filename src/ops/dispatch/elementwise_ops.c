#include "tensine/core/tensor.h"
#include "tensine/ops/dispatch/elementwise_ops.h"
#include "tensine/ops/kernels/elementwise_dispatch.h"

#include <assert.h>


TsTensor* ts_add(
    const TsTensor* input_1,
    const TsTensor* input_2
) {
    assert(input_1 && input_2);
    assert(input_1->dtype == input_2->dtype);
    assert(input_1->numel == input_2->numel);

    TsTensor* output = ts_tensor_empty_like(input_1);
    if (!output) return NULL;
    ts_elementwise_add_dispatch(input_1, input_2, output);

    return output;
}


TsTensor* ts_mul(
    const TsTensor* input_1,
    const TsTensor* input_2
) {
    assert(input_1 && input_2);
    assert(input_1->dtype == input_2->dtype);
    assert(input_1->numel == input_2->numel);

    TsTensor* output = ts_tensor_empty_like(input_1);
    if (!output) return NULL;
    ts_elementwise_mul_dispatch(input_1, input_2, output);

    return output;
}