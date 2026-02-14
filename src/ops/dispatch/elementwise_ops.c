#include "tensine/core/tensor.h"
#include "tensine/ops/dispatch/elementwise_ops.h"
#include "tensine/ops/kernels/elementwise_dispatch.h"

#include <assert.h>


// TODO: Fix only second tensor getting broadcasted
static const TsTensor* broadcast_input(
    const TsTensor* input_1,
    const TsTensor* input_2
) {
    assert (input_2->ndim <= input_1->ndim);
    if (input_1->ndim > input_2->ndim) {
        return ts_tensor_broadcast(
            input_2,
            input_1->shape,
            input_1->ndim
        );
    }

    // TODO: Add more checks for shapes to decide broadcast
    for (size_t dim = 0; dim < input_1->ndim; ++dim) {
        if (input_2->shape[dim] < input_1->shape[dim]) {
            return ts_tensor_broadcast(
                input_2,
                input_1->shape,
                input_1->ndim
            );
        }
    }

    return input_2;
}


TsTensor* ts_add(
    const TsTensor* input_1,
    const TsTensor* input_2
) {
    assert(input_1 && input_2);
    assert(input_1->dtype == input_2->dtype);

    const TsTensor* broadcasted_input_2 = broadcast_input(input_1, input_2);
    if (!broadcasted_input_2) return NULL;

    TsTensor* output = ts_tensor_empty_like(input_1);
    if (!output) return NULL;
    ts_elementwise_add_dispatch(input_1, broadcasted_input_2, output);

    return output;
}


TsTensor* ts_mul(
    const TsTensor* input_1,
    const TsTensor* input_2
) {
    assert(input_1 && input_2);
    assert(input_1->dtype == input_2->dtype);

    const TsTensor* broadcasted_input_2 = broadcast_input(input_1, input_2);
    if (!broadcasted_input_2) return NULL;

    TsTensor* output = ts_tensor_empty_like(input_1);
    if (!output) return NULL;
    ts_elementwise_mul_dispatch(input_1, broadcasted_input_2, output);

    return output;
}
