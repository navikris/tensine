#include <assert.h>

#include "../../core/tensor.h"
#include "elementwise_ops.h"
#include "../kernels/elementwise_dispatch.h"


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