#include <assert.h>

#include "../../core/dtype.h"
#include "../../core/tensor.h"
#include "add/add.h"


void ts_elementwise_add_dispatch(
    const TsTensor* input_1,
    const TsTensor* input_2,
    TsTensor* output
) {
    assert(input_1 && input_2 && output);
    // assuming contiguous behaviour for now
    switch (output->dtype) {
        case TS_DTYPE_FLOAT32: {
            add_f32_cpu(
                (float*)input_1->storage->data,
                (float*)input_2->storage->data,
                (float*)output->storage->data,
                output->numel
            );
            break;
        }
        default:
            assert(0 && "dtype not supported for add");
    }
}
