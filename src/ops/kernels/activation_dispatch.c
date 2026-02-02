#include "tensine/core/dtype.h"
#include "tensine/core/tensor.h"
#include "tensine/ops/kernels/activation/activation.h"

#include <stdint.h>
#include <assert.h>


void ts_softmax_dispatch(
    const TsTensor* input,
    TsTensor* output
) {
    assert(input && output);
    // assuming contiguous behaviour for now
    switch (output->dtype) {
        case TS_DTYPE_FLOAT32: {
            softmax_f32_cpu(
                (float*)input->storage->data,
                (float*)output->storage->data,
                input->shape,
                output->ndim
            );
            break;
        }
        case TS_DTYPE_FLOAT64: {
            softmax_f64_cpu(
                (double*)input->storage->data,
                (double*)output->storage->data,
                input->shape,
                output->ndim
            );
            break;
        }
        case TS_DTYPE_INT32: {
            softmax_i32_cpu(
                (int32_t*)input->storage->data,
                (int32_t*)output->storage->data,
                input->shape,
                output->ndim
            );
            break;
        }
        case TS_DTYPE_INT64: {
            softmax_i64_cpu(
                (int64_t*)input->storage->data,
                (int64_t*)output->storage->data,
                input->shape,
                output->ndim
            );
            break;
        }
        default:
            assert(0 && "dtype not supported for softmax");
    }
}
