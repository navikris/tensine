#include "tensine/core/dtype.h"
#include "tensine/core/tensor.h"
#include "tensine/ops/kernels/matmul/matmul.h"

#include <stdint.h>
#include <assert.h>


void ts_matmul_dispatch(
    const TsTensor* input_1,
    const TsTensor* input_2,
    TsTensor* output
) {
    assert(input_1 && input_2 && output);
    // assuming contiguous behaviour for now
    switch (output->dtype) {
        case TS_DTYPE_FLOAT32: {
            matmul_f32_cpu(
                (float*)input_1->storage->data,
                (float*)input_2->storage->data,
                (float*)output->storage->data,
                input_1->shape,
                input_2->shape,
                output->shape,
                output->ndim
            );
            break;
        }
        case TS_DTYPE_FLOAT64: {
            matmul_f64_cpu(
                (double*)input_1->storage->data,
                (double*)input_2->storage->data,
                (double*)output->storage->data,
                input_1->shape,
                input_2->shape,
                output->shape,
                output->ndim
            );
            break;
        }
        case TS_DTYPE_INT32: {
            matmul_i32_cpu(
                (int32_t*)input_1->storage->data,
                (int32_t*)input_2->storage->data,
                (int32_t*)output->storage->data,
                input_1->shape,
                input_2->shape,
                output->shape,
                output->ndim
            );
            break;
        }
        case TS_DTYPE_INT64: {
            matmul_i64_cpu(
                (int64_t*)input_1->storage->data,
                (int64_t*)input_2->storage->data,
                (int64_t*)output->storage->data,
                input_1->shape,
                input_2->shape,
                output->shape,
                output->ndim
            );
            break;
        }
        default:
            assert(0 && "dtype not supported for matmul");
    }
}
