#include "tensine/core/dtype.h"
#include "tensine/core/tensor.h"
#include "tensine/ops/kernels/pooling/maxpool.h"

#include <stdint.h>
#include <assert.h>


void ts_maxpool2d_dispatch(
    const TsTensor* input,
    TsTensor* output,
    const size_t* kernel_size,
    const size_t* stride,
    const size_t* padding
) {
    assert(input && output);
    // assuming contiguous behaviour for now
    switch (output->dtype) {
        case TS_DTYPE_FLOAT32: {
            maxpool2d_f32_cpu(
                (float*)input->storage->data,
                (float*)output->storage->data,
                input->shape,
                output->shape,
                kernel_size,
                stride,
                padding,
                output->ndim
            );
            break;
        }
        case TS_DTYPE_FLOAT64: {
            maxpool2d_f64_cpu(
                (double*)input->storage->data,
                (double*)output->storage->data,
                input->shape,
                output->shape,
                kernel_size,
                stride,
                padding,
                output->ndim
            );
            break;
        }
        case TS_DTYPE_INT32: {
            maxpool2d_i32_cpu(
                (int32_t*)input->storage->data,
                (int32_t*)output->storage->data,
                input->shape,
                output->shape,
                kernel_size,
                stride,
                padding,
                output->ndim
            );
            break;
        }
        case TS_DTYPE_INT64: {
            maxpool2d_i64_cpu(
                (int64_t*)input->storage->data,
                (int64_t*)output->storage->data,
                input->shape,
                output->shape,
                kernel_size,
                stride,
                padding,
                output->ndim
            );
            break;
        }
        default:
            assert(0 && "dtype not supported for maxpool2d");
    }
}
