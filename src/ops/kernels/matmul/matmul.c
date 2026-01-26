#include "tensine/ops/kernels/matmul/matmul.h"
#include "tensine/ops/kernels/primitives/vector.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>


void matmul_f32_cpu(
    const float* restrict input_1,
    const float* restrict input_2,
    float* restrict output,
    const size_t* input_1_shape,
    const size_t* input_2_shape,
    const size_t* output_shape,
    size_t ndim
) {
    assert(ndim == 2);
    size_t input_1_stride = 1; 
    size_t input_2_stride = input_2_shape[1];
    size_t numel_to_macc = input_1_shape[1];

    for (size_t row = 0; row < output_shape[0]; ++row) {
        for (size_t col = 0; col < output_shape[1]; ++col) {
            size_t input_1_offset = row * input_1_shape[1];
            size_t input_2_offset = col;
            size_t output_offset = row * output_shape[1] + col;
            vector_mac_f32_cpu(
                (input_1 + input_1_offset),
                (input_2 + input_2_offset),
                (output + output_offset),
                input_1_stride,
                input_2_stride,
                numel_to_macc
            );
        }
    }
}
