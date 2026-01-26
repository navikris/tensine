#ifndef TENSINE_OPS_MATMUL_H
#define TENSINE_OPS_MATMUL_H


#include <stddef.h>
#include <stdint.h>


void matmul_f32_cpu(
    const float* restrict input_1,
    const float* restrict input_2,
    float* restrict output,
    const size_t* input_1_shape,
    const size_t* input_2_shape,
    const size_t* output_shape,
    size_t ndim
);

#endif /* TENSINE_OPS_MATMUL_H */