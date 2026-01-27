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
void matmul_f64_cpu(
    const double* restrict input_1,
    const double* restrict input_2,
    double* restrict output,
    const size_t* input_1_shape,
    const size_t* input_2_shape,
    const size_t* output_shape,
    size_t ndim
);
void matmul_i32_cpu(
    const int32_t* restrict input_1,
    const int32_t* restrict input_2,
    int32_t* restrict output,
    const size_t* input_1_shape,
    const size_t* input_2_shape,
    const size_t* output_shape,
    size_t ndim
);
void matmul_i64_cpu(
    const int64_t* restrict input_1,
    const int64_t* restrict input_2,
    int64_t* restrict output,
    const size_t* input_1_shape,
    const size_t* input_2_shape,
    const size_t* output_shape,
    size_t ndim
);

#endif /* TENSINE_OPS_MATMUL_H */