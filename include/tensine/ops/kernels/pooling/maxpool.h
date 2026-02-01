#ifndef TENSINE_OPS_MAXPOOL_H
#define TENSINE_OPS_MAXPOOL_H


#include <stddef.h>
#include <stdint.h>


void maxpool2d_f32_cpu(
    const float* restrict input,
    float* restrict output,
    const size_t* input_shape,
    const size_t* output_shape,
    const size_t* kernel_size,
    const size_t* stride,
    const size_t* padding,
    size_t ndim
);
void maxpool2d_f64_cpu(
    const double* restrict input,
    double* restrict output,
    const size_t* input_shape,
    const size_t* output_shape,
    const size_t* kernel_size,
    const size_t* stride,
    const size_t* padding,
    size_t ndim
);
void maxpool2d_i32_cpu(
    const int32_t* restrict input,
    int32_t* restrict output,
    const size_t* input_shape,
    const size_t* output_shape,
    const size_t* kernel_size,
    const size_t* stride,
    const size_t* padding,
    size_t ndim
);
void maxpool2d_i64_cpu(
    const int64_t* restrict input,
    int64_t* restrict output,
    const size_t* input_shape,
    const size_t* output_shape,
    const size_t* kernel_size,
    const size_t* stride,
    const size_t* padding,
    size_t ndim
);

#endif /* TENSINE_OPS_MAXPOOL_H */