#ifndef TENSINE_OPS_ACTIVATION_H
#define TENSINE_OPS_ACTIVATION_H


#include <stddef.h>
#include <stdint.h>


void softmax_f32_cpu(
    const float* restrict input,
    float* restrict output,
    const size_t* input_shape,
    size_t ndim
);
void softmax_f64_cpu(
    const double* restrict input,
    double* restrict output,
    const size_t* input_shape,
    size_t ndim
);
void softmax_i32_cpu(
    const int32_t* restrict input,
    int32_t* restrict output,
    const size_t* input_shape,
    size_t ndim
);
void softmax_i64_cpu(
    const int64_t* restrict input,
    int64_t* restrict output,
    const size_t* input_shape,
    size_t ndim
);

#endif /* TENSINE_OPS_ACTIVATION_H */