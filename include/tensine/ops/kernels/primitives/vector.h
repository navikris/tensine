#ifndef TENSINE_OPS_VECTOR_H
#define TENSINE_OPS_VECTOR_H


#include <stddef.h>
#include <stdint.h>


void vector_mac_f32_cpu(const float* input_1, const float* input_2, float* output, size_t input_1_stride, size_t input_2_stride, size_t numel);
void vector_mac_f64_cpu(const double* input_1, const double* input_2, double* output, size_t input_1_stride, size_t input_2_stride, size_t numel);
void vector_mac_i32_cpu(const int32_t* input_1, const int32_t* input_2, int32_t* output, size_t input_1_stride, size_t input_2_stride, size_t numel);
void vector_mac_i64_cpu(const int64_t* input_1, const int64_t* input_2, int64_t* output, size_t input_1_stride, size_t input_2_stride, size_t numel);


#endif /* TENSINE_OPS_VECTOR_H */