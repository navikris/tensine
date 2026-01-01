#ifndef TENSINE_OPS_ADD_H
#define TENSINE_OPS_ADD_H


void add_f32_cpu(const float* input_1, const float* input_2, float* output, size_t numel);
void add_f64_cpu(const double* input_1, const double* input_2, double* output, size_t numel);
void add_i32_cpu(const int32_t* input_1, const int32_t* input_2, int32_t* output, size_t numel);
void add_i64_cpu(const int64_t* input_1, const int64_t* input_2, int64_t* output, size_t numel);


#endif /* TENSINE_OPS_ADD_H */