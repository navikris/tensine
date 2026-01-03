#include "tensine/ops/kernels/mul/mul.h"
#include "tensine/ops/kernels/common/elementwise.h"

#include <stdlib.h>
#include <assert.h>
#include <stdint.h>


#define DEFINE_MUL_KERNEL(T, NAME)              \
void mul_##NAME##_cpu(                          \
    const T* restrict input_1,                  \
    const T* restrict input_2,                  \
    T* restrict output,                         \
    size_t numel                                \
) {                                             \
    for (size_t i = 0; i < numel; ++i)          \
        output[i] = input_1[i] * input_2[i];    \
}


DEFINE_MUL_KERNEL(float, f32);
DEFINE_MUL_KERNEL(double, f64);
DEFINE_MUL_KERNEL(int32_t, i32);
DEFINE_MUL_KERNEL(int64_t, i64);
