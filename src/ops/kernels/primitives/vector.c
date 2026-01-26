#include "tensine/ops/kernels/primitives/vector.h"
#include "tensine/ops/kernels/common/elementwise.h"

#include <stdlib.h>
#include <assert.h>
#include <stdint.h>


#define DEFINE_VECTOR_MAC_KERNEL(T, NAME)                               \
void vector_mac_##NAME##_cpu(                                           \
    const T* restrict input_1,                                          \
    const T* restrict input_2,                                          \
    T* restrict output,                                                 \
    size_t input_1_stride,                                              \
    size_t input_2_stride,                                              \
    size_t numel                                                        \
) {                                                                     \
    size_t input_1_offset = (size_t)0;                                  \
    size_t input_2_offset = (size_t)0;                                  \
    *output = input_1[0] * input_2[0];                                  \
    for (size_t i = 1; i < numel; ++i) {                                \
        input_1_offset += input_1_stride;                               \
        input_2_offset += input_2_stride;                               \
        *output += input_1[input_1_offset] * input_2[input_2_offset];   \
    }                                                                   \
}


DEFINE_VECTOR_MAC_KERNEL(float, f32);
DEFINE_VECTOR_MAC_KERNEL(double, f64);
DEFINE_VECTOR_MAC_KERNEL(int32_t, i32);
DEFINE_VECTOR_MAC_KERNEL(int64_t, i64);