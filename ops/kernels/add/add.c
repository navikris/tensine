#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

#include "add.h"
#include "../common/elementwise.h"


#define DEFINE_ADD_KERNEL(T, NAME)              \
void add_##NAME##_cpu(                          \
    const T* restrict input_1,                  \
    const T* restrict input_2,                  \
    T* restrict output,                         \
    size_t numel                                \
) {                                             \
    for (size_t i = 0; i < numel; ++i)          \
        output[i] = input_1[i] + input_2[i];    \
}


DEFINE_ADD_KERNEL(float, f32);
DEFINE_ADD_KERNEL(double, f64);
DEFINE_ADD_KERNEL(int32_t, i32);
DEFINE_ADD_KERNEL(int64_t, i64);
