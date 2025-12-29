#include <stdlib.h>
#include <assert.h>

#include "add.h"
#include "../common/elementwise.h"


float* add_f32_cpu(
    float* input_1,
    float* input_2,
    float* output,
    size_t numel
) {
    assert(input_1 && input_2 && output);
    assert(numel);

    for (size_t i=0; i < numel; i++) {
        *(output + i) = *(input_1 + i) + *(input_2 + i);
    }

    return output;
}
