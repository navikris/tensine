#include "tensine/ops/kernels/common/tensor_utils.h"

#include <stdlib.h>


size_t* get_strides(const size_t* shape, size_t ndim) {
    size_t* strides = (size_t*)malloc(ndim * sizeof(size_t));
    
    strides[ndim - 1] = 1;
    for (size_t dim = (ndim - 1); dim-- > 0;) {
        strides[dim] = shape[dim + 1] * strides[dim + 1];
    }
    return strides;
}
