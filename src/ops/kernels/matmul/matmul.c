#include "tensine/ops/kernels/matmul/matmul.h"
#include "tensine/ops/kernels/primitives/vector.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>


size_t* _get_strides(const size_t* shape, size_t ndim) {
    size_t* strides = (size_t*)malloc(ndim * sizeof(size_t));
    
    strides[ndim - 1] = 1;
    for (size_t dim = (ndim - 1); dim-- > 0;) {
        strides[dim] = shape[dim + 1] * strides[dim + 1];
    }
    return strides;
}


#define T float
#define NAME f32
#include "matmul.inl"
#undef T
#undef NAME


#define T double
#define NAME f64
#include "matmul.inl"
#undef T
#undef NAME


#define T int32_t
#define NAME i32
#include "matmul.inl"
#undef T
#undef NAME


#define T int64_t
#define NAME i64
#include "matmul.inl"
#undef T
#undef NAME
