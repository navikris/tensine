#include "tensine/ops/kernels/activation/activation.h"
#include "tensine/ops/kernels/common/tensor_utils.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>


#define T float
#define NAME f32
#include "softmax.inl"
#undef T
#undef NAME


#define T double
#define NAME f64
#include "softmax.inl"
#undef T
#undef NAME


#define T int32_t
#define NAME i32
#include "softmax.inl"
#undef T
#undef NAME


#define T int64_t
#define NAME i64
#include "softmax.inl"
#undef T
#undef NAME
