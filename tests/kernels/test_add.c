#include "tensine/core/dtype.h"
#include "tensine/core/tensor.h"
#include "tensine/ops/dispatch/elementwise_ops.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>


typedef struct {
    size_t ndim;
    const size_t* shape;
} TestAdd;


static const size_t ndim1 = 1;
static const size_t ndim2 = 2;
static const size_t ndim3 = 3;
static const size_t ndim4 = 4;


static const size_t shape1[] = {6};
static const size_t shape2[] = {2, 3};
static const size_t shape3[] = {1, 2, 3};
static const size_t shape4[] = {2, 3, 1, 2};


static size_t _get_numel(const size_t* shape, size_t ndim) {
    size_t numel = 1;
    for (size_t i = 0; i < ndim; ++i) {
        assert(shape[i] != 0);
        assert(numel < SIZE_MAX / shape[i]);
        numel *= shape[i];
    }

    return numel;
}


static void test_add_fp32(void) {
    TestAdd TEST_CASES[] = {
        {ndim1, shape1},
        {ndim2, shape2},
        {ndim3, shape3},
        {ndim4, shape4},
    };
    float in_buffer1[] = {1.2, 4.3, 5.2, 11.4, 6.7, 2.5, 7.2, 8.3, 9.0, 3.7, 3.14, 6.66};
    float in_buffer2[] = {1.2, 4.3, 5.2, 11.4, 6.7, 2.5, 7.2, 8.3, 9.0, 3.7, 3.14, 6.66};

    int num_test_cases = (int)(sizeof(TEST_CASES) / sizeof(TEST_CASES[0]));
    int test_id = 0;
    
    // Tests for ts_add()
    printf("\n========== Testing OP:Add ==========\n");
    while(test_id < num_test_cases) {
        printf("-TestID:%d\n", test_id);
        TsTensor* tensor_a = ts_tensor_from_buffer(
            in_buffer1,
            TS_DTYPE_FLOAT32,
            TEST_CASES[test_id].ndim,
            TEST_CASES[test_id].shape,
            0,
            NULL
        );
        TsTensor* tensor_b = ts_tensor_from_buffer(
            in_buffer2,
            TS_DTYPE_FLOAT32,
            TEST_CASES[test_id].ndim,
            TEST_CASES[test_id].shape,
            0,
            NULL
        );
        assert(tensor_a && tensor_b);
        assert(tensor_a->storage != NULL);
        assert(tensor_b->storage != NULL);
        assert(ts_tensor_is_contiguous(tensor_a));
        assert(ts_tensor_is_contiguous(tensor_b));

        TsTensor* output = ts_add(tensor_a, tensor_b);
        assert(output);
        assert(output->storage != NULL);
        assert(ts_tensor_is_contiguous(output));

        // Validating
        float* out_buffer = (float*)output->storage->data;
        size_t numel = _get_numel(TEST_CASES[test_id].shape, TEST_CASES[test_id].ndim);
        for (size_t i = 0; i < numel; ++i) {
            assert(out_buffer[i] == in_buffer1[i] * 2.0f);
        }

        ts_tensor_free(tensor_a);
        ts_tensor_free(tensor_b);
        ts_tensor_free(output);

        printf("\t|NDIM:%d", ((int)TEST_CASES[test_id].ndim));
        printf("\t|DTYPE:%s", ts_dtype_name(TS_DTYPE_FLOAT32));
        printf("\t PASSED\n");
        ++test_id;
    }
}


int main(void) {
    test_add_fp32();
    return 0;
}
