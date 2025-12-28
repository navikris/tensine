#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "../tensor.h"
#include "../dtype.h"


typedef struct {
    size_t ndim;
    const size_t* shape;
} TestTensorCreate;


static const size_t ndim1 = 1;
static const size_t ndim2 = 2;
static const size_t ndim3 = 3;
static const size_t ndim4 = 4;


static const size_t shape1[] = {9};
static const size_t shape2[] = {2, 3};
static const size_t shape3[] = {1, 2, 3};
static const size_t shape4[] = {2, 3, 1, 2};


static void test_tensor_create(void) {
    TestTensorCreate TEST_CASES[] = {
        {ndim1, shape1},
        {ndim2, shape2},
        {ndim3, shape3},
        {ndim4, shape4},
    };
    int num_test_cases = (int)(sizeof(TEST_CASES) / sizeof(TEST_CASES[0]));
    int test_id = 0;
    
    // Tests for ts_tensor_create()
    printf("\n========== Testing Tensor Basic ==========\n");
    while(test_id < num_test_cases) {
        printf("-TestID:%d\n", test_id);
        for (TsDType DTYPE = TS_DTYPE_FLOAT32; DTYPE < TS_DTYPE_COUNT; ++DTYPE) {
            TsTensor* tensor_basic = ts_tensor_create(
                DTYPE,
                TEST_CASES[test_id].ndim,
                TEST_CASES[test_id].shape,
                0,
                NULL
            );
            assert(tensor_basic != NULL);
            assert(tensor_basic->ndim == TEST_CASES[test_id].ndim);
            assert(tensor_basic->storage != NULL);
            assert(ts_tensor_is_contiguous(tensor_basic));

            ts_tensor_free(tensor_basic);

            printf("\t|NDIM:%d", ((int)TEST_CASES[test_id].ndim));
            printf("\t|DTYPE:%s", ts_dtype_name(DTYPE));
            printf("\t PASSED\n");
        }
        ++test_id;
    }

    // Tests for ts_tensor_from_buffer()
    test_id = 0;
    float buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    printf("\n========== Testing Tensor From Buffer ==========\n");
    while(test_id < num_test_cases) {
        printf("-TestID:%d\n", test_id);
        for (TsDType DTYPE = TS_DTYPE_FLOAT32; DTYPE < TS_DTYPE_COUNT; ++DTYPE) {
            TsTensor* tensor_from_buffer = ts_tensor_from_buffer(
                buffer,
                DTYPE,
                TEST_CASES[test_id].ndim,
                TEST_CASES[test_id].shape,
                0,
                NULL
            );
            assert(tensor_from_buffer != NULL);
            assert(tensor_from_buffer->ndim == TEST_CASES[test_id].ndim);
            assert(tensor_from_buffer->storage != NULL);
            assert(ts_tensor_is_contiguous(tensor_from_buffer));

            ts_tensor_free(tensor_from_buffer);

            printf("\t|NDIM:%d", ((int)TEST_CASES[test_id].ndim));
            printf("\t|DTYPE:%s", ts_dtype_name(DTYPE));
            printf("\t PASSED\n");
        }
        ++test_id;
    }
}


static void test_tensor_shallow_copy(void) {
    TestTensorCreate TEST_CASES[] = {
        {ndim1, shape1},
        {ndim2, shape2},
        {ndim3, shape3},
        {ndim4, shape4},
    };
    int num_test_cases = (int)(sizeof(TEST_CASES) / sizeof(TEST_CASES[0]));
    int test_id = 0;
    
    // Tests for ts_tensor_shallow_copy()
    printf("\n========== Testing Tensor Shallow Copy ==========\n");
    while(test_id < num_test_cases) {
        printf("-TestID:%d\n", test_id);
        for (TsDType DTYPE = TS_DTYPE_FLOAT32; DTYPE < TS_DTYPE_COUNT; ++DTYPE) {
            TsTensor* tensor_basic = ts_tensor_create(
                DTYPE,
                TEST_CASES[test_id].ndim,
                TEST_CASES[test_id].shape,
                0,
                NULL
            );
            assert(tensor_basic != NULL);
            assert(tensor_basic->storage != NULL);
            assert(tensor_basic->ndim == (TEST_CASES[test_id].ndim));
            assert(ts_tensor_is_contiguous(tensor_basic));

            TsTensor* tensor_copy = ts_tensor_shallow_copy(tensor_basic);
            assert(tensor_copy != NULL);
            assert(tensor_copy->storage != NULL);
            assert(tensor_copy->ndim == (TEST_CASES[test_id].ndim));
            assert(ts_tensor_is_contiguous(tensor_copy));

            assert(tensor_basic->storage == tensor_copy->storage);
            assert(tensor_basic->storage->ref_count == 2);

            ts_tensor_free(tensor_basic);
            ts_tensor_free(tensor_copy);

            printf("\t|NDIM:%d", ((int)TEST_CASES[test_id].ndim));
            printf("\t|DTYPE:%s", ts_dtype_name(DTYPE));
            printf("\t PASSED\n");
        }
        ++test_id;
    }
}


static void test_tensor_clone(void) {
    TestTensorCreate TEST_CASES[] = {
        {ndim1, shape1},
        {ndim2, shape2},
        {ndim3, shape3},
        {ndim4, shape4},
    };
    int num_test_cases = (int)(sizeof(TEST_CASES) / sizeof(TEST_CASES[0]));
    int test_id = 0;
    
    // Tests for ts_tensor_clone()
    printf("\n========== Testing Tensor Clone ==========\n");
    while(test_id < num_test_cases) {
        printf("-TestID:%d\n", test_id);
        for (TsDType DTYPE = TS_DTYPE_FLOAT32; DTYPE < TS_DTYPE_COUNT; ++DTYPE) {
            TsTensor* tensor_basic = ts_tensor_create(
                DTYPE,
                TEST_CASES[test_id].ndim,
                TEST_CASES[test_id].shape,
                0,
                NULL
            );
            assert(tensor_basic != NULL);
            assert(tensor_basic->storage != NULL);
            assert(tensor_basic->ndim == (TEST_CASES[test_id].ndim));
            assert(tensor_basic->storage->ref_count == 1);
            assert(tensor_basic->storage->owns_data == 1);
            assert(ts_tensor_is_contiguous(tensor_basic));

            TsTensor* tensor_clone = ts_tensor_clone(tensor_basic);
            assert(tensor_clone != NULL);
            assert(tensor_clone->storage != NULL);
            assert(tensor_clone->ndim == (TEST_CASES[test_id].ndim));
            assert(tensor_clone->storage->ref_count == 1);
            assert(tensor_clone->storage->owns_data == 1);
            assert(ts_tensor_is_contiguous(tensor_clone));

            ts_tensor_free(tensor_basic);
            ts_tensor_free(tensor_clone);

            printf("\t|NDIM:%d", ((int)TEST_CASES[test_id].ndim));
            printf("\t|DTYPE:%s", ts_dtype_name(DTYPE));
            printf("\t PASSED\n");
        }
        ++test_id;
    }
}


int main(void) {
    test_tensor_create();
    test_tensor_shallow_copy();
    test_tensor_clone();
    return 0;
}
