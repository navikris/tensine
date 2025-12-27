#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "../tensor.h"
#include "../dtype.h"


static void test_tensor_create_basic(void) {
    size_t shape[2] = {2, 3};

    for (TsDType DTYPE = TS_DTYPE_FLOAT32; DTYPE < TS_DTYPE_COUNT; ++DTYPE) {
        TsTensor* t = ts_tensor_create(
            DTYPE,
            2,
            shape,
            0,
            NULL
        );

        assert(t != NULL);
        assert(t->dtype == DTYPE);
        assert(t->ndim == 2);
        assert(t->numel == 6);
        assert(t->storage != NULL);
        assert(t->shape[0] == 2 && t->shape[1] == 3);
        assert(ts_tensor_is_contiguous(t));
        ts_tensor_free(t);
    }
}


static void test_tensor_from_buffer(void) {
    float buffer[6] = {1, 2, 3, 4, 5, 6};
    size_t shape[2] = {2, 3};

    TsTensor* t = ts_tensor_from_buffer(
        buffer,
        TS_DTYPE_FLOAT32,
        2,
        shape,
        0,
        NULL
    );

    assert(t != NULL);
    assert(t->storage->data == buffer);
    assert(t->storage->owns_data == 0);

    ts_tensor_free(t);
}


int main(void) {
    test_tensor_create_basic();
    printf("test_tensor_create_basic passed\n");
    test_tensor_from_buffer();
    printf("test_tensor_from_buffer passed\n");
    return 0;
}
