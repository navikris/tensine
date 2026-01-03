#include "tensine/core/dtype.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>


int main(void) {
    assert(ts_dtype_itemsize(TS_DTYPE_FLOAT32) == 4);
    assert(ts_dtype_itemsize(TS_DTYPE_FLOAT64) == 8);
    assert(ts_dtype_itemsize(TS_DTYPE_INT32) == 4);
    assert(ts_dtype_itemsize(TS_DTYPE_INT64) == 8);
    assert(strcmp(ts_dtype_name(TS_DTYPE_BOOL), "bool") == 0);
    printf("dtype info checks passed\n");

    return 0;
}
