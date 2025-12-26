#include "dtype.h"


static const TsDTypeInfo TS_DTYPE_TABLE[TS_DTYPE_COUNT] = {
    [TS_DTYPE_FLOAT32] = {TS_DTYPE_FLOAT32, "float32", 4, 1, 1},
    [TS_DTYPE_FLOAT64] = {TS_DTYPE_FLOAT64, "float64", 8, 1, 1},
    [TS_DTYPE_INT32]   = {TS_DTYPE_INT32,   "int32",   4, 0, 1},
    [TS_DTYPE_INT64]   = {TS_DTYPE_INT64,   "int64",   8, 0, 1},
    [TS_DTYPE_BOOL]    = {TS_DTYPE_BOOL,    "bool",    1, 0, 0},
};


const TsDTypeInfo* ts_dtype_info(TsDType dt) {
    if ((unsigned)dt >= TS_DTYPE_COUNT)
        return NULL;
    return &TS_DTYPE_TABLE[dt];
}


const char* ts_dtype_name(TsDType dt) {
    const TsDTypeInfo* info = ts_dtype_info(dt);
    return info ? info->name : "";
}


size_t ts_dtype_itemsize(TsDType dt) {
    const TsDTypeInfo* info = ts_dtype_info(dt);
    return info ? info->itemsize : (size_t)0;
}


int ts_dtype_is_floating(TsDType dt) {
    const TsDTypeInfo* info = ts_dtype_info(dt);
    return info ? info->is_floating : 0;
}


int ts_dtype_is_signed(TsDType dt) {
    const TsDTypeInfo* info = ts_dtype_info(dt);
    return info ? info->is_signed : 0;
}
