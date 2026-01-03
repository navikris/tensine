#ifndef TENSINE_DTYPE_H
#define TENSINE_DTYPE_H


#include <stddef.h>


typedef enum {
    TS_DTYPE_FLOAT32 = 0,
    TS_DTYPE_FLOAT64,
    TS_DTYPE_INT32,
    TS_DTYPE_INT64,
    TS_DTYPE_BOOL,
    TS_DTYPE_COUNT
} TsDType;


typedef struct {
    TsDType dtype;
    const char* name;
    size_t itemsize;
    int is_floating;
    int is_signed;
} TsDTypeInfo;


/* Public API */
const TsDTypeInfo* ts_dtype_info(TsDType dt);
const char* ts_dtype_name(TsDType dt);
size_t ts_dtype_itemsize(TsDType dt);
int ts_dtype_is_floating(TsDType dt);
int ts_dtype_is_signed(TsDType dt);


#endif /* TENSINE_DTYPE_H */
