#include "tensine/core/dtype.h"
#include "tensine_bindings.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;


void bind_dtype(py::module_& m) {
    py::enum_<TsDType>(m, "Dtype")
        .value("float32", TsDType::TS_DTYPE_FLOAT32)
        .value("float64", TsDType::TS_DTYPE_FLOAT64)
        .value("int32", TsDType::TS_DTYPE_INT32)
        .value("int64", TsDType::TS_DTYPE_INT64)
        .value("bool", TsDType::TS_DTYPE_BOOL)
        .value("count", TsDType::TS_DTYPE_COUNT)
        .export_values();

    py::class_<TsDTypeInfo>(m, "DtypeInfo")
        .def_readonly("dtype", &TsDTypeInfo::dtype)
        .def_readonly("name", &TsDTypeInfo::name)
        .def_readonly("itemsize", &TsDTypeInfo::itemsize)
        .def_readonly("is_floating", &TsDTypeInfo::is_floating)
        .def_readonly("is_signed", &TsDTypeInfo::is_signed);
}
