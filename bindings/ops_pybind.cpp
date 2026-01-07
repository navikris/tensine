#include "tensine/core/tensor.h"
#include "tensine/ops/dispatch/elementwise_ops.h"
#include "tensine_bindings.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;


void bind_ops(py::module_& m) {
    m.def("add", [](const TsTensor& input1, const TsTensor& input2) {
        return ts_add(&input1, &input2);
    },
    py::arg("input1"),
    py::arg("input2")
    );

    m.def("mul", [](const TsTensor& input1, const TsTensor& input2) {
        return ts_mul(&input1, &input2);
    },
    py::arg("input1"),
    py::arg("input2")
    );
}
