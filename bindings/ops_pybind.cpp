#include "tensine/core/tensor.h"
#include "tensine/ops/dispatch/elementwise_ops.h"
#include "tensine/ops/dispatch/matmul_ops.h"
#include "tensine/ops/dispatch/pooling_ops.h"
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

    m.def("matmul", [](const TsTensor& input1, const TsTensor& input2) {
        return ts_matmul(&input1, &input2);
    },
    py::arg("input1"),
    py::arg("input2")
    );

    m.def("maxpool2d", [](const TsTensor& input, std::vector<size_t> kernel_size, std::vector<size_t> stride, std::vector<size_t> padding) {
        return ts_maxpool2d(&input, kernel_size.data(), stride.data(), padding.data());
    },
    py::arg("input"),
    py::arg("kernel_size"),
    py::arg("stride") = std::vector<size_t> {1, 1},
    py::arg("padding") = std::vector<size_t> {0, 0}
    );
}
