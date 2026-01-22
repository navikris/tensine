#include "tensine/core/dtype.h"
#include "tensine/core/tensor.h"
#include "tensine_bindings.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


namespace py = pybind11;


TsDType _get_buf_type(const py::array& buf) {
    auto dt = buf.dtype();

    if (dt.kind() == 'f' && dt.itemsize() == 4)
        return TsDType::TS_DTYPE_FLOAT32;

    if (dt.kind() == 'f' && dt.itemsize() == 8)
        return TsDType::TS_DTYPE_FLOAT64;

    if (dt.kind() == 'i' && dt.itemsize() == 4)
        return TsDType::TS_DTYPE_INT32;

    if (dt.kind() == 'i' && dt.itemsize() == 8)
        return TsDType::TS_DTYPE_INT64;

    throw std::runtime_error("Unsupported numpy dtype");
}


/* TODO:
    1. Add shape as property and return as array
    2. Check for contiguity for from buffer method
    3. Add support for initalizing from list as well
*/
void bind_tensor(py::module_& m) {
    py::class_<TsTensorStorage>(m, "TensorStorage")
        .def_readwrite("data", &TsTensorStorage::data)  
        .def_readonly("owns_data", &TsTensorStorage::owns_data)  
        .def_readonly("nbytes", &TsTensorStorage::nbytes)  
        .def_readonly("ref_count", &TsTensorStorage::ref_count);

    py::class_<TsTensor>(m, "Tensor")
        .def(py::init([](
            std::vector<size_t> shape,
            TsDType dtype,
            bool requires_grad
        ) {
            return ts_tensor_create(
                dtype,
                shape.size(),
                shape.data(),
                requires_grad,
                nullptr
            );
        }),
        py::arg("shape"),
        py::arg("dtype") = TsDType::TS_DTYPE_FLOAT32,
        py::arg("requires_grad") = false
        )
        .def(py::init([](
            py::array array,
            bool requires_grad
        ) {
            py::buffer_info info = array.request();
            TsDType dtype = _get_buf_type(array);
            return ts_tensor_from_buffer(
                info.ptr,
                dtype,
                info.ndim,
                reinterpret_cast<size_t*>(info.shape.data()),
                requires_grad,
                nullptr
            );
        }),
        py::arg("buf"),
        py::arg("requires_grad") = false
        )
        .def(py::init([](
            TsTensorStorage& storage,
            std::vector<size_t> shape,
            TsDType dtype,
            bool requires_grad
        ) {
            return ts_tensor_from_storage(
                &storage,
                dtype,
                shape.size(),
                shape.data(),
                requires_grad,
                nullptr
            );
        }),
        py::arg("storage"),
        py::arg("shape"),
        py::arg("dtype") = TsDType::TS_DTYPE_FLOAT32,
        py::arg("requires_grad") = false
        )
        .def("__del__", [](TsTensor* t) {
            if (t) ts_tensor_free(t);
        })

        .def_readonly("dtype", &TsTensor::dtype)
        .def_readonly("shape", &TsTensor::shape)
        .def_readonly("requires_grad", &TsTensor::requires_grad)

        // empty_like factory
        .def_static("empty_like",
            [](const TsTensor& src) {
                return ts_tensor_empty_like(&src);
            }
        )
        
        .def("print",[](const TsTensor& t) {
            ts_tensor_print(&t);
        });
}
