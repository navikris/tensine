#include "tensine/core/dtype.h"
#include "tensine/core/tensor.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;


PYBIND11_MODULE(tensine, m) {
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
        py::arg("dtype"),
        py::arg("requires_grad") = false
        )
        .def(py::init([](
            py::buffer buf,
            TsDType dtype,
            bool requires_grad
        ) {
            py::buffer_info info = buf.request();
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
        py::arg("dtype"),
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
        py::arg("dtype"),
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
