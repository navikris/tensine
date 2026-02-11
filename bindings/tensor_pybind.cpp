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


py::dtype ts_dtype_to_numpy(TsDType dtype) {
    switch (dtype) {
        case TS_DTYPE_FLOAT32: return py::dtype::of<float>();
        case TS_DTYPE_FLOAT64: return py::dtype::of<double>();
        case TS_DTYPE_INT32:   return py::dtype::of<int32_t>();
        case TS_DTYPE_INT64:   return py::dtype::of<int64_t>();
        default:
            throw std::runtime_error("Unsupported dtype");
    }
}


py::array copy_tensor_to_numpy(const TsTensor* t) {
    py::dtype dtype = ts_dtype_to_numpy(t->dtype);

    std::vector<ssize_t> shape(t->ndim);
    for (size_t i = 0; i < t->ndim; i++) {
        shape[i] = t->shape[i];
    }

    py::array arr(dtype, shape);
    std::memcpy(arr.mutable_data(), t->storage->data, t->numel * dtype.itemsize());

    return arr;
}


std::vector<ssize_t> get_tensor_shape(const TsTensor* t) {
    std::vector<ssize_t> shape(t->ndim);
    for (size_t i = 0; i < t->ndim; i++) {
        shape[i] = t->shape[i];
    }
    return shape;
}


/* TODO:
    1. Check for contiguity for from buffer method
    2. Add support for initalizing from list as well
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
            
            // HACK: Need to add support for cloning data from buffer
            TsTensor* copy_tensor = ts_tensor_from_buffer(
                info.ptr,
                dtype,
                info.ndim,
                reinterpret_cast<size_t*>(info.shape.data()),
                requires_grad,
                nullptr
            );
            TsTensor* new_tensor = ts_tensor_clone(copy_tensor);
            ts_tensor_free(copy_tensor);
            return new_tensor;
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
        .def_readonly("requires_grad", &TsTensor::requires_grad)

        // empty_like factory
        .def_static("empty_like",
            [](const TsTensor& src) {
                return ts_tensor_empty_like(&src);
            }
        )

        .def("print",[](const TsTensor& t) {
            ts_tensor_print(&t);
        })

        .def("shape",[](const TsTensor& t) {
            return get_tensor_shape(&t);
        })

        .def("to_numpy",[](const TsTensor& t) {
            return copy_tensor_to_numpy(&t);
        })

        .def("reshape",[](const TsTensor& t, std::vector<size_t> shape) {
            return ts_tensor_reshape(&t, shape.data(), shape.size());
        })

        .def("permute",[](const TsTensor& t, std::vector<size_t> order) {
            return ts_tensor_permute(&t, order.data());
        })

        .def("transpose",[](const TsTensor& t, size_t dim_1, size_t dim_2) {
            return ts_tensor_transpose(&t, dim_1, dim_2);
        })

        .def("slice",[](const TsTensor& t, std::vector<size_t> start_idxs, std::vector<size_t> end_idxs) {
            return ts_tensor_slice(&t, start_idxs.data(), end_idxs.data());
        });
}
