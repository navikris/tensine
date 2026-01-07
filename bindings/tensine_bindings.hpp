#ifndef TENSINE_PYTHON_BINDINGS_H
#define TENSINE_PYTHON_BINDINGS_H


#include <pybind11/pybind11.h>


namespace py = pybind11;


void bind_dtype(py::module_& m);
void bind_tensor(py::module_& m);
void bind_ops(py::module_& m);


#endif /* TENSINE_PYTHON_BINDINGS_H */
