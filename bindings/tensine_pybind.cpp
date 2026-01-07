#include "tensine_bindings.hpp"

#include<pybind11/pybind11.h>


namespace py = pybind11;


PYBIND11_MODULE(tensine, m) {
    m.doc() = "Tensine tensor library";

    bind_dtype(m);
    bind_tensor(m);
    bind_ops(m);
}
