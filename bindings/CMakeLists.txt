# tensine/bindings/CMakeLists.txt
cmake_minimum_required(VERSION 3.31.6)

file(GLOB_RECURSE sources "*.cpp")

add_library(tensine_py MODULE)

target_sources(tensine_py PRIVATE ${sources})
target_link_libraries(tensine_py
    PRIVATE tensine_core pybind11::module
)

set_target_properties(tensine_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python
    PREFIX ""
    OUTPUT_NAME "tensine"
)
