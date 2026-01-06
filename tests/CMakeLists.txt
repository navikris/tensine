# tensine/tests/CMakeLists.txt
cmake_minimum_required(VERSION 3.31.6)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

file(GLOB_RECURSE test_sources "*.c")

foreach(test_src ${test_sources})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_compile_options(${test_name} PRIVATE -Wall -Wextra)
    target_link_libraries(${test_name} PRIVATE tensine_core)
endforeach()
