# tensine/src/CMakeLists.txt
cmake_minimum_required(VERSION 3.31.6)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

function(add_recursive_sources target dir)
    # Recursively find all .c and .h files in `dir` and its subfolders
    file(GLOB_RECURSE sources CONFIGURE_DEPENDS
        "${dir}/*.c"
        "${dir}/*.h"
    )
    # Add these sources to the target
    target_sources(${target} PRIVATE ${sources})
endfunction()

add_library(tensine_core SHARED)
target_compile_options(tensine_core PRIVATE -Wall -Wextra)

add_recursive_sources(tensine_core "${CMAKE_CURRENT_SOURCE_DIR}")
add_recursive_sources(tensine_core "${PROJECT_SOURCE_DIR}/include")

target_include_directories(tensine_core PUBLIC
    # For build-time
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    # For install-time
    $<INSTALL_INTERFACE:include>
)
