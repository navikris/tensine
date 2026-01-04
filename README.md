# Tensine

A lightweight, low-level tensor library written in **C**. The project focuses on understanding and implementing core tensor concepts—**strides, views, dtype dispatch, and kernels**—in a clean, minimal, and hackable way.

---

## Project Structure

```
tensine/
├── include/
│   └── tensine/
|       ├── core/
|       └── ops/
├── src/
|   ├── core/
|   ├── ops/
|   |   └── dispatch/
|   |   └── kernels/
│   └── CMakeLists.txt
├── tests/
|   ├── core/
|   ├── kernels/
│   └── CMakeLists.txt
├── CMakeLists.txt
└── README.md
```

---

## Building the Project

### Requirements

* CMake ≥ 3.31.6
* C compiler (GCC / Clang)


### Build (C Library + Tests)

```
mkdir build
cd build
cmake ..
cmake --build .
```
