# Tensine

Tensine is a lightweight, low-level tensor library written in **C**. The project focuses on understanding and implementing core tensor concepts—**strides, views, dtype dispatch, and kernels**—in a clean, minimal, and hackable way.

**Python bindings via pybind11**

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
* Python ≥ 3.8 (for bindings)
* pybind11 (included as submodule)


### Clone with Submodules

```
git clone --recurse-submodules <repo-url>
```

If already cloned:

```
git submodule update --init --recursive
```

### Build (C Library + Tests)

```
mkdir build
cd build
cmake ..
cmake --build .
```

---

## License

MIT License

## Third-Party Dependencies

This project uses the following third-party libraries:

- **pybind11**  
  Licensed under the BSD 3-Clause License  
  https://github.com/pybind/pybind11
