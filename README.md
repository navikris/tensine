# Tensine

Tensine is a lightweight, low-level tensor library written in **C**. The project focuses on understanding and implementing core tensor concepts—**strides, views, dtype dispatch, and kernels**—in a clean, minimal, and hackable way.

**Python bindings via pybind11**

---

## Project Structure

```
tensine/
├── bindings/
├── include/
│   └── tensine/
|       ├── core/
|       └── ops/
├── src/
|   ├── core/
|   └── ops/
|       └── dispatch/
|       └── kernels/
├── tests/
|   ├── core/
|   └── kernels/
└── README.md
```

---

## Building the Project

### Requirements

* CMake ≥ 3.31.6
* C compiler (GCC)
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
chmod +x build.sh

./build.sh

export PYTHONPATH="$(pwd)/python:$PYTHONPATH"
```
---

## License

MIT License

## Third-Party Dependencies

This project uses the following third-party libraries:

- **pybind11**  
  Licensed under the BSD 3-Clause License  
  https://github.com/pybind/pybind11
