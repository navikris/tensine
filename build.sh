#!/bin/bash
set -e

# Building the library 
echo "Building C library... "
mkdir -p build
cd build
cmake ..
cmake --build .

# Python 
cd ..

echo "Creating venv"
python3 -m venv python_env

echo "Installing dependencies"
python_env/bin/python -m pip install --upgrade pip
python_env/bin/python -m pip install -r requirements.txt

export PYTHONPATH="$(pwd)/python:$PYTHONPATH"

echo "Done :)"
