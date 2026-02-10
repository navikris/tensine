#!/bin/bash
set -e

# Setting up python env 
echo "Creating venv"
python3 -m venv python_env

echo "Installing dependencies"
python_env/bin/python -m pip install --upgrade pip
python_env/bin/python -m pip install -r requirements.txt


# Building the library
echo "Building C library... "
cmake --build build

echo "Done :)"
