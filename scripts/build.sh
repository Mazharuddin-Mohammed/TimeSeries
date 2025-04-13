#!/bin/bash

set -e

# Create build directory
mkdir -p build
cd build

# Set pybind11 path
PYBIND11_CMAKE_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")

# Configure CMake with pybind11 path
cmake -DCMAKE_PREFIX_PATH="$PYBIND11_CMAKE_DIR" ..

# Build
make -j$(nproc)

# Copy bindings with platform-specific name to frontend/tsx
cp bindings/tsx_backend*.so ../frontend/tsx/

# Install Python package from frontend/tsx
cd ../frontend/tsx
pip install -e .
cd ../..