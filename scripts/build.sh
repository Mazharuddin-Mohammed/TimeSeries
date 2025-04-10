#!/bin/bash

set -e

# Create build directory
mkdir -p build
cd build

# Configure CMake
cmake ..

# Build
make -j$(nproc)

# Copy bindings to frontend
cp bindings/tsx_backend*.so ../frontend/tsx/

# Install Python package
cd ../frontend
pip install -e .
cd ..