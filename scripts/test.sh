#!/bin/bash

set -e

# Run C++ tests
./build/backend/test_arima

# Run Python tests from the root, ensuring frontend/tsx is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/frontend/tsx
pytest frontend/tests/