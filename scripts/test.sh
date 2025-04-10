#!/bin/bash

set -e

# Run C++ tests
./build/backend/test_arima

# Run Python tests
cd frontend
pytest tests/
cd ..