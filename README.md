# TimeSeries

A high-performance time series analysis library with CUDA-accelerated C++ backend and Python frontend. This library provides efficient implementations of time series forecasting models, starting with ARIMA (AutoRegressive Integrated Moving Average).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## Features

- **ARIMA Model Implementation**: Supports p (autoregressive), d (differencing), and q (moving average) parameters
- **GPU Acceleration**: Utilizes CUDA for high-performance matrix operations when available
- **CPU Fallback**: Automatically falls back to CPU implementation when GPU is not available
- **Python Interface**: Easy-to-use Python frontend with a simple API
- **Extensible Architecture**: Designed to be extended with additional time series models

## Prerequisites

- CMake 3.22+
- CUDA Toolkit 12.x (optional, falls back to CPU if not available)
- Python 3.8+
- pybind11 (`pip install pybind11`)

## Installation

### From Source

1. Clone the repository:
   ```bash
   git clone https://github.com/Mazharuddin-Mohammed/TimeSeries.git
   cd TimeSeries
   ```

2. Build the library:
   ```bash
   ./scripts/build.sh
   ```

3. The build script will automatically install the Python package in development mode.

## Testing

Run the tests to verify the installation:

```bash
./scripts/test.sh
```

This will run both C++ and Python tests.

## Usage

### Basic ARIMA Model

```python
from tsx import ARIMA

# Sample time series data
data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

# Create an ARIMA model with p=1, d=0, q=0 (AR(1) model)
model = ARIMA(p=1, d=0, q=0)

# Fit the model to estimate parameters
params = model.fit(data)

# Print the estimated parameters
print(f"Estimated ARIMA parameters: {params}")
```

### Advanced Usage

For more complex time series with seasonality or trends, you can adjust the differencing parameter:

```python
# Create an ARIMA model with p=2, d=1, q=0
# This applies first-order differencing to handle trends
model = ARIMA(p=2, d=1, q=0)
params = model.fit(data)
```

## Project Structure

- `backend/`: C++ implementation of time series models
  - `include/`: Header files
  - `src/`: Source files
  - `tests/`: C++ tests
- `bindings/`: Python bindings using pybind11
- `frontend/`: Python package
  - `tsx/`: Python module
  - `tests/`: Python tests
- `scripts/`: Build and test scripts

## Performance

The library leverages GPU acceleration when available, providing significant performance improvements for large datasets. For smaller datasets or when GPU is not available, it automatically falls back to an efficient CPU implementation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

<a href="https://www.buymeacoffee.com/mazharm" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

Hi, I'm a passionate scientist dedicated to bridging the gap between academia and industry through research exploitation and knowledge transfer. My work focuses on turning cutting-edge academic discoveries into practical, real-world solutions that drive innovation and impact. By fostering collaboration and translating complex research into accessible applications, I aim to create a seamless flow of knowledge that benefits both worlds.

If you'd like to support my mission to connect ideas with impact, consider buying me a coffee! Your support helps fuel my efforts to build stronger bridges between researchers and industry innovators. Thank you for being part of this journey!