# TimeSeries

A time series analysis library with CUDA-accelerated C++ backend and Python frontend.

## Prerequisites
- CMake 3.25+
- CUDA Toolkit 12.x
- Python 3.8+
- pybind11 (`pip install pybind11`)

## Build
```bash
./scripts/build.sh
```

## Test
```bash
./scripts/test.sh
```

## Usage
```python
from tsx import ARIMA
data = [1.0, 2.0, 3.0, 4.0, 5.0]
model = ARIMA(p=1, d=0, q=0)
params = model.fit(data)
print(params)
```