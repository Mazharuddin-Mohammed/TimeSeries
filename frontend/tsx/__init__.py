"""
@file __init__.py
@brief Python frontend for the TimeSeries library.

This file provides the Python interface for the TimeSeries library,
which includes the ARIMA model implementation. It handles importing
the C++ extension module and provides a Pythonic wrapper around it.

@author Mazharuddin Mohammed
"""

import os
import sys
import importlib.util

# Try to import the C++ extension module
try:
    # First try normal import
    import tsx_backend
except ImportError:
    # If that fails, try to load it directly from the current directory
    try:
        # Find the .so file in the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        extension_files = [f for f in os.listdir(current_dir) if f.startswith('tsx_backend') and f.endswith('.so')]

        if extension_files:
            extension_path = os.path.join(current_dir, extension_files[0])
            spec = importlib.util.spec_from_file_location('tsx_backend', extension_path)
            tsx_backend = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tsx_backend)
        else:
            raise ImportError(f"Could not find tsx_backend extension in {current_dir}")
    except Exception as e:
        raise ImportError(f"Failed to import tsx_backend: {e}")

class ARIMA:
    def __init__(self, p=1, d=0, q=0):
        self.model = tsx_backend.ARIMA(p, d, q)

    def fit(self, data):
        return self.model.fit(data)

if __name__ == "__main__":
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    model = ARIMA(p=2, d=1, q=0)
    params = model.fit(data)
    print(f"Estimated ARIMA parameters: {params}")