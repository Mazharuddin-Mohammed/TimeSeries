/**
 * @file tsx_bind.cpp
 * @brief Python bindings for the C++ ARIMA implementation.
 *
 * This file creates Python bindings for the C++ ARIMA implementation
 * using pybind11. It exposes the ARIMA class and its methods to Python,
 * allowing seamless integration between the C++ backend and Python frontend.
 *
 * @author Mazharuddin Mohammed
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <core/ts_arima.h>

namespace py = pybind11;
using namespace tsx;

PYBIND11_MODULE(tsx_backend, m) {
    py::class_<ARIMA>(m, "ARIMA")
        .def(py::init<int, int, int>())
        .def("fit", [](ARIMA& self, std::vector<double> data) {
            std::vector<double> params(self.get_p());  // Use getter
            ARIMA_fit(&self, data.data(), data.size(), params.data());
            return params;
        });
}