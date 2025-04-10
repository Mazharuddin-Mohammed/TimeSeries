#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tsx/core/ts_arima.h>

namespace py = pybind11;
using namespace tsx;

PYBIND11_MODULE(tsx_backend, m) {
    py::class_<ARIMA>(m, "ARIMA")
        .def(py::init<int, int, int>())
        .def("fit", [](ARIMA& self, std::vector<double> data) {
            std::vector<double> params(self.p_);
            ARIMA_fit(&self, data.data(), data.size(), params.data());
            return params;
        });
}