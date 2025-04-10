#include <tsx/core/ts_arima.h>
#include <cassert>
#include <vector>

int main() {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    tsx::ARIMA model(1, 0, 0);
    std::vector<double> params = model.fit(data);
    assert(params.size() == 1); // Basic check
    return 0;
}