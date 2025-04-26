/**
 * @file test_arima.cpp
 * @brief Test file for the ARIMA model implementation.
 *
 * This file contains basic tests for the ARIMA model to verify
 * that the implementation works correctly. It tests model creation
 * and parameter estimation functionality.
 *
 * @author Mazharuddin Mohammed
 */

#include <core/ts_arima.h>
#include <cassert>
#include <vector>

int main() {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    tsx::ARIMA model(1, 0, 0);
    std::vector<double> params = model.fit(data);
    assert(params.size() == 1); // Basic check
    return 0;
}