#ifndef TS_ARIMA_H
#define TS_ARIMA_H

#include <vector>

namespace tsx {
class ARIMA {
public:
    ARIMA(int p, int d, int q);
    std::vector<double> fit(const std::vector<double>& data);
    int get_p() const { return p_; }  // Added getter for p_

private:
    int p_, d_, q_; // AR, differencing, MA orders
    std::vector<double> difference(const std::vector<double>& data, int d);
    std::vector<double> estimate_ar_params(const std::vector<double>& data);
    void solve_least_squares_gpu(double* X, double* y, int n, int p, double* params);
};

// C-style interface for Python bindings
extern "C" {
    ARIMA* ARIMA_new(int p, int d, int q);
    void ARIMA_fit(ARIMA* model, double* data, int len, double* params);
    void ARIMA_delete(ARIMA* model);
}
} // namespace tsx

#endif // TS_ARIMA_H