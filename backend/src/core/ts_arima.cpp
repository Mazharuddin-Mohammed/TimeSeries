#include <tsx/core/ts_arima.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <cstring>

namespace tsx {
ARIMA::ARIMA(int p, int d, int q) : p_(p), d_(d), q_(q) {
    if (p < 0 || d < 0 || q < 0) {
        throw std::invalid_argument("ARIMA orders must be non-negative");
    }
}

std::vector<double> ARIMA::fit(const std::vector<double>& data) {
    if (data.size() <= p_ + d_) {
        throw std::runtime_error("Data too short for ARIMA model");
    }
    std::vector<double> diff_data = difference(data, d_);
    return estimate_ar_params(diff_data);
}

std::vector<double> ARIMA::difference(const std::vector<double>& data, int d) {
    std::vector<double> result = data;
    for (int i = 0; i < d; ++i) {
        std::vector<double> temp(result.size() - 1);
        for (size_t j = 0; j < temp.size(); ++j) {
            temp[j] = result[j + 1] - result[j];
        }
        result = temp;
    }
    return result;
}

std::vector<double> ARIMA::estimate_ar_params(const std::vector<double>& data) {
    int n = data.size() - p_;
    std::vector<double> X(n * p_);
    std::vector<double> y(n);

    for (int i = 0; i < n; ++i) {
        y[i] = data[i + p_];
        for (int j = 0; j < p_; ++j) {
            X[i * p_ + j] = data[i + p_ - j - 1];
        }
    }

    std::vector<double> params(p_);
    solve_least_squares_gpu(X.data(), y.data(), n, p_, params.data());
    return params;
}

void ARIMA::solve_least_squares_gpu(double* X, double* y, int n, int p, double* params) {
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS initialization failed");
    }

    double *d_X, *d_y, *d_XtX, *d_Xty;
    cudaMalloc(&d_X, n * p * sizeof(double));
    cudaMalloc(&d_y, n * sizeof(double));
    cudaMalloc(&d_XtX, p * p * sizeof(double));
    cudaMalloc(&d_Xty, p * sizeof(double));

    cudaMemcpy(d_X, X, n * p * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice);

    double alpha = 1.0, beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, p, n, &alpha, d_X, n, d_X, n, &beta, d_XtX, p);
    cublasDgemv(handle, CUBLAS_OP_T, n, p, &alpha, d_X, n, d_y, 1, &beta, d_Xty, 1);

    std::vector<double> XtX(p * p), Xty(p);
    cudaMemcpy(XtX.data(), d_XtX, p * p * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Xty.data(), d_Xty, p * sizeof(double), cudaMemcpyDeviceToHost);

    memcpy(params, Xty.data(), p * sizeof(double)); // Placeholder

    cudaFree(d_X); cudaFree(d_y); cudaFree(d_XtX); cudaFree(d_Xty);
    cublasDestroy(handle);
}

extern "C" {
    ARIMA* ARIMA_new(int p, int d, int q) { return new ARIMA(p, d, q); }
    void ARIMA_fit(ARIMA* model, double* data, int len, double* params) {
        std::vector<double> input(data, data + len);
        std::vector<double> result = model->fit(input);
        std::copy(result.begin(), result.end(), params);
    }
    void ARIMA_delete(ARIMA* model) { delete model; }
}
} // namespace tsx