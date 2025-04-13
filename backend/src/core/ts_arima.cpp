#include <core/ts_arima.h>
#include <utils/matrix_ops.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>

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
    // Initialize resources to nullptr/0 for safe cleanup
    cublasHandle_t handle = nullptr;
    double *d_X = nullptr, *d_y = nullptr, *d_XtX = nullptr, *d_Xty = nullptr;

    try {
        // Try to use GPU if available
        cudaError_t cuda_status = cudaFree(0);  // Simple test to check if CUDA is available
        bool use_gpu = (cuda_status == cudaSuccess);

        if (use_gpu) {
            // Initialize cuBLAS
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cuBLAS initialization failed");
            }

            // Allocate device memory
            if (cudaMalloc(&d_X, n * p * sizeof(double)) != cudaSuccess ||
                cudaMalloc(&d_y, n * sizeof(double)) != cudaSuccess ||
                cudaMalloc(&d_XtX, p * p * sizeof(double)) != cudaSuccess ||
                cudaMalloc(&d_Xty, p * sizeof(double)) != cudaSuccess) {
                throw std::runtime_error("CUDA memory allocation failed");
            }

            // Copy data to device
            if (cudaMemcpy(d_X, X, n * p * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
                throw std::runtime_error("CUDA memory copy to device failed");
            }

            // Compute X^T * X and X^T * y
            double alpha = 1.0, beta = 0.0;
            if (cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, p, n, &alpha, d_X, n, d_X, n, &beta, d_XtX, p) != CUBLAS_STATUS_SUCCESS ||
                cublasDgemv(handle, CUBLAS_OP_T, n, p, &alpha, d_X, n, d_y, 1, &beta, d_Xty, 1) != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("cuBLAS matrix operations failed");
            }

            // Copy results back to host
            std::vector<double> XtX(p * p), Xty(p);
            if (cudaMemcpy(XtX.data(), d_XtX, p * p * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(Xty.data(), d_Xty, p * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
                throw std::runtime_error("CUDA memory copy to host failed");
            }

            // Solve the linear system using our utility function
            std::vector<double> params_vec(p);
            if (!solve_linear_system(XtX, Xty, params_vec, p)) {
                throw std::runtime_error("Failed to solve linear system - matrix may be singular");
            }

            // Copy results to output array
            std::copy(params_vec.begin(), params_vec.end(), params);
        } else {
            // CPU fallback implementation
            std::vector<double> X_vec(X, X + n * p);
            std::vector<double> y_vec(y, y + n);

            // Compute X^T * X
            std::vector<double> X_transpose(n * p);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < p; ++j) {
                    X_transpose[j * n + i] = X_vec[i * p + j];
                }
            }

            std::vector<double> XtX(p * p);
            matrix_matrix_multiply(X_transpose, X_vec, XtX, p, n, p);

            // Compute X^T * y
            std::vector<double> Xty(p);
            matrix_vector_multiply(X_transpose, y_vec, Xty, p, n);

            // Solve the linear system
            std::vector<double> params_vec(p);
            if (!solve_linear_system(XtX, Xty, params_vec, p)) {
                throw std::runtime_error("Failed to solve linear system - matrix may be singular");
            }

            // Copy results to output array
            std::copy(params_vec.begin(), params_vec.end(), params);
        }

    } catch (const std::exception& e) {
        // Clean up resources before re-throwing
        if (d_X) cudaFree(d_X);
        if (d_y) cudaFree(d_y);
        if (d_XtX) cudaFree(d_XtX);
        if (d_Xty) cudaFree(d_Xty);
        if (handle) cublasDestroy(handle);
        throw; // Re-throw the exception
    }

    // Clean up resources
    if (d_X) cudaFree(d_X);
    if (d_y) cudaFree(d_y);
    if (d_XtX) cudaFree(d_XtX);
    if (d_Xty) cudaFree(d_Xty);
    if (handle) cublasDestroy(handle);
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