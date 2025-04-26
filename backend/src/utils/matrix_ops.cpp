/**
 * @file matrix_ops.cpp
 * @brief Implementation of matrix operations utility functions.
 *
 * This file implements various matrix operations including matrix-vector multiplication,
 * matrix-matrix multiplication, and solving linear systems using Gaussian elimination
 * with partial pivoting. These functions support the ARIMA model implementation.
 *
 * @author Mazharuddin Mohammed
 */

#include <utils/matrix_ops.h>
#include <stdexcept>
#include <algorithm>

namespace tsx {

void matrix_vector_multiply(const std::vector<double>& A, const std::vector<double>& x,
                           std::vector<double>& y, int rows, int cols) {
    if (x.size() != static_cast<size_t>(cols) || A.size() != static_cast<size_t>(rows * cols)) {
        throw std::invalid_argument("Matrix-vector dimensions mismatch");
    }

    y.resize(rows);
    for (int i = 0; i < rows; ++i) {
        y[i] = 0.0;
        for (int j = 0; j < cols; ++j) {
            y[i] += A[i * cols + j] * x[j];
        }
    }
}

void matrix_matrix_multiply(const std::vector<double>& A, const std::vector<double>& B,
                           std::vector<double>& C, int A_rows, int A_cols, int B_cols) {
    if (A.size() != static_cast<size_t>(A_rows * A_cols) ||
        B.size() != static_cast<size_t>(A_cols * B_cols)) {
        throw std::invalid_argument("Matrix-matrix dimensions mismatch");
    }

    C.resize(A_rows * B_cols);
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            C[i * B_cols + j] = 0.0;
            for (int k = 0; k < A_cols; ++k) {
                C[i * B_cols + j] += A[i * A_cols + k] * B[k * B_cols + j];
            }
        }
    }
}

bool solve_linear_system(std::vector<double>& A, std::vector<double>& b,
                        std::vector<double>& x, int n) {
    if (A.size() != static_cast<size_t>(n * n) || b.size() != static_cast<size_t>(n)) {
        throw std::invalid_argument("Linear system dimensions mismatch");
    }

    // Check if matrix is singular
    if (is_singular(A, n)) {
        return false;
    }

    x.resize(n);
    std::vector<int> pivot(n);

    // Gaussian elimination with partial pivoting
    for (int k = 0; k < n; ++k) {
        // Find pivot
        int max_idx = k;
        double max_val = std::abs(A[k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            double val = std::abs(A[i * n + k]);
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        pivot[k] = max_idx;

        // Swap rows if needed
        if (max_idx != k) {
            for (int j = 0; j < n; ++j) {
                std::swap(A[k * n + j], A[max_idx * n + j]);
            }
            std::swap(b[k], b[max_idx]);
        }

        // Check for singularity
        if (std::abs(A[k * n + k]) < 1e-10) {
            return false;
        }

        // Eliminate below
        for (int i = k + 1; i < n; ++i) {
            double factor = A[i * n + k] / A[k * n + k];
            A[i * n + k] = 0.0;  // Just to be explicit
            for (int j = k + 1; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution
    for (int i = n - 1; i >= 0; --i) {
        x[i] = b[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= A[i * n + j] * x[j];
        }
        x[i] /= A[i * n + i];
    }

    return true;
}

bool is_singular(const std::vector<double>& A, int n, double tolerance) {
    // Simple check: if any diagonal element is close to zero, the matrix is likely singular
    for (int i = 0; i < n; ++i) {
        if (std::abs(A[i * n + i]) < tolerance) {
            return true;
        }
    }

    // For a more accurate check, we would compute the determinant or use LU decomposition
    // But this simple check is often sufficient for our purposes
    return false;
}

} // namespace tsx