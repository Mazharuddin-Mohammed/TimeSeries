/**
 * @file matrix_ops.h
 * @brief Header file for matrix operations utility functions.
 *
 * This file provides utility functions for common matrix operations such as
 * matrix-vector multiplication, matrix-matrix multiplication, and solving
 * linear systems. These operations support the ARIMA model implementation.
 *
 * @author Mazharuddin Mohammed
 */

#ifndef TSX_MATRIX_OPS_H
#define TSX_MATRIX_OPS_H

#include <vector>
#include <cmath>

namespace tsx {

// Matrix-vector multiplication: y = A * x
void matrix_vector_multiply(const std::vector<double>& A, const std::vector<double>& x,
                           std::vector<double>& y, int rows, int cols);

// Matrix-matrix multiplication: C = A * B
void matrix_matrix_multiply(const std::vector<double>& A, const std::vector<double>& B,
                           std::vector<double>& C, int A_rows, int A_cols, int B_cols);

// Solve linear system Ax = b using Gaussian elimination with partial pivoting
bool solve_linear_system(std::vector<double>& A, std::vector<double>& b,
                        std::vector<double>& x, int n);

// Check if a matrix is singular (determinant close to zero)
bool is_singular(const std::vector<double>& A, int n, double tolerance = 1e-10);

} // namespace tsx

#endif // TSX_MATRIX_OPS_H