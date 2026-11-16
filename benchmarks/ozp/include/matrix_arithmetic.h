#pragma once

// The "parithmetic" functions used in purify
// daxpy type: dst += alpha * src
template <typename T>
void padd(T *src, T *dst, unsigned int size, T alpha = T(1));

// transpose the matrix
template <typename T> void transpose(T *src, T *dst, unsigned int size);

// compute the frobenius norm
template <typename T> T pnorm(T *matrix, unsigned int size);

// scale the matrix
template <typename T, typename U>
void pscale(T *matrix, unsigned int size, U scalar);

// multiply two matrices
template <typename T, typename U>
int pmultiply(T *a, T *b, U *c, unsigned int size);

// multiply two matrices - force accumulation in double
template <typename T> int pmultiply_acc(T *a, T *b, T *c, unsigned int size);

// compute the dot product of two matrices
template <typename T> T pdot(T *a, T *b, unsigned int size);

// compute the trace of a matrix
template <typename T> T ptrace(T *mat, unsigned int size);

// symmetrize the matrix (to fix drift)
template <typename T> void psymmetrize(T *mat, unsigned int size);

// helper to compare matrices in the driver
template <typename T> T compare_matrices(T *a, T *b, unsigned int size);

// helper to check commutator error norm(ab - ba)
template <typename T> T commutator_error(T *a, T *b, unsigned int size);

// approximate the largest and smallest eigenvalue
template <typename T>
void get_eig_circle(T *matrix, unsigned int size, T &h_min, T &h_max);

// Include the implementation of templates
#include "matrix_arithmetic_imp.h"
