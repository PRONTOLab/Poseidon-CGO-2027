#pragma once

#include <cstdlib>
#include <ctime>
#include <limits>
#include <string>

// allocate a matrix size x size
template <typename T> T *allocate_matrix(unsigned int size);

// destruct the matrix
template <typename T> void free_matrix(T *matrix);

// print the matrix to the standard out
template <typename T> void print_matrix(T *matrix, unsigned int size);

// fill a matrix with random entries between 0 - 1
template <typename T> void fill_random_matrix(T *matrix, unsigned int size);

// fill the matrix with all zeros
template <typename T> void fill_zeros(T *matrix, unsigned int size);

// fill the matrix so it is the identity matrix
template <typename T> void fill_identity(T *matrix, unsigned int size);

// find the largest value in each row
template <typename T> T *max_in_rows(T *matrix, unsigned int size);

// find the largest value in each column
template <typename T> T *max_in_cols(T *matrix, unsigned int size);

// copy one matrix to another
template <typename T, typename U> void pcopy(T *src, U *dst, unsigned int size);

// read matrix market coordinate files
template <typename T> T **read_mm(const std::string &filename);

// write matrix market coordinate file
template <typename T>
void write_mm(const std::string &filename, T *matrix, unsigned int size);

// calculate the sparsity of a matrix
template <typename T>
double calculate_sparsity(const T *matrix, unsigned int size,
                          T threshold = std::numeric_limits<T>::min());

// these are helpers for the Ozaki scheme scaling.
template <typename T> void scale_rows_in(T *matrix, T *scal, unsigned int size);
template <typename T> void scale_cols_in(T *matrix, T *scal, unsigned int size);
template <typename T>
void scale_out(T *matrix, T *scal_a, T *scal_b, unsigned int size);

// Include the implementation of templates
#include "matrix_helpers_imp.h"
