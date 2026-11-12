// matrix_arithmetic_impl.h
#include "matrix_helpers.h"
#include <cmath>

template <typename T> void padd(T *src, T *dst, unsigned int size, T alpha) {
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      dst[i * size + j] += alpha * src[i * size + j];
    }
  }
}

template <typename T> void ptranspose(T *src, T *dst, unsigned int size) {
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      dst[i * size + j] = src[j * size + i];
    }
  }
}

template <typename T> T pnorm(T *matrix, unsigned int size) {
  T sum = 0;
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      sum += matrix[i * size + j] * matrix[i * size + j];
    }
  }
  return std::sqrt(sum);
}

template <typename T, typename U>
int pmultiply(T *a, T *b, U *c, unsigned int size) {
  fill_zeros(c, size);
#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      for (size_t k = 0; k < size; ++k) {
        c[i * size + j] += a[i * size + k] * b[k * size + j];
      }
    }
  }
  return 1;
}

template <typename T> int pmultiply_acc(T *a, T *b, T *c, unsigned int size) {
  auto sa = allocate_matrix<float>(size);
  auto sb = allocate_matrix<float>(size);
  pcopy(a, sa, size);
  pcopy(b, sb, size);
  auto retval = pmultiply(sa, sb, c, size);
  free_matrix(sa);
  free_matrix(sb);
  return retval;
}

template <typename T> T pdot(T *a, T *b, unsigned int size) {
  T sum = 0;
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      sum += a[i * size + j] * b[i * size + j];
    }
  }
  return sum;
}

template <typename T> T ptrace(T *mat, unsigned int size) {
  T trace = 0;
  for (size_t i = 0; i < size; ++i) {
    trace += mat[i * size + i];
  }
  return trace;
}

template <typename T, typename U>
void pscale(T *matrix, unsigned int size, U scalar) {
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      matrix[i * size + j] *= static_cast<T>(scalar);
    }
  }
}

template <typename T> T compare_matrices(T *a, T *b, unsigned int size) {
  auto diff = allocate_matrix<T>(size);
  pcopy(a, diff, size);
  padd(b, diff, size, static_cast<T>(-1));
  auto norm = pnorm(diff, size);
  free_matrix(diff);
  return norm;
}

template <typename T> T commutator_error(T *a, T *b, unsigned int size) {
  auto ab = allocate_matrix<T>(size);
  auto ba = allocate_matrix<T>(size);

  pmultiply(a, b, ab, size);
  pmultiply(b, a, ba, size);
  padd(ab, ba, size, static_cast<T>(-1));

  auto norm = pnorm(ba, size);

  free_matrix(ab);
  free_matrix(ba);
  return norm;
}

template <typename T>
void get_eig_circle(T *matrix, unsigned int size, T &h_min, T &h_max) {
  h_max = std::numeric_limits<T>::lowest();
  h_min = std::numeric_limits<T>::max();

  for (size_t i = 0; i < size; ++i) {
    T row_sum = 0;
    for (size_t j = 0; j < size; ++j) {
      if (i != j) {
        row_sum += std::abs(matrix[i * size + j]);
      }
    }

    h_max = std::max(h_max, matrix[i * size + i] + row_sum);
    h_min = std::min(h_min, matrix[i * size + i] - row_sum);
  }
}

template <typename T> void psymmetrize(T *mat, unsigned int size) {
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = i + 1; j < size; ++j) {
      T avg = (mat[i * size + j] + mat[j * size + i]) / 2;
      mat[i * size + j] = avg;
      mat[j * size + i] = avg;
    }
  }
}
