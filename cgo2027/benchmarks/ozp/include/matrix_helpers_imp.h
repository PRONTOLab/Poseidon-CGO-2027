#include <cmath>
#include <fstream>
#include <sstream>

template <typename T> T *allocate_matrix(unsigned int size) {
  T *matrix = new T[size * size];
  return matrix;
}

template <typename T> void fill_random_matrix(T *matrix, unsigned int size) {
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      matrix[i * size + j] = static_cast<T>(rand()) / RAND_MAX;
    }
  }
}

template <typename T> void fill_zeros(T *matrix, unsigned int size) {
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      matrix[i * size + j] = static_cast<T>(0);
    }
  }
}

template <typename T>
double calculate_sparsity(const T *matrix, unsigned int size, T threshold) {
  size_t sparse_sets_count = 0;
  size_t total_sets = 0;

  for (size_t j = 0; j < size; ++j) {
    for (size_t i = 0; i < size; i += 4) {
      size_t zero_count = 0;

      for (size_t k = 0; k < 4; ++k) {
        if (i + k < size) { // Ensure we don't go out of bounds
          if (std::abs(matrix[(i + k) * size + j]) < threshold) {
            zero_count++;
          }
        } else {
          // Implicit zero padding if size is not divisible by 4
          zero_count++;
        }
      }

      // Increment if there are at least two zeros
      if (zero_count >= 2) {
        sparse_sets_count++;
      }
      total_sets++;
    }
  }

  // Calculate sparsity as the percentage of sparse sets
  return 100 - 100 * static_cast<double>(sparse_sets_count) / total_sets;
}

template <typename T> void fill_identity(T *matrix, unsigned int size) {
  fill_zeros(matrix, size);
  for (size_t i = 0; i < size; ++i) {
    matrix[i * size + i] = static_cast<T>(1);
  }
}

template <typename T> void free_matrix(T *matrix) { delete matrix; }

template <typename T> void print_matrix(T *matrix, unsigned int size) {
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      printf("%.16f ", static_cast<double>(matrix[i * size + j]));
    }
    printf("\n");
  }
}

template <typename T>
void write_mm(const std::string &filename, T *matrix, unsigned int size) {
  std::ofstream file(filename);
  file << "%%MatrixMarket matrix coordinate real general\n";
  file << size << " " << size << " " << size * size << "\n";
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      file << i + 1 << " " << j + 1 << " " << matrix[i * size + j] << "\n";
    }
  }
}

template <typename T> T *max_in_rows(T *matrix, unsigned int size) {
  T *max_values = new T[size];
  for (size_t i = 0; i < size; ++i) {
    T max_val = std::abs(matrix[i * size + 0]);
    for (size_t j = 1; j < size; ++j) {
      T abs_val = std::abs(matrix[i * size + j]);
      if (abs_val > max_val)
        max_val = abs_val;
    }
    max_values[i] = max_val;
  }
  return max_values;
}

template <typename T> T *max_in_cols(T *matrix, unsigned int size) {
  T *max_values = new T[size]();
  for (size_t j = 0; j < size; ++j) {
    for (size_t i = 0; i < size; ++i) {
      T abs_val = std::abs(matrix[i * size + j]);
      if (abs_val > max_values[j])
        max_values[j] = abs_val;
    }
  }
  return max_values;
}

template <typename T, typename U>
void pcopy(T *src, U *dst, unsigned int size) {
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      dst[i * size + j] = static_cast<T>(src[i * size + j]);
    }
  }
}

template <typename T>
void scale_rows_in(T *matrix, T *scal, unsigned int size) {
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      matrix[i * size + j] = std::pow(2.0, -scal[i]) * matrix[i * size + j];
    }
  }
}

template <typename T>
void scale_cols_in(T *matrix, T *scal, unsigned int size) {
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      matrix[i * size + j] = pow(2.0, -scal[j]) * matrix[i * size + j];
    }
  }
}

template <typename T>
void scale_out(T *matrix, T *scal_a, T *scal_b, unsigned int size) {
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      matrix[i * size + j] =
          pow(2.0, scal_a[i] + scal_b[j]) * matrix[i * size + j];
    }
  }
}

template <typename T>
T *read_mm(const std::string &filename, unsigned int &size) {
  std::ifstream file(filename);
  std::string line;
  unsigned int rows, cols;
  size_t nonzeros;
  bool is_symmetric = false;

  // Read the header and check for symmetry
  while (std::getline(file, line)) {
    if (line[0] == '%' && line.find("symmetric") != std::string::npos) {
      is_symmetric = true;
    } else if (line[0] != '%') {
      break;
    }
  }

  std::istringstream iss(line);
  iss >> rows >> cols >> nonzeros;
  size = rows;

  T *matrix = allocate_matrix<T>(size);
  fill_zeros(matrix, size);

  int row, col;
  T value;
  while (nonzeros--) {
    file >> row >> col >> value;
    row--;
    col--;
    matrix[row * size + col] = value;
    if (is_symmetric && row != col) {
      matrix[col * size + row] = value;
    }
  }

  return matrix;
}
