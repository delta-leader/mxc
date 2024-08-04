#pragma once

#include <complex>
#include <cmath>

class MatrixAccessor {
public:
  virtual std::complex<double> operator()(double d) const = 0;
};

class Laplace3D : public MatrixAccessor {
public:
  double singularity;
  Laplace3D (double s) : singularity(1. / s) {}
  std::complex<double> operator()(double d) const override {
    if (d == 0.)
      return std::complex<double>(singularity, 0.);
    else
      return std::complex<double>(1. / d, 0.);
  }
};

class Yukawa3D : public MatrixAccessor {
public:
  double singularity, alpha;
  Yukawa3D (double s, double a) : singularity(1. / s), alpha(a) {}
  std::complex<double> operator()(double d) const override {
    if (d == 0.)
      return std::complex<double>(singularity, 0.);
    else
      return std::complex<double>(std::exp(-alpha * d) / d, 0.);
  }
};

class Gaussian : public MatrixAccessor {
public:
  double alpha;
  Gaussian (double a) : alpha(1. / (a * a)) {}
  std::complex<double> operator()(double d) const override {
    return std::complex<double>(std::exp(- alpha * d * d), 0.);
  }
};

class Helmholtz3D : public MatrixAccessor {
public:
  double k;
  double singularity;
  Helmholtz3D(double wave_number, double s) : k(wave_number), singularity(1. / s) {}
  std::complex<double> operator()(double d) const override {
    if (d == 0.)
      return std::complex<double>(singularity, 0.);
    else
      return std::exp(std::complex(0., -k * d)) / d;
  }
};

/*
Generates a matrix from a kernel function
In:
  kernel: kernel function
  m: number of rows
  n: number of columns
  bi: points in the row
  bj: points in the column
Out:
  Aij: generated matrix
*/
void gen_matrix(const MatrixAccessor& kernel, const long long M, const long long N, const double* const bi, const double* const bj, std::complex<double> Aij[]);

/*
ACA
In:
  kernel: kernel function
  epsilon: accuracy
  max_rank: maximum rank
  nrows: number of rows
  ncols: number of columns
  row_bodies: row points
  col_bodies: column points
Out:
  row_piv: selected rows (normally ignored)
  col_piv: selected columns
Returns:
  iters: number of iterations (i.e. the real rank)
*/
long long adaptive_cross_approximation(const MatrixAccessor& kernel, const double epsilon, const long long max_rank, const long long nrows, const long long ncols, const double row_bodies[], const double col_bodies[], long long row_piv[], long long col_piv[]);

/*
reference matrix vector multiplication
In:
  kernel: kernel function
  nrows: number of rows
  ncols: number of columns
  X: the vector to be multiplied
  row_bodies: the row points
  col_bodies: the column points
Out:
  B: the result
*/
void mat_vec_reference(const MatrixAccessor& kernel, const long long nrows, const long long ncols, std::complex<double> B[], const std::complex<double> X[], const double row_bodies[], const double col_bodies[]);
