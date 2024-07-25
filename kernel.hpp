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
  eval: kernel function
  m: number of rows
  n: number of columns
  bi: points in the row
  bj: points in the column
Out:
  Aij: generated matrix
*/
void gen_matrix(const MatrixAccessor& eval, const long long M, const long long N, const double* const bi, const double* const bj, std::complex<double> Aij[]);

/*
ACA
In:
  epsilon: accuracy
  eval: kernel function
  M: number of rows
  N: number of columns
  max_rank: maximum rank
  bi: row points
  bj: column points
Out:
  ipiv: selected rows 
  jpiv: selected columns
Returns:
  iters: number of iterations (i.e. the real rank)
*/
long long adaptive_cross_approximation(const double epi, const MatrixAccessor& eval, const long long M, const long long N, const long long max_rank, const double bi[], const double bj[], long long ipiv[], long long jpiv[]);

void mat_vec_reference(const MatrixAccessor& eval, long long nrows, long long ncols, std::complex<double> B[], const std::complex<double> X[], const double ibodies[], const double jbodies[]);
