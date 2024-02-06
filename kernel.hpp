#pragma once

#include <cstdint>
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


void gen_matrix(const MatrixAccessor& eval, int64_t m, int64_t n, const double* bi, const double* bj, std::complex<double> Aij[], int64_t lda);

int64_t interpolative_decomp_aca(double epi, const MatrixAccessor& eval, int64_t M, int64_t N, int64_t K, const double bi[], const double bj[], int64_t ipiv[], std::complex<double> U[], int64_t ldu);

void mat_vec_reference(const MatrixAccessor& eval, int64_t M, int64_t N, int64_t nrhs, std::complex<double> B[], int64_t ldB, const std::complex<double> X[], int64_t ldX, const double ibodies[], const double jbodies[]);
