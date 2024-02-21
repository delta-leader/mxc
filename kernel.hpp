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


void gen_matrix(const MatrixAccessor& eval, long long m, long long n, const double* bi, const double* bj, std::complex<double> Aij[], long long lda);

long long interpolative_decomp_aca(double epi, const MatrixAccessor& eval, long long M, long long N, long long K, const double bi[], const double bj[], long long ipiv[], std::complex<double> U[], long long ldu);

void mat_vec_reference(const MatrixAccessor& eval, long long M, long long N, long long nrhs, std::complex<double> B[], long long ldB, const std::complex<double> X[], long long ldX, const double ibodies[], const double jbodies[]);
