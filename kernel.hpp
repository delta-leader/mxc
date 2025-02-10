#pragma once

#include <complex>
#include <cmath>

template<class T> class Accessor {
public:
  long long M, N;
  Accessor(long long M, long long N) : M(M), N(N) {};
  virtual void Aij(long long m, long long n, long long i, long long j, T* A_out, long long stride) const = 0;
  virtual void Aij_mulB(long long mC, long long nC, long long k, long long iA, long long jA, const T* B_in, long long strideB, T* C_out, long long strideC) const = 0;
};

class DenseDMat : public Accessor<double> {
public:
  double* A;
  DenseDMat(long long M, long long N);
  ~DenseDMat();
  void Aij(long long m, long long n, long long i, long long j, double* A_out, long long stride) const override;
  void Aij_mulB(long long mC, long long nC, long long k, long long iA, long long jA, const double* B_in, long long strideB, double* C_out, long long strideC) const override;
};

class DenseZMat : public Accessor<std::complex<double>> {
public:
  std::complex<double>* A;
  DenseZMat(long long M, long long N);
  ~DenseZMat();
  void Aij(long long m, long long n, long long i, long long j, std::complex<double>* A_out, long long stride) const override;
  void Aij_mulB(long long mC, long long nC, long long k, long long iA, long long jA, const std::complex<double>* B_in, long long strideB, std::complex<double>* C_out, long long strideC) const override;
};

void Drsvd(long long m, long long n, long long k, long long p, long long niters, const double* A, long long lda, double* S, double* U, long long ldu, double* V, long long ldv);
void Zrsvd(long long m, long long n, long long k, long long p, long long niters, const std::complex<double>* A, long long lda, double* S, std::complex<double>* U, long long ldu, std::complex<double>* V, long long ldv);

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
  Gaussian (double a) : alpha(a) {}
  std::complex<double> operator()(double d) const override {
    if (d == 0.)
      return std::complex<double>(1.0001, 0.);
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


void gen_matrix(const MatrixAccessor& eval, long long m, long long n, const double* bi, const double* bj, std::complex<double> Aij[]);

