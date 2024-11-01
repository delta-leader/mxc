#pragma once

#include <complex>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <comm-mpi.hpp>

template <typename DT = std::complex<double>>
class MyVector {
  public:
    std::vector<DT> data;
    MyVector(const long long len, DT value=0) {
      data = std::vector<DT>(len, value);
    }
    template <typename OT>
    MyVector(const MyVector<OT>& vector) {
      data = std::vector<DT>(vector.data.size());
      std::transform(vector.data.data(), vector.data.data() + data.size(), data.begin(), [](OT value) -> DT {return DT(value);});
    }
    auto operator& () {
      return &data;
    }
    auto begin() {
      return data.begin();
    }
    auto end() {
      return data.end();
    }
    DT& operator[] (const long long index) {
      return data[index];
    }
    const DT& operator[] (const long long index) const {
      return data[index];
    }
    void generate_random(const long long seed=999, double a=0, double b=1) {
      std::mt19937 gen(seed);
      std::uniform_real_distribution uniform_dist(a, b);
      std::generate(data.begin(), data.end(), 
        [&]() { return (DT) uniform_dist(gen); }
      );
    }
    void zero() {
      std::fill(data.begin(), data.end(), DT{0});
    }
};

template <typename DT = std::complex<double>>
class MatrixAccessor {
public:
  virtual DT operator()(double d) const = 0;
  virtual ~MatrixAccessor() = default;
};

template <typename DT = double>
class Laplace3D : public MatrixAccessor<DT> {
public:
  double singularity;
  Laplace3D (double s) : singularity(1. / s) {}
  DT operator()(double d) const override {
    if (d == 0.)
      return (DT) singularity;
    else
      return (DT) (1. / d);
  }
};

template <typename DT = double>
class Yukawa3D : public MatrixAccessor<DT> {
public:
  double singularity, alpha;
  Yukawa3D (double s, double a) : singularity(1. / s), alpha(a) {}
  DT operator()(double d) const override {
    if (d == 0.)
      return (DT) singularity;
    else
      return (DT) (std::exp(-alpha * d) / d);
  }
};

template <typename DT = double>
class Gaussian : public MatrixAccessor<DT> {
public:
  double alpha, shift;
  Gaussian (double a, double s=0) : alpha(a), shift(s) {}
  DT operator()(double d) const override {
    if (d == 0.)
      return (DT) (std::exp(- alpha * d * d) + shift);
    else
      return (DT) (std::exp(- alpha * d * d));
  }
};

template <typename DT = std::complex<double>>
class Helmholtz3D : public MatrixAccessor<DT> {
public:
  double k;
  double singularity;
  Helmholtz3D(double wave_number, double s) : k(wave_number), singularity(1. / s) {
    std::cout << "Helmholtz-kernel without complex numbers" << std::endl;
  }
  DT operator()(double d) const override {
    if (d == 0.)
      return (DT) singularity;
    else
      return std::real(std::exp(std::complex(0., -k * d)) / d);
  }
};

template <typename DT>
class Helmholtz3D<std::complex<DT>> : public MatrixAccessor<std::complex<DT>> {
public:
  double k;
  double singularity;
  Helmholtz3D(double wave_number, double s) : k(wave_number), singularity(1. / s) {}
  std::complex<DT> operator()(double d) const override {
    if (d == 0.)
      return (std::complex<DT>) singularity;
    else
      return std::exp(std::complex(0., -k * d)) / d;
  }
};

template <typename DT = double>
class Imq : public MatrixAccessor<DT> {
public:
  double alpha, shift;
  Imq (double a, double s=0) : alpha(a), shift(s) {}
  DT operator()(double d) const override {
    if (d == 0.)
      return (DT) (1 / std::sqrt(1 + alpha * d * d) + shift);
    else
      return (DT) (1 / std::sqrt(1 + alpha * d * d));
  }
};

template <typename DT = double>
class Matern : public MatrixAccessor<DT> {
public:
  double alpha, shift;
  double s = std::sqrt(3);
  Matern (double a, double s=0) : alpha(a), shift(s) {}
  DT operator()(double d) const override {
    if (d == 0.)
      return (DT) ((1 + s * alpha * d) * std::exp(-s * alpha * d) + shift);
    else
      return (DT) ((1 + s * alpha * d) * std::exp(-s * alpha * d));
  }
};

template <typename DT>
void gen_matrix(const MatrixAccessor<DT>& eval, long long m, long long n, const double* bi, const double* bj, DT Aij[]);

template <typename DT>
long long adaptive_cross_approximation(double epi, const MatrixAccessor<DT>& eval, long long M, long long N, long long K, const double bi[], const double bj[], long long ipiv[], long long jpiv[], DT u[], DT v[]);

template <typename DT>
void mat_vec_reference(const MatrixAccessor<DT>& eval, long long M, long long N, DT B[], const DT X[], const double ibodies[], const double jbodies[]);

template <typename DT>
double rel_backward_error(const MatrixAccessor<DT>& eval, long long M, long long N, const DT B[], const DT X[], const double ibodies[], const double jbodies[], MPI_Comm world = MPI_COMM_WORLD);