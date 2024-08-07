#pragma once

#include <complex>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

template <typename DT = std::complex<double>>
class Vector_dt {
  private:
    std::vector<DT> data;
  public:
    Vector_dt(const long long len, DT value=0) {
      data = std::vector<DT>(len, value);
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
    void generate_random(const long long seed=999) {
      std::mt19937 gen(seed);
      std::uniform_real_distribution uniform_dist(0., 1.);
      std::generate(data.begin(), data.end(), 
        [&]() { return (DT) uniform_dist(gen); }
      );
    }
    void reset() {
      std::fill(data.begin(), data.end(), DT{0});
    }
};

template <typename T>
class Vector_dt<std::complex<T>> {
  typedef std::complex<T> DT;
  private:
    std::vector<DT> data;
  public:
    Vector_dt(const long long len, DT value=0) {
      data = std::vector<DT>(len, value);
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
    void generate_random(const long long seed=999) {
      std::mt19937 gen(seed);
      std::uniform_real_distribution uniform_dist(0., 1.);
      std::generate(data.begin(), data.end(), 
        [&]() { return DT(uniform_dist(gen), 0.); }
      );
    }
    void reset() {
      std::fill(data.begin(), data.end(), DT{0});
    }
};

template <typename DT = std::complex<double>>
class MatrixAccessor {
public:
  virtual DT operator()(double d) const = 0;
};

template <typename DT = double>
class Laplace3D : public MatrixAccessor<DT> {
public:
  double singularity;
  Laplace3D (double s) : singularity(1. / s) {}
  DT operator()(double d) const override {
    if (d == 0.)
      return singularity;
    else
      return 1. / d;
  }
};

template <typename DT>
class Laplace3D<std::complex<DT>> : public MatrixAccessor<std::complex<DT>> {
public:
  double singularity;
  Laplace3D (double s) : singularity(1. / s) {}
  std::complex<DT> operator()(double d) const override {
    if (d == 0.)
      return std::complex<DT>(singularity, 0.);
    else
      return std::complex<DT>(1. / d, 0.);
  }
};

template <typename DT = double>
class Yukawa3D : public MatrixAccessor<DT> {
public:
  double singularity, alpha;
  Yukawa3D (double s, double a) : singularity(1. / s), alpha(a) {}
  DT operator()(double d) const override {
    if (d == 0.)
      return singularity;
    else
      return std::exp(-alpha * d) / d;
  }
};

template <typename DT>
class Yukawa3D<std::complex<DT>> : public MatrixAccessor<std::complex<DT>> {
public:
  double singularity, alpha;
  Yukawa3D (double s, double a) : singularity(1. / s), alpha(a) {}
  std::complex<DT> operator()(double d) const override {
    if (d == 0.)
      return std::complex<DT>(singularity, 0.);
    else
      return std::complex<DT>(std::exp(-alpha * d) / d, 0.);
  }
};

template <typename DT = double>
class Gaussian : public MatrixAccessor<DT> {
public:
  double alpha;
  Gaussian (double a) : alpha(1. / (a * a)) {}
  DT operator()(double d) const override {
    return std::exp(- alpha * d * d);
  }
};

template <typename DT>
class Gaussian<std::complex<DT>> : public MatrixAccessor<std::complex<DT>> {
public:
  double alpha;
  Gaussian (double a) : alpha(1. / (a * a)) {}
  std::complex<DT> operator()(double d) const override {
    return std::complex<DT>(std::exp(- alpha * d * d), 0.);
  }
};

template <typename DT = std::complex<double>>
class Helmholtz3D : public MatrixAccessor<DT> {
public:
  double k;
  double singularity;
  Helmholtz3D(double wave_number, double s) : k(wave_number), singularity(1. / s) {
    std::cout<<"Helmholtz kernel without complex numbers!"<<std::endl;
  }
  DT operator()(double d) const override {
    if (d == 0.)
      return singularity;
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
      return std::complex<DT>(singularity, 0.);
    else
      return std::exp(std::complex<DT>(0., -k * d)) / d;
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
template <typename DT>
void gen_matrix(const MatrixAccessor<DT>& kernel, const long long M, const long long N, const double* const bi, const double* const bj, DT Aij[]);

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
template <typename DT>
long long adaptive_cross_approximation(const MatrixAccessor<DT>& kernel, const double epsilon, const long long max_rank, const long long nrows, const long long ncols, const double row_bodies[], const double col_bodies[], long long row_piv[], long long col_piv[]);

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
template <typename DT>
void mat_vec_reference(const MatrixAccessor<DT>& kernel, const long long nrows, const long long ncols, DT B[], const DT X[], const double row_bodies[], const double col_bodies[]);
