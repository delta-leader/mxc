#pragma once

#include <complex>
#include <cmath>

#include <Eigen/Dense>
#include <include/elast3d.hpp>

class MatrixGenerator {
private:
  double omega;
  const std::complex<double> alpha;
  const double mu0;
  const double mu1;
public:
  MatrixGenerator(double omega=1.0);
  void gen_matrix(const elastWave3d::nodal_point* xnodes, long long num_xnodes, const elastWave3d::element* xelems, long long num_xelems, const elastWave3d::nodal_point* ynodes, long long num_ynodes, const elastWave3d::element* yelems, long long num_yelems, std::complex<double> cmat[]) const;
  void gen_matrix_sorted(const elastWave3d::nodal_point* xnodes, long long num_xnodes, const elastWave3d::element* xelems, long long num_xelems, const elastWave3d::nodal_point* ynodes, long long num_ynodes, const elastWave3d::element* yelems, long long num_yelems, std::vector<long long>& nodes_indices, std::vector<long long>& elems_indices, const double scale, std::complex<double> cmat[]) const;
  void gen_rhs(const elastWave3d::nodal_point* nodes, long long num_nodes, const elastWave3d::element* elems, long long num_elems, std::complex<double> rhs[], bool equation_type=true) const;
  double get_max_nodes(const elastWave3d::nodal_point* xnodes, long long num_xnodes, const elastWave3d::element* xelems, long long num_xelems) const;
  double get_max_elems(const elastWave3d::nodal_point* xnodes, long long num_xnodes, const elastWave3d::element* xelems, long long num_xelems) const;
};

class Accessor {
public:
  long long M, N;
  Accessor(long long M, long long N) : M(M), N(N) {};
  virtual void op_Aij_mulB(char opA, long long mC, long long nC, long long k, long long iA, long long jA, const std::complex<double>* B_in, long long strideB, std::complex<double>* C_out, long long strideC) const = 0;
};

class DenseZMat : public Accessor {
public:
  std::complex<double>* A;
  DenseZMat(long long M, long long N);
  ~DenseZMat();
  void op_Aij_mulB(char opA, long long mC, long long nC, long long k, long long iA, long long jA, const std::complex<double>* B_in, long long strideB, std::complex<double>* C_out, long long strideC) const override;
};

void Zrsvd(double epi, long long m, long long n, long long* k, long long p, long long niters, const Accessor& A, long long iA, long long jA, double* S, std::complex<double>* U, long long ldu, std::complex<double>* V, long long ldv);

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
void gen_matrix(const Eigen::Ref<const Eigen::MatrixXcd> &mat, long long m, long long n, const long long* rows, const long long* cols, Eigen::Ref<Eigen::MatrixXcd> Aij);
void gen_matrix_hidr(const Eigen::Ref<const Eigen::MatrixXcd> &mat, long long m, long long n, const long long* rows, const long long* cols, Eigen::Ref<Eigen::MatrixXcd> Aij);
void gen_matrix(const Eigen::Ref<const Eigen::MatrixXcd> &mat, std::vector<long long>& rows, std::vector<long long>& cols, Eigen::MatrixXcd& Aij);

