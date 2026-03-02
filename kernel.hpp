#pragma once

#include <complex>
#include <cmath>

#include <Eigen/Dense>
#include <include/elast3d.hpp>
#include <mpi.h>

class MatrixGenerator {
private:
  double mu0;
  double mu1;
  long long num_nodes, num_elems;
  std::vector<struct elastWave3d::nodal_point> nodes;
  std::vector<struct elastWave3d::element> elems;
  std::vector<long long> nodes_idx, elems_idx;
  double scale = 1;
  Eigen::MatrixXcd A;
  MPI_File fh_matrix, fh_rhs;
  bool matrix_from_file = false;
  bool rhs_from_file = false;

public:
  MatrixGenerator(const int size, const int spheres=0);
  long long get_num_nodes() const {return num_nodes;};
  long long get_num_elems() const {return num_elems;};
  long long get_num_total() const {return num_elems + num_nodes;};
  const std::vector<struct elastWave3d::nodal_point>& get_nodes() const {return nodes;};
  const std::vector<struct elastWave3d::element>& get_elems() const {return elems;};
  std::vector<long long>& get_nodes_idx() {return nodes_idx;};
  std::vector<long long>& get_elems_idx() {return elems_idx;};
  //void gen_matrix(std::complex<double> cmat[], const double omega, double scale = 0) const;
  void gen_matrix_single_layer(std::complex<double> cmat[], const double omega) const;
  //void gen_matrix_sorted(std::complex<double> cmat[], const double omega, double scale = 0, bool cache = false) const;
  //void gen_matrix_sorted(std::complex<double> cmat[], long long start, const long long num_rows, const double omega, double scale = 0, bool cache = false) const;
  void gen_matrix_sorted_single_layer(std::complex<double> cmat[], long long start, const long long num_rows, const double omega) const;
  void gen_matrix_sorted_single_layer(std::complex<double> cmat[], long long row_start, const long long num_rows, const long long col_start, const long long num_cols, const double omega) const;
  //void gen_matrix_sorted_from_file(std::complex<double> cmat[], long long start, const long long num_rows) const;
  void gen_matrix_sorted_from_file_single_layer(std::complex<double> cmat[], long long start, const long long num_rows) const;
  //void gen_rhs(std::complex<double> rhs[], const double omega, double scale = 0, bool equation_type = true) const;
  //void gen_rhs_sorted(std::complex<double> rhs[], const double omega, double scale = 0, bool equation_type = true) const;
  //void gen_rhs_sorted(std::complex<double> rhs[], long long start, long long num_rows, const double omega, double scale, bool equation_type = true) const;
  void gen_rhs_sorted_single_layer(std::complex<double> rhs[], long long start, long long num_rows, const double omega, const double theta_in = 0, bool equation_type = true) const;
  void gen_rhs_sorted_from_file(std::complex<double> rhs[], long long start, long long num_rows) const;
  //double get_max_nodes(const double omega) const;
  //double get_max_elems(const double omega) const;
  //double calc_scale(const double omega);
  //void gen_matrix_element(std::complex<double> cmat[], const long long row_indices[], const long long num_rows, const long long col_indices[], const long long num_cols, const double omega, double scale = 0) const;
  void gen_matrix_element_single_layer(std::complex<double> cmat[], const long long row_indices[], const long long num_rows, const long long col_indices[], const long long num_cols, const double omega) const;
  void gen_matrix_element_from_file(std::complex<double> cmat[], const long long row_indices[], const long long num_rows, const long long col_indices[], const long long num_cols) const;
  //void gen_matrix_idx_element(std::complex<double> cmat[], const long long row_indices[], const long long num_rows, const long long col_indices[], const long long num_cols, const double omega, double scale = 0) const;
  void gen_matrix_idx_element_single_layer(std::complex<double> cmat[], const long long row_indices[], const long long num_rows, const long long col_indices[], const long long num_cols, const double omega) const;
  void gen_matrix_idx_element_from_file(std::complex<double> cmat[], const long long row_indices[], const long long num_rows, const long long col_indices[], const long long num_cols) const;
  void gen_matrix_hidr_sorted_single_layer(std::complex<double> cmat[], long long row_start, const long long num_rows, const long long col_indices[], const long long num_cols, const double omega) const;
  //void generateA(const double omega, double scale = 0);
  void writeA(const std::string& filename);
  void readA(const std::string& filename);
  void open_matrix_file(const std::string& filename);
  void open_rhs_file(const std::string& filename);
  //void read_mat_metadata(double& mat_size, double& scale, double& omega, double& leaf_size) const;
  void read_mat_metadata_single_layer(double& mat_size, double& omega, double& leaf_size) const;
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
void gen_matrix(const Eigen::Ref<const Eigen::MatrixXcd> &mat, long long m, long long n, const long long* rows, const long long* cols, long long start, Eigen::Ref<Eigen::MatrixXcd> Aij);
void gen_matrix_hidr(const Eigen::Ref<const Eigen::MatrixXcd> &mat, long long m, long long n, const long long* rows, const long long* cols, Eigen::Ref<Eigen::MatrixXcd> Aij);
void gen_matrix(const Eigen::Ref<const Eigen::MatrixXcd> &mat, std::vector<long long>& rows, std::vector<long long>& cols, Eigen::MatrixXcd& Aij);

void mat_vec_reference(const MatrixGenerator& matgen, long long M, long long N, std::complex<double> B[], const std::complex<double> X[], const long long row_offset, const double omega);
