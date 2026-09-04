#pragma once

#include <complex>
#include <vector>

#include <Eigen/Dense>
#include <mpi.h>
#include <include/elast3d.hpp>


class MatrixGenerator {
private:
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
  template <typename DT>
  void gen_matrix_single_layer(DT cmat[], const double omega) const;
  template <typename DT>
  void gen_matrix_sorted_single_layer(DT cmat[], long long start, const long long num_rows, const double omega) const;
  template <typename DT>
  void gen_matrix_sorted_single_layer(DT cmat[], long long row_start, const long long num_rows, const long long col_start, const long long num_cols, const double omega) const;
  void gen_rhs_sorted_single_layer(std::complex<double> rhs[], long long start, long long num_rows, const double omega, const double theta = 0) const;
  template <typename DT>
  void gen_matrix_element_single_layer(DT cmat[], const long long row_indices[], const long long num_rows, const long long col_indices[], const long long num_cols, const double omega) const;
  template <typename DT>
  void gen_matrix_idx_element_single_layer(DT cmat[], const long long row_indices[], const long long num_rows, const long long col_indices[], const long long num_cols, const double omega) const;
   template <typename DT>
  void gen_matrix_hidr_sorted_single_layer(DT cmat[], long long row_start, const long long num_rows, const long long col_indices[], const long long num_cols, const double omega) const;
  void writeA(const std::string& filename);
  void readA(const std::string& filename);
  void open_matrix_file(const std::string& filename);
  void open_rhs_file(const std::string& filename);
  void read_mat_metadata_single_layer(double& mat_size, double& omega, double& leaf_size) const;
};

void mat_vec_reference(const MatrixGenerator& matgen, long long M, long long N, std::complex<double> B[], const std::complex<double> X[], const long long row_offset, const double omega);
