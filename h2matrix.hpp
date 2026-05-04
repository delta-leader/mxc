#pragma once

#include <matrix_container.hpp>
#include <kernel.hpp>
#include <hidr.hpp>
#include <Eigen/Dense>

class MatrixAccessor;
class Hmatrix;
class CSR;
class Cell;
class ColCommMPI;

template <typename DT = std::complex<double>>
class H2Matrix {
private:
  std::vector<long long> UpperStride;
  MatrixDataContainer<double> S;
  MatrixDataContainer<long long> S_ind;
  //MatrixDataContainer<long long> S_ind_orig;

  std::vector<long long> CRows;
  std::vector<long long> CCols;

  std::vector<long long> NA;
  std::vector<long long> NbXoffsets;
  std::vector<long long> NbZoffsets;

public:
  long long lenX;
  long long LowerZ;
  long long n_mat;

  std::vector<long long> Dims;
  std::vector<long long> DimsLr;
  std::vector<long long> dim_offsets;

  std::vector<long long> ARows;
  std::vector<long long> ACols;
  MatrixDataContainer<DT> Q;
  MatrixDataContainer<DT> R;
  MatrixDataContainer<DT> A;
  MatrixDataContainer<DT> C;
  MatrixDataContainer<DT> U;

  MatrixDataContainer<DT> X;
  MatrixDataContainer<DT> Y;
  MatrixDataContainer<DT> Z;
  MatrixDataContainer<DT> W;
  int info;

  bool lowest = false;

  // for storing the assembled matrix
  //MatrixDataContainer<std::complex<double>> Mat;
  // for storing #columns in Mat
  //std::vector<long long> Cols;

  H2Matrix() = default;
  H2Matrix(const H2Matrix& h2matrix);

  //void constructSharedHMatrix(double epi, long long rank, const Cell cells[], const CSR& Far, const Hmatrix& hA, const ColCommMPI& comm, const H2Matrix& Aupper);
  
  //void construct(const MatrixAccessor& eval, double epi, const Cell cells[], const CSR& Near, const double bodies[], const Hmatrix& wsa, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm);
  //void construct(const Eigen::Ref<const Eigen::MatrixXcd>& mat, double epi, const Cell cells[], const CSR& Near, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm);
  //void construct_sparse(const Eigen::Ref<const Eigen::MatrixXcd>& mat, double epi, const Cell cells[], const CSR& Near, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm);
  //void construct(const Eigen::Ref<const Eigen::MatrixXcd>& mat, double epi, const Cell cells[], const CSR& Near, const HiDR& hidr, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm);
  //void constructBLR(const Eigen::Ref<const Eigen::MatrixXcd>& mat, double epi, const Cell cells[], const CSR& Near, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm);
  //void construct(const MatrixGenerator& matgen, double epi, const Cell cells[], const CSR& Near, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm, const double omega, const double scale);
  void construct(const MatrixGenerator& matgen, double epi, const Cell cells[], const CSR& Near, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm, const double omega);
  //void construct_dense(const MatrixGenerator& matgen, double epi, const Cell cells[], const CSR& Near, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm, const double omega);
  void construct_hidr(const MatrixGenerator& matgen, double epi, const Cell cells[], const CSR& Near, const HiDR& hidr, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm, const double omega);
  //void construct_proto(const MatrixGenerator& matgen, double epi, const Cell cells[], const CSR& Near, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm, const double omega, const double scale);
  //void construct(const Eigen::Ref<const Eigen::MatrixXcd>& mat, double epi, const Cell cells[], const CSR& Near, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm, const MatrixGenerator& matgen, const double omega, const double scale);
  
  void matVecUpwardPass(const DT* X_in, const ColCommMPI& comm);
  void matVecDense(const DT* X_in, DT* X_out, const ColCommMPI& comm);
  void matVecHorizontalandDownwardPass(DT* Y_out, const ColCommMPI& comm);
  void matVecLeafHorizontalPass(DT* X_io, const ColCommMPI& comm);

  void factorize(const ColCommMPI& comm);
  void factorizeCopyNext(const H2Matrix& lowerA, const ColCommMPI& lowerComm);
  void forwardSubstitute(const DT* X_in, const ColCommMPI& comm);
  void backwardSubstitute(DT* Y_out, const ColCommMPI& comm);
};

