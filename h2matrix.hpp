#pragma once

#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <kernel.hpp>
#include <hidr.hpp>
#include <matrix_container.hpp>


template <typename DT = std::complex<double>>
class H2Matrix {
private:
  std::vector<long long> UpperStride;
  MatrixDataContainer<long long> S_ind;

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

  H2Matrix() = default;
  H2Matrix(const H2Matrix& h2matrix);

  void construct(const MatrixGenerator& matgen, double epi, const Cell cells[], const CSR& Near, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm, const double omega, bool verbose = false);
  void construct_hidr(const MatrixGenerator& matgen, double epi, const Cell cells[], const CSR& Near, const HiDR& hidr, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm, const double omega, bool verbose = false);
  
  void matVecUpwardPass(const DT* X_in, const ColCommMPI& comm);
  void matVecDense(const DT* X_in, DT* X_out, const ColCommMPI& comm);
  void matVecHorizontalandDownwardPass(DT* Y_out, const ColCommMPI& comm);
  void matVecLeafHorizontalPass(DT* X_io, const ColCommMPI& comm);

  void factorize(const ColCommMPI& comm);
  void factorizeCopyNext(const H2Matrix& lowerA, const ColCommMPI& lowerComm);
  void forwardSubstitute(const DT* X_in, const ColCommMPI& comm);
  void backwardSubstitute(DT* Y_out, const ColCommMPI& comm);
  
  void write(long long level, std::string& file) const;
  bool read(long long level, std::string& file);
};

