#pragma once

#include <matrix_container.hpp>

template <typename DT>
class MatrixAccessor;
class CSR;
class Cell;
class ColCommMPI;
class MatrixDesc;

template <typename DT = std::complex<double>>
class WellSeparatedApproximation {
private:
  long long lbegin = 0;
  long long lend = 0;
  std::vector<std::vector<double>> M;

public:
  void construct(const MatrixAccessor<DT>& eval, double epi, long long rank, long long lbegin, long long lend, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation<DT>& upper);
  long long fbodies_size_at_i(long long i) const;
  const double* fbodies_at_i(long long i) const;
};

template <typename DT = std::complex<double>>
class H2Matrix {
private:
  std::vector<long long> UpperStride;
  MatrixDataContainer<double> S;

  std::vector<long long> CRows;
  std::vector<long long> CCols;

  std::vector<long long> NA;
  std::vector<long long> NbXoffsets;
  std::vector<long long> NbZoffsets;

public:
  template <typename OT> friend class H2Matrix;
  long long lenX;
  long long LowerZ;

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
  template <typename OT>
  H2Matrix(const H2Matrix<OT>& h2matrix);
  
  void construct(const MatrixAccessor<DT>& eval, double epi, const Cell cells[], const CSR& Near, const CSR& Far, const double bodies[], const WellSeparatedApproximation<DT>& wsa, const ColCommMPI& comm, H2Matrix<DT>& lowerA, const ColCommMPI& lowerComm);

  void matVecUpwardPass(const DT* X_in, const ColCommMPI& comm);
  void matVecHorizontalandDownwardPass(DT* Y_out, const ColCommMPI& comm);
  void matVecLeafHorizontalPass(DT* X_io, const ColCommMPI& comm);

  void factorize(const ColCommMPI& comm);
  void factorizeCopyNext(const H2Matrix& lowerA, const ColCommMPI& lowerComm);
  void forwardSubstitute(const DT* X_in, const ColCommMPI& comm);
  void backwardSubstitute(DT* Y_out, const ColCommMPI& comm);
};

