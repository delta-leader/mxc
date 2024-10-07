#pragma once

#include <matrix_container.hpp>
#include <csr_matrix.hpp>

class MatrixAccessor;
class CSR;
class Cell;
class ColCommMPI;
class MatrixDesc;

class WellSeparatedApproximation {
private:
  long long lbegin = 0;
  long long lend = 0;
  std::vector<std::vector<double>> M;

public:
  void construct(const MatrixAccessor& eval, double epi, long long rank, long long lbegin, long long lend, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& upper);
  long long fbodies_size_at_i(long long i) const;
  const double* fbodies_at_i(long long i) const;
};

class H2Matrix {
private:
  std::vector<long long> UpperStride;
  MatrixDataContainer<double> S;

  std::vector<long long> CRows;
  std::vector<long long> CCols;

  std::vector<long long> NA;
  std::vector<long long> NbXoffsets;
  std::vector<long long> NbZoffsets;
  long long LowerZ;

public:
  long long lenX;

  std::vector<long long> Dims;
  std::vector<long long> DimsLr;

  std::vector<long long> ARows;
  std::vector<long long> ACols;
  MatrixDataContainer<std::complex<double>> Q;
  MatrixDataContainer<std::complex<double>> R;
  MatrixDataContainer<std::complex<double>> A;
  MatrixDataContainer<std::complex<double>> C;

  MatrixDataContainer<std::complex<double>> X;
  MatrixDataContainer<std::complex<double>> Y;
  MatrixDataContainer<std::complex<double>> Z;
  MatrixDataContainer<std::complex<double>> W;

  CSRMatrix csrU;
  CSRMatrix csrC;
  CSRMatrix csrA;
  
  void construct(const MatrixAccessor& eval, double epi, const Cell cells[], const CSR& Near, const CSR& Far, const double bodies[], const WellSeparatedApproximation& wsa, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm);

  void matVecUpwardPass(const std::complex<double>* X_in, const ColCommMPI& comm);
  void matVecHorizontalandDownwardPass(std::complex<double>* Y_out, const ColCommMPI& comm);
  void matVecLeafHorizontalPass(std::complex<double>* X_io, const ColCommMPI& comm);

  void factorize(const ColCommMPI& comm);
  void factorizeCopyNext(const H2Matrix& lowerA, const ColCommMPI& lowerComm);
  void forwardSubstitute(const std::complex<double>* X_in, const ColCommMPI& comm);
  void backwardSubstitute(std::complex<double>* Y_out, const ColCommMPI& comm);
};

