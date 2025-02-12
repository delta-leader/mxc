#pragma once

#include <matrix_container.hpp>

class MatrixAccessor;
class WellSeparatedApproximation;
class CSR;
class Cell;
class ColCommMPI;

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
  long long lenX;
  long long LowerZ;

  std::vector<long long> Dims;
  std::vector<long long> DimsLr;

  std::vector<long long> ARows;
  std::vector<long long> ACols;
  MatrixDataContainer<std::complex<double>> Q;
  MatrixDataContainer<std::complex<double>> R;
  MatrixDataContainer<std::complex<double>> A;
  MatrixDataContainer<std::complex<double>> C;
  MatrixDataContainer<std::complex<double>> U;

  MatrixDataContainer<std::complex<double>> X;
  MatrixDataContainer<std::complex<double>> Y;
  MatrixDataContainer<std::complex<double>> Z;
  MatrixDataContainer<std::complex<double>> W;
  int info;
  
  void construct(const MatrixAccessor& eval, double epi, const Cell cells[], const CSR& Near, const double bodies[], const WellSeparatedApproximation& wsa, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm);

  void matVecUpwardPass(const std::complex<double>* X_in, const ColCommMPI& comm);
  void matVecHorizontalandDownwardPass(std::complex<double>* Y_out, const ColCommMPI& comm);
  void matVecLeafHorizontalPass(std::complex<double>* X_io, const ColCommMPI& comm);

  void factorize(const ColCommMPI& comm);
  void factorizeCopyNext(const H2Matrix& lowerA, const ColCommMPI& lowerComm);
  void forwardSubstitute(const std::complex<double>* X_in, const ColCommMPI& comm);
  void backwardSubstitute(std::complex<double>* Y_out, const ColCommMPI& comm);
};

