#pragma once

#include <matrix_container.hpp>
#include <vector>
#include <numeric>
#include <complex>

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
  std::vector<long long> C;

  std::vector<long long> NA;
  std::vector<long long> LowerX;
  std::vector<long long> NbXoffsets;

public:
  std::vector<long long> Dims;
  std::vector<long long> DimsLr;

  std::vector<long long> ARows;
  std::vector<long long> ACols;
  MatrixDataContainer<std::complex<double>> Q;
  MatrixDataContainer<std::complex<double>> R;
  MatrixDataContainer<std::complex<double>> A;

  MatrixDataContainer<std::complex<double>> X;
  MatrixDataContainer<std::complex<double>> Y;
  
  void construct(const MatrixAccessor& eval, double epi, const Cell cells[], const CSR& Near, const CSR& Far, const double bodies[], const WellSeparatedApproximation& wsa, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm);

  void upwardCopyNext(char src, char dst, const ColCommMPI& comm, const H2Matrix& lowerA);
  void downwardCopyNext(char src, char dst, const H2Matrix& upperA, const ColCommMPI& upperComm);

  void matVecUpwardPass(const ColCommMPI& comm);
  void matVecHorizontalandDownwardPass(const H2Matrix& upperA, const ColCommMPI& comm);
  void matVecLeafHorizontalPass(const ColCommMPI& comm);

  void factorize(const ColCommMPI& comm);
  void factorizeCopyNext(const H2Matrix& lowerA, const ColCommMPI& lowerComm);
  void forwardSubstitute(const ColCommMPI& comm);
  void backwardSubstitute(const ColCommMPI& comm);
};

