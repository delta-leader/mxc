#pragma once

#include <data_container.hpp>
#include <complex>

class MatrixAccessor;
class CSR;
class Cell;
class ColCommMPI;

class WellSeparatedApproximation {
private:
  long long lbegin;
  long long lend;
  std::vector<std::vector<double>> M;

public:
  WellSeparatedApproximation() : lbegin(0), lend(0) {}
  WellSeparatedApproximation(const MatrixAccessor& eval, double epi, long long rank, long long lbegin, long long lend, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& upper);

  long long fbodies_size_at_i(long long i) const;
  const double* fbodies_at_i(long long i) const;
};

class H2Matrix {
private:
  std::vector<long long> DimsLr;
  std::vector<long long> UpperStride;
  MatrixDataContainer<std::complex<double>> Q;
  MatrixDataContainer<std::complex<double>> R;
  MatrixDataContainer<double> S;

  std::vector<long long> CRows;
  std::vector<long long> CCols;
  std::vector<std::complex<double>*> C;

  std::vector<long long> ARows;
  std::vector<long long> ACols;
  MatrixDataContainer<std::complex<double>> A;
  std::vector<std::complex<double>*> NA;
  std::vector<int> Ipivots;

  std::vector<std::complex<double>*> NX;
  std::vector<std::complex<double>*> NY;

public:
  std::vector<long long> Dims;
  MatrixDataContainer<std::complex<double>> X;
  MatrixDataContainer<std::complex<double>> Y;
  
  H2Matrix() {}
  H2Matrix(const MatrixAccessor& eval, double epi, const Cell cells[], const CSR& Near, const CSR& Far, const double bodies[], const WellSeparatedApproximation& wsa, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm, bool use_near_bodies = false);

  void matVecUpwardPass(const ColCommMPI& comm);
  void matVecHorizontalandDownwardPass(const ColCommMPI& comm);
  void matVecLeafHorizontalPass(const ColCommMPI& comm);
  void resetX();

  void factorize(const ColCommMPI& comm);
  void factorizeCopyNext(const ColCommMPI& comm, const H2Matrix& lowerA, const ColCommMPI& lowerComm);
  void forwardSubstitute(const ColCommMPI& comm);
  void backwardSubstitute(const ColCommMPI& comm);
};

