#pragma once

#include <vector>
#include <complex>

class MatrixAccessor;
class CSR;
class Cell;
class CellComm;

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

class ClusterBasis {
private:
  std::vector<std::complex<double>> Qdata;
  std::vector<std::complex<double>> Cdata;
  std::vector<std::complex<double>> Adata;

  std::vector<double> Sdata;
  std::vector<const double*> S;
  std::vector<long long> ParentSequenceNum;
  std::vector<long long> elementsOnRow;

public:
  std::vector<long long> Dims;
  std::vector<long long> DimsLr;
  std::vector<const std::complex<double>*> Q;

  std::vector<long long> CRows;
  std::vector<long long> CCols;
  std::vector<const std::complex<double>*> C;

  std::vector<long long> ARows;
  std::vector<long long> ACols;
  std::vector<const std::complex<double>*> A;
  
  ClusterBasis() {}
  ClusterBasis(const MatrixAccessor& eval, double epi, const Cell cells[], const CSR& Near, const CSR& Far, const double bodies[], const WellSeparatedApproximation& wsa, const CellComm& comm, const ClusterBasis& prev_basis, const CellComm& prev_comm);
  long long copyOffset(long long i) const;
};

class MatVec {
private:
  std::vector<std::vector<long long>> offsets;
  std::vector<std::vector<long long>> upperIndex;
  std::vector<std::vector<long long>> upperOffsets;

  const ClusterBasis* Basis;
  const CellComm* Comm;
  long long Levels;

public:
  MatVec(const ClusterBasis basis[], const Cell cells[], const CellComm comm[], long long levels);

  void operator() (long long nrhs, std::complex<double> X[]) const;
};
