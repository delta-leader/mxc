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
  std::vector<std::complex<double>> Rdata;
  std::vector<std::complex<double>> Cdata;

  std::vector<double> Sdata;
  std::vector<const double*> S;
  std::vector<long long> ParentSequenceNum;
  std::vector<long long> elementsOnRow;
  std::vector<long long> localChildOffsets;
  std::vector<long long> localChildLrDims;
  long long localChildIndex;
  long long selfChildIndex;

public:
  std::vector<long long> Dims;
  std::vector<long long> DimsLr;
  std::vector<const std::complex<double>*> Q;
  std::vector<std::complex<double>*> R;

  std::vector<long long> CRows;
  std::vector<long long> CCols;
  std::vector<long long> CColsLocal;
  std::vector<const std::complex<double>*> C;
  
  ClusterBasis() {}
  ClusterBasis(const MatrixAccessor& eval, double epi, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& wsa, const CellComm& comm, const ClusterBasis& prev_basis, const CellComm& prev_comm);
  long long copyOffset(long long i) const;
  long long childWriteOffset() const;
};

class MatVec {
private:
  const MatrixAccessor* EvalFunc;
  const ClusterBasis* Basis;
  const double* Bodies;
  const Cell* Cells;
  const CSR* Near;
  const CellComm* Comm;
  long long Levels;

public:
  MatVec(const MatrixAccessor& eval, const ClusterBasis basis[], const double bodies[], const Cell cells[], const CSR& near, const CellComm comm[], long long levels);

  void operator() (long long nrhs, std::complex<double> X[]) const;
};
