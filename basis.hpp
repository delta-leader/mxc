#pragma once

#include <vector>
#include <cstdint>
#include <complex>

class MatrixAccessor;
class CSR;
class Cell;
class CellComm;

class WellSeparatedApproximation {
private:
  int64_t lbegin;
  int64_t lend;
  std::vector<std::vector<double>> M;

public:
  WellSeparatedApproximation() : lbegin(0), lend(0) {}
  WellSeparatedApproximation(const MatrixAccessor& eval, double epi, int64_t rank, int64_t lbegin, int64_t lend, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& upper);

  int64_t fbodies_size_at_i(int64_t i) const;
  const double* fbodies_at_i(int64_t i) const;
};

class ClusterBasis {
private:
  std::vector<double> Mdata;
  std::vector<std::complex<double>> Vdata;

public:
  std::vector<int64_t> Dims;
  std::vector<int64_t> DimsLr;
  std::vector<std::complex<double>*> V;
  
  ClusterBasis() {}
  ClusterBasis(const MatrixAccessor& eval, double epi, const Cell cells[], const double bodies[], const WellSeparatedApproximation& wsa, const CellComm& comm, const ClusterBasis& prev_basis, const CellComm& prev_comm);

  const double* ske_at_i(int64_t i) const;
};

class MatVec {
private:
  const MatrixAccessor* EvalFunc;
  const ClusterBasis* Basis;
  const double* Bodies;
  const Cell* Cells;
  const CSR* Near;
  const CSR* Far;
  const CellComm* Comm;
  int64_t Levels;

public:
  MatVec(const MatrixAccessor& eval, const ClusterBasis basis[], const double bodies[], const Cell cells[], const CSR& near, const CSR& rels_far, const CellComm comm[], int64_t levels);

  void operator() (int64_t nrhs, std::complex<double> X[], int64_t ldX) const;
};
