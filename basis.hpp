#pragma once

#include <vector>
#include <cstdint>
#include <complex>

class Eval;
class CSR;
class Cell;
class CellComm;

class ClusterBasis {
private:
  std::vector<double> Mdata;
  std::vector<std::complex<double>> Vdata;

public:
  std::vector<int64_t> Dims;
  std::vector<int64_t> DimsLr;
  std::vector<std::complex<double>*> V;
  
  ClusterBasis() {}
  ClusterBasis(const Eval& eval, double epi, const Cell cells[], const CSR& Near, const double bodies[], int64_t nbodies, const CellComm& comm);
  ClusterBasis(const Eval& eval, double epi, const ClusterBasis& prev, const Cell cells[], const CSR& Near, const double bodies[], int64_t nbodies, const CellComm& comm, const CellComm& prev_comm);

  const double* ske_at_i(int64_t i) const;
};

class MatVec {
private:
  const Eval* EvalFunc;
  const ClusterBasis* Basis;
  const double* Bodies;
  const Cell* Cells;
  const CSR* Near;
  const CSR* Far;
  const CellComm* Comm;
  int64_t Levels;

public:
  MatVec(const Eval& eval, const ClusterBasis basis[], const double bodies[], const Cell cells[], const CSR& near, const CSR& rels_far, const CellComm comm[], int64_t levels);

  void operator() (int64_t nrhs, std::complex<double> X[], int64_t ldX) const;
};
