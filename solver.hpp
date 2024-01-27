#pragma once

#include <vector>
#include <cstdint>
#include <complex>

class CSR;
class Cell;
class CellComm;
class Eval;
class MatVecBasis;

class Solver {
public:
  std::vector<int64_t> Dims;
  std::vector<int64_t> DimsLr;

  std::vector<std::complex<double>*> Q;
  std::vector<std::complex<double>*> R;
  std::vector<std::complex<double>> Qdata;
  std::vector<std::complex<double>> Rdata;

  std::vector<std::complex<double>*> A;
  std::vector<int64_t> X;
  std::vector<int64_t> Y;
  std::vector<std::complex<double>> Adata;

  std::vector<std::complex<double>*> F;
  std::vector<int64_t> FX;
  std::vector<int64_t> FY;
  std::vector<std::complex<double>> Fdata;

  Solver(const int64_t dims[], const CSR& Near, const CellComm& comm);

  void setData_leaf(const Eval& eval, const Cell cells[], const double bodies[], const CellComm& comm);

  void setData_far(const Eval& eval, const MatVecBasis& basis, const CellComm& comm);
  
};

