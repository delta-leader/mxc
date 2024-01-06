
#pragma once

#include <vector>
#include <cstdint>
#include <complex>

class Eval;
class Matrix;
class CSR;
class Cell;
class CellComm;

class Base {
public:
  std::vector<int64_t> Dims, DimsLr;
  std::vector<Matrix> Uo, R;
  std::vector<double> Mdata;
  std::vector<std::complex<double>> Udata, Rdata;

  const double* ske_at_i(int64_t i) const;
};

void buildBasis(const Eval& eval, double epi, Base basis[], const Cell* cells, const CSR& rel_near, int64_t levels, const CellComm* comm, const double* bodies, int64_t nbodies);

void matVecA(const Eval& eval, int64_t nrhs, std::complex<double> X[], int64_t ldX, const Base basis[], const double bodies[], const Cell cells[], const CSR& rels_near, const CSR& rels_far, const CellComm comm[], int64_t levels);

void solveRelErr(double* err_out, const std::complex<double>* X, const std::complex<double>* ref, int64_t lenX);
