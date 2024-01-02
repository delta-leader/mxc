
#pragma once

#include <vector>
#include <cstdint>

class EvalDouble;
class Matrix;
class CSR;
class Cell;
class CellComm;

class Base {
public:
  int64_t dimR, dimS, dimN;
  std::vector<int64_t> Dims, DimsLr;
  std::vector<Matrix> Uo, R;
  std::vector<double> Mdata, Udata, Rdata;
};

void buildBasis(const EvalDouble& eval, double epi, Base basis[], const Cell* cells, const CSR* rel_near, int64_t levels, const CellComm* comm, const double* bodies, int64_t nbodies);

void matVecA(const EvalDouble& eval, const Base basis[], const double bodies[], const Cell cells[], const CSR* rels_near, const CSR* rels_far, double X[], const CellComm comm[], int64_t levels);

void solveRelErr(double* err_out, const double* X, const double* ref, int64_t lenX);
