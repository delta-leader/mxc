
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
  Matrix *Uo, *R;
  double *M_cpu, *U_cpu, *R_cpu; 
};

void buildBasis(const EvalDouble& eval, Base basis[], Cell* cells, const CSR* rel_near, int64_t levels,
  const CellComm* comm, const double* bodies, int64_t nbodies, double epi, int64_t alignment);

void basis_free(Base* basis);

void matVecA(const EvalDouble& eval, const Base basis[], const double bodies[], const Cell cells[], const CSR* rels_near, const CSR* rels_far, double X[], const CellComm comm[], int64_t levels);

void solveRelErr(double* err_out, const double* X, const double* ref, int64_t lenX);
