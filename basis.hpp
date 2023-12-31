
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

