#pragma once

#include <vector>
#include <complex>

class MatrixAccessor;
class Cell;
class CSR;
class CellComm;

class HMatrix {
  std::vector<long long> offsets;
public:
  std::vector<std::vector<std::complex<double>>> U;
  std::vector<std::vector<std::complex<double>>> Vh;
  std::vector<long long> M, N, K, Y, X;

  HMatrix() {}
  HMatrix(const MatrixAccessor& eval, double epi, long long rank, const Cell cells[], const CSR& Far, const double bodies[], const CellComm comm[], long long levels);
};
