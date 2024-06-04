#pragma once

#include <vector>
#include <complex>

class MatrixAccessor;
class Cell;
class CSR;

class HMatrix {
public:
  std::vector<std::vector<std::complex<double>>> U;
  std::vector<std::vector<std::complex<double>>> Vh;
  std::vector<long long> M, N, K;

  HMatrix() {}
  HMatrix(const MatrixAccessor& eval, double epi, long long rank, long long lbegin, long long lend, const Cell cells[], const CSR& Far, const double bodies[]);
};
