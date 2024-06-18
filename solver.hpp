#pragma once

#include <vector>
#include <complex>

class MatrixAccessor;
class CSR;
class Cell;
class ColCommMPI;
class H2Matrix;

class H2MatrixSolver {
private:
  long long levels;
  std::vector<std::vector<long long>> offsets;
  std::vector<std::vector<long long>> upperIndex;
  std::vector<std::vector<long long>> upperOffsets;

  const H2Matrix* A;
  const ColCommMPI* Comm;

public:
  H2MatrixSolver(const H2Matrix A[], const Cell cells[], const ColCommMPI comm[], long long levels);

  void matVecMul(std::complex<double> X[]) const;
  void solvePrecondition(std::complex<double> X[]) const;
  std::pair<double, long long> solveGMRES(double tol, std::complex<double> X[], const std::complex<double> B[], long long inner_iters, long long outer_iters) const;
};
