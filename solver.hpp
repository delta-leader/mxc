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
  H2Matrix* A;
  const ColCommMPI* Comm;

public:
  H2MatrixSolver(H2Matrix A[], const ColCommMPI comm[], long long levels);

  void matVecMul(std::complex<double> X[]);
  void solvePrecondition(std::complex<double> X[]);
  std::pair<double, long long> solveGMRES(double tol, std::complex<double> X[], const std::complex<double> B[], long long inner_iters, long long outer_iters);
};
