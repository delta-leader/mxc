#pragma once

#include <vector>
#include <complex>

class MatrixAccessor;
class CSR;
class Cell;
class ColCommMPI;

class WellSeparatedApproximation {
private:
  long long lbegin;
  long long lend;
  std::vector<std::vector<double>> M;

public:
  WellSeparatedApproximation() : lbegin(0), lend(0) {}
  WellSeparatedApproximation(const MatrixAccessor& eval, double epi, long long rank, long long lbegin, long long lend, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& upper);

  long long fbodies_size_at_i(long long i) const;
  const double* fbodies_at_i(long long i) const;
};

class H2Matrix {
private:
  std::vector<std::complex<double>> Qdata;
  std::vector<std::complex<double>> Adata;
  std::vector<std::complex<double>> Rdata;
  std::vector<double> Sdata;

  std::vector<std::complex<double>*> R;
  std::vector<double*> S;
  std::vector<long long> elementsOnRow;

public:
  std::vector<long long> Dims;
  std::vector<long long> DimsLr;
  std::vector<const std::complex<double>*> Q;

  std::vector<long long> CRows;
  std::vector<long long> CCols;
  std::vector<const std::complex<double>*> C;
  std::vector<long long> Cstride;

  std::vector<long long> ARows;
  std::vector<long long> ACols;
  std::vector<const std::complex<double>*> A;
  std::vector<const std::complex<double>*> NXT;
  std::vector<long long> Nstride;
  
  H2Matrix() {}
  H2Matrix(const MatrixAccessor& eval, double epi, const Cell cells[], const CSR& Near, const CSR& Far, const double bodies[], const WellSeparatedApproximation& wsa, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm, bool use_near_bodies = false);
};

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
  virtual void solvePrecondition(std::complex<double> X[]) const;
  std::pair<double, long long> solveGMRES(double tol, std::complex<double> X[], const std::complex<double> B[], long long inner_iters, long long outer_iters) const;
};
