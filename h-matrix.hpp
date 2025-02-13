
#pragma once

#include <vector>
#include <complex>

class Accessor;
class CSR;
class Cell;

class LowRankMatrix {
private:
  long long M, N, rank;
  std::vector<std::complex<double>> U, V;
  std::vector<double> S;

public:
  LowRankMatrix(double epi, long long m, long long n, long long k, long long p, long long niters, const Accessor& eval, long long iA, long long jA);
  LowRankMatrix(long long m, long long n, long long iA, long long jA, const LowRankMatrix& A);
};

class WellSeparatedApproximation {
private:
  long long lbegin = 0;
  long long lend = 0;
  std::vector<std::vector<double>> M;
  std::vector<LowRankMatrix> L;

public:
  void construct(const Accessor& eval, long long lbegin, long long lend, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& upper);
  long long fbodies_size_at_i(long long i) const;
  const double* fbodies_at_i(long long i) const;
};

