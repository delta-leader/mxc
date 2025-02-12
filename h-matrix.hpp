
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

public:
  LowRankMatrix(const Accessor& eval, long long m, long long n, long long i, long long j);
};

class WellSeparatedApproximation {
private:
  long long lbegin = 0;
  long long lend = 0;
  std::vector<std::vector<double>> M;

public:
  void construct(const Accessor& eval, long long lbegin, long long lend, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& upper);
  long long fbodies_size_at_i(long long i) const;
  const double* fbodies_at_i(long long i) const;
};

