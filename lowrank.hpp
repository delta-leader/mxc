
#pragma once

#include <vector>
#include <cstdint>
#include <complex>

class LowRank {
public:
  std::vector<std::complex<double>> V;
  std::vector<double> BodiesJ;
  int64_t N;
  int64_t Rank;

  LowRank(double epi, int64_t M, int64_t N, std::complex<double> A[], int64_t lda, const double* bj, int64_t incb);
};
