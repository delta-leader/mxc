
#pragma once

#include <vector>
#include <cstdint>
#include <complex>

class LowRank {
public:
  std::vector<std::complex<double>> U;
  std::vector<std::complex<double>> V;
  int64_t M;
  int64_t N;
  int64_t Rank;

  LowRank(double epi, int64_t M, int64_t N, int64_t K, int64_t P, const std::complex<double> A[], int64_t lda);
};
