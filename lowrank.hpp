
#pragma once

#include <vector>
#include <cstdint>
#include <complex>

class LowRank {
public:
  std::vector<std::complex<double>> V;
  std::vector<int64_t> Jpiv;
  int64_t N;
  int64_t Rank;

  LowRank(double epi, int64_t M, int64_t N, std::complex<double> A[], int64_t lda);

  void SelectR(void* Xout, const void* Xin, int64_t elem) const;
};
