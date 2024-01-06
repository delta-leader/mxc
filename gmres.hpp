
#pragma once

#include <vector>
#include <cstdint>
#include <complex>

class GMRES {
public:
  int64_t N, M;
  std::vector<std::complex<double>> SIN, COS;
  std::vector<std::complex<double>> ERR;
  
};
