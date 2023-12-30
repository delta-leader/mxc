
#pragma once

#include <cstdint>
#include <vector>

class CSR {
public:
  std::vector<int64_t> RowIndex;
  std::vector<int64_t> ColIndex;

  int64_t lookupIJ(int64_t i, int64_t j) const {
    if (i < 0 || (i + 1) >= (int64_t)RowIndex.size())
    { return -1; }
    const int64_t* col = &ColIndex[0];
    int64_t ibegin = RowIndex[i];
    int64_t iend = RowIndex[i + 1];
    const int64_t* col_iter = &col[ibegin];
    while (col_iter != &col[iend] && *col_iter != j)
      ++col_iter;
    int64_t k = col_iter - col;
    return (k < iend) ? k : -1;
  }

};
