
#include <h-matrix.hpp>
#include <build_tree.hpp>
#include <kernel.hpp>

LowRankMatrix::LowRankMatrix(const Accessor& eval, long long m, long long n, long long i, long long j) {
  
}

void WellSeparatedApproximation::construct(const Accessor& eval, long long lbegin, long long len, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& upper) {
  WellSeparatedApproximation::lbegin = lbegin;
  lend = lbegin + len;
  M.resize(len);
  for (long long i = upper.lbegin; i < upper.lend; i++)
    for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; c++)
      if (lbegin <= c && c < lend)
        M[c - lbegin] = std::vector<double>(upper.M[i - upper.lbegin].begin(), upper.M[i - upper.lbegin].end());
  
  for (long long y = lbegin; y < lend; y++) {
    for (long long yx = Far.RowIndex[y]; yx < Far.RowIndex[y + 1]; yx++) {
      long long x = Far.ColIndex[yx];
      long long n = cells[x].Body[1] - cells[x].Body[0];
      const double* xbodies = &bodies[3 * cells[x].Body[0]];
      M[y - lbegin].insert(M[y - lbegin].end(), xbodies, &xbodies[3 * n]);
    }
  }
}
  
long long WellSeparatedApproximation::fbodies_size_at_i(long long i) const {
  return 0 <= i && i < (long long)M.size() ? M[i].size() / 3 : 0;
}
  
const double* WellSeparatedApproximation::fbodies_at_i(long long i) const {
  return 0 <= i && i < (long long)M.size() ? M[i].data() : nullptr;
}

