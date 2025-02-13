
#include <h-matrix.hpp>
#include <build_tree.hpp>
#include <kernel.hpp>

#include <Eigen/Dense>
#include <iostream>

LowRankMatrix::LowRankMatrix(double epi, long long m, long long n, long long k, long long p, long long niters, const Accessor& eval, long long iA, long long jA) : M(m), N(n), rank(k), U(M * rank), V(N * rank), S(rank) {
  Zrsvd(epi, m, n, &k, p, niters, eval, iA, jA, S.data(), U.data(), m, V.data(), n);
  if (k < rank) {
    rank = k;
    U.resize(M * rank);
    V.resize(N * rank);
    S.resize(rank);
  }
}

LowRankMatrix::LowRankMatrix(long long m, long long n, long long iA, long long jA, const LowRankMatrix& A) : M(m), N(n), rank(A.rank), U(M * rank), V(N * rank), S(rank) {
  Eigen::Stride<Eigen::Dynamic, 1> ldu(A.M, 1), ldv(A.N, 1);
  Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matU(&A.U[iA], m, rank, ldu);
  Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matV(&A.V[jA], n, rank, ldv);

  Eigen::Map<Eigen::MatrixXcd>(U.data(), m, rank) = matU;
  Eigen::Map<Eigen::MatrixXcd>(V.data(), n, rank) = matV;
  Eigen::Map<Eigen::VectorXd>(S.data(), rank) = Eigen::Map<const Eigen::VectorXd>(A.S.data(), rank);
}

void WellSeparatedApproximation::construct(const Accessor& eval, long long lbegin, long long len, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& upper) {
  WellSeparatedApproximation::lbegin = lbegin;
  lend = lbegin + len;
  M.resize(len);
  for (long long i = upper.lbegin; i < upper.lend; i++)
    for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; c++)
      if (lbegin <= c && c < lend)
        M[c - lbegin] = std::vector<double>(upper.M[i - upper.lbegin].begin(), upper.M[i - upper.lbegin].end());

  long long llen = Far.RowIndex[lend] - Far.RowIndex[lbegin];
  L.reserve(llen);
  
  for (long long y = lbegin; y < lend; y++) {
    for (long long yx = Far.RowIndex[y]; yx < Far.RowIndex[y + 1]; yx++) {
      long long x = Far.ColIndex[yx];
      long long n = cells[x].Body[1] - cells[x].Body[0];
      long long m = cells[y].Body[1] - cells[y].Body[0];
      const double* xbodies = &bodies[3 * cells[x].Body[0]];
      M[y - lbegin].insert(M[y - lbegin].end(), xbodies, &xbodies[3 * n]);
      L.emplace_back(1.e-12, m, n, 200, 200, 2, eval, cells[y].Body[0], cells[x].Body[0]);
    }
  }
}

long long WellSeparatedApproximation::fbodies_size_at_i(long long i) const {
  return 0 <= i && i < (long long)M.size() ? M[i].size() / 3 : 0;
}
  
const double* WellSeparatedApproximation::fbodies_at_i(long long i) const {
  return 0 <= i && i < (long long)M.size() ? M[i].data() : nullptr;
}

