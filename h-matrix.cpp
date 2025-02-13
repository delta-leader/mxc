
#include <h-matrix.hpp>
#include <build_tree.hpp>
#include <kernel.hpp>

#include <Eigen/Dense>
#include <Eigen/SVD>

LowRankMatrix::LowRankMatrix(double epi, long long m, long long n, long long k, long long p, long long niters, const Accessor& eval, long long iA, long long jA) : M(m), N(n), rank(k), U(m * rank), V(N * rank), S(rank) {
  Zrsvd(epi, m, n, &k, p, niters, eval, iA, jA, S.data(), U.data(), m, V.data(), n);
  if (k < rank) {
    rank = k;
    U.resize(m * rank);
    V.resize(n * rank);
    S.resize(rank);
  }
}

void LowRankMatrix::lowRankSumRow(double epi, long long m, long long n, long long* k, std::complex<double>* A, long long lda, long long lenL, const LowRankMatrix L[]) {
  Eigen::Stride<Eigen::Dynamic, 1> ldA(lda, 1);
  Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matA_in(A, m, n, ldA);
  Eigen::MatrixXcd U = matA_in;
  for (long long i = 0; i < lenL; ++i) {
    long long x = U.cols();
    long long r = L[i].rank;
    U.resize(m, x + r);
    U.rightCols(r) = Eigen::Map<const Eigen::MatrixXcd>(L[i].U.data(), m, r) * Eigen::Map<const Eigen::VectorXd>(L[i].S.data(), r).asDiagonal();
  }

  long long x = U.cols();
  Eigen::MatrixXcd W = Eigen::MatrixXcd::Zero(m, std::min(m, x));
  if (m < x) {
    Eigen::HouseholderQR<Eigen::MatrixXcd> qr(U.adjoint());
    W = qr.matrixQR().triangularView<Eigen::Upper>().adjoint();
  }
  else
    W.leftCols(x) = U;

  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(W, Eigen::ComputeThinU);  
  if (0. < epi && epi < 1.)
  { svd.setThreshold(epi); *k = std::min(*k, (long long)svd.rank()); }

  Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matA_out(A, m, *k, ldA);
  matA_out = svd.matrixU().leftCols(*k);
}

void WellSeparatedApproximation::construct(double epi, const Accessor& eval, long long rank, long long p, long long niters, long long lbegin, long long len, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& upper) {
  WellSeparatedApproximation::lbegin = lbegin;
  lend = lbegin + len;

  long long llen = Far.RowIndex[lend] - Far.RowIndex[lbegin];
  M.resize(len);
  N.resize(len);
  A.resize(len);
  L.reserve(llen);
  m.resize(len);

  for (long long i = upper.lbegin; i < upper.lend; i++)
    for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; c++)
      if (lbegin <= c && c < lend) {
        long long y = cells[c].Body[1] - cells[c].Body[0];
        long long x = upper.N[i - upper.lbegin];
        long long off = cells[c].Body[0] - cells[i].Body[0];
        Eigen::Stride<Eigen::Dynamic, 1> lda(upper.M[i - upper.lbegin], 1);
        Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matA(&(upper.A[i - upper.lbegin])[off], y, x, lda);

        N[c - lbegin] = x;
        A[c - lbegin].reserve(y * rank);
        A[c - lbegin].resize(y * x);
        Eigen::Map<Eigen::MatrixXcd>(A[c - lbegin].data(), y, x) = matA;

        m[c - lbegin].insert(m[c - lbegin].end(), upper.m[i - upper.lbegin].begin(), upper.m[i - upper.lbegin].end());
      }
  
  for (long long y = lbegin; y < lend; y++) {
    M[y - lbegin] = cells[y].Body[1] - cells[y].Body[0];

    for (long long yx = Far.RowIndex[y]; yx < Far.RowIndex[y + 1]; yx++) {
      long long x = Far.ColIndex[yx];
      long long n = cells[x].Body[1] - cells[x].Body[0];
      long long mm = cells[y].Body[1] - cells[y].Body[0];
      const double* xbodies = &bodies[3 * cells[x].Body[0]];
      m[y - lbegin].insert(m[y - lbegin].end(), xbodies, &xbodies[3 * n]);
      L.emplace_back(epi, mm, n, rank, p, niters, eval, cells[y].Body[0], cells[x].Body[0]);
    }

    long long r = 0;
    long long llis = Far.RowIndex[y + 1] - Far.RowIndex[y];
    if (0 < llis) {
      r = std::min(M[y - lbegin], rank);
      if (N[y - lbegin] < r)
        A[y - lbegin].resize(M[y - lbegin] * r);
      const LowRankMatrix* lis = &L[Far.RowIndex[y] - Far.RowIndex[lbegin]];
      LowRankMatrix::lowRankSumRow(epi, M[y - lbegin], N[y - lbegin], &r, A[y - lbegin].data(), M[y - lbegin], llis, lis);
    }

    if (r < rank)
      A[y - lbegin].resize(M[y - lbegin] * r);
    N[y - lbegin] = r;
  }
}

long long WellSeparatedApproximation::fbodies_size_at_i(long long i) const {
  return 0 <= i && i < (long long)m.size() ? m[i].size() / 3 : 0;
}
  
const double* WellSeparatedApproximation::fbodies_at_i(long long i) const {
  return 0 <= i && i < (long long)m.size() ? m[i].data() : nullptr;
}

