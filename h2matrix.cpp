#include <h2matrix.hpp>
#include <build_tree.hpp>
#include <comm.hpp>
#include <kernel.hpp>

#include <mkl.h>
#include <algorithm>
#include <numeric>
#include <cmath>

WellSeparatedApproximation::WellSeparatedApproximation(const MatrixAccessor& eval, double epi, long long rank, long long lbegin, long long len, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& upper) :
  lbegin(lbegin), lend(lbegin + len), M(len) {
  std::vector<std::vector<double>> Fbodies(len);
  for (long long i = upper.lbegin; i < upper.lend; i++)
    for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; c++)
      if (lbegin <= c && c < lend)
        M[c - lbegin] = std::vector<double>(upper.M[i - upper.lbegin].begin(), upper.M[i - upper.lbegin].end());

  for (long long y = lbegin; y < lend; y++) {
    for (long long yx = Far.RowIndex[y]; yx < Far.RowIndex[y + 1]; yx++) {
      long long x = Far.ColIndex[yx];
      long long m = cells[y].Body[1] - cells[y].Body[0];
      long long n = cells[x].Body[1] - cells[x].Body[0];
      const double* Xbodies = &bodies[3 * cells[x].Body[0]];
      const double* Ybodies = &bodies[3 * cells[y].Body[0]];

      long long k = std::min(rank, std::min(m, n));
      std::vector<long long> ipiv(k);
      std::vector<std::complex<double>> U(n * k);
      long long iters = interpolative_decomp_aca(epi, eval, n, m, k, Xbodies, Ybodies, &ipiv[0], &U[0]);
      std::vector<double> Fbodies(3 * iters);
      for (long long i = 0; i < iters; i++)
        std::copy(&Xbodies[3 * ipiv[i]], &Xbodies[3 * (ipiv[i] + 1)], &Fbodies[3 * i]);
      M[y - lbegin].insert(M[y - lbegin].end(), Fbodies.begin(), Fbodies.end());
    }
  }
}

long long WellSeparatedApproximation::fbodies_size_at_i(long long i) const {
  return 0 <= i && i < (long long)M.size() ? M[i].size() / 3 : 0;
}

const double* WellSeparatedApproximation::fbodies_at_i(long long i) const {
  return 0 <= i && i < (long long)M.size() ? M[i].data() : nullptr;
}

long long compute_basis(const MatrixAccessor& eval, double epi, long long M, long long N, double Xbodies[], const double Fbodies[], std::complex<double> A[]) {
  long long K = std::min(M, N), rank = 0;
  std::complex<double> one(1., 0.), zero(0., 0.);
  std::vector<std::complex<double>> B(M * N), TAU(M);
  std::vector<long long> jpiv(M, 0);

  if (0 < K) {
    gen_matrix(eval, N, M, Fbodies, Xbodies, &B[0]);
    if (K < N) {
      LAPACKE_zgeqrf(LAPACK_COL_MAJOR, N, M, &B[0], N, &TAU[0]);
      LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, zero, zero, &B[1], N);
    }

    LAPACKE_zgeqp3(LAPACK_COL_MAJOR, K, M, &B[0], N, &jpiv[0], &TAU[0]);
    double s0 = epi * std::abs(B[0]);
    if (std::numeric_limits<double>::min() < s0)
      while (rank < K && s0 <= std::abs(B[rank * (N + 1)]))
        ++rank;
  }

  if (0 < rank && rank < M) {
    cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, rank, M - rank, &one, &B[0], N, &B[rank * N], N);
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'F', rank, rank, zero, one, &B[0], N);
    MKL_Zomatcopy('C', 'T', rank, M, one, &B[0], N, A, M);

    for (long long i = 0; i < M; i++) {
      long long piv = std::distance(&jpiv[0], std::find(&jpiv[i], &jpiv[M], i + 1));
      jpiv[piv] = jpiv[i];
      jpiv[i] = piv + 1;
    }
    LAPACKE_zlaswp(LAPACK_COL_MAJOR, rank, A, M, 1, M, &jpiv[0], 1);
    LAPACKE_dlaswp(LAPACK_ROW_MAJOR, 3, Xbodies, 3, 1, M, &jpiv[0], -1);
  }
  else
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'F', M, M, zero, one, A, M);
  return rank;
}

H2Matrix::H2Matrix(const MatrixAccessor& eval, double epi, const Cell cells[], const CSR& Near, const CSR& Far, const double bodies[], const WellSeparatedApproximation& wsa, const CellComm& comm, const H2Matrix& prev_basis, const CellComm& prev_comm) {
  long long xlen = comm.lenNeighbors();
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long ybegin = comm.oGlobal();

  long long ychild = cells[ybegin].Child[0];
  long long localChildIndex = prev_comm.iLocal(ychild);
  std::vector<long long> localChildOffsets(nodes + 1);
  localChildOffsets[0] = 0;
  std::transform(&cells[ybegin], &cells[ybegin + nodes], localChildOffsets.begin() + 1, [=](const Cell& c) { return c.Child[1] - ychild; });

  std::vector<long long> localChildLrDims(localChildOffsets.back());
  std::copy(&prev_basis.DimsLr[localChildIndex], &prev_basis.DimsLr[localChildIndex + localChildLrDims.size()], localChildLrDims.begin());

  Dims = std::vector<long long>(xlen, 0);
  DimsLr = std::vector<long long>(xlen, 0);
  ParentSequenceNum = std::vector<long long>(xlen);
  elementsOnRow = std::vector<long long>(xlen);
  S = std::vector<const double*>(xlen);
  Q = std::vector<const std::complex<double>*>(xlen);

  std::transform(&cells[ybegin], &cells[ybegin + nodes], &ParentSequenceNum[ibegin], 
    [&](const Cell& c) { return 0 <= c.Parent ? std::distance(&cells[cells[c.Parent].Child[0]], &c) : 0; });
  if (localChildOffsets.back() == 0)
    std::transform(&cells[ybegin], &cells[ybegin + nodes], &Dims[ibegin], [](const Cell& c) { return c.Body[1] - c.Body[0]; });
  else
    std::transform(localChildOffsets.begin(), localChildOffsets.begin() + nodes, localChildOffsets.begin() + 1, &Dims[ibegin], 
      [&](long long start, long long end) { return std::reduce(&localChildLrDims[start], &localChildLrDims[end]); });

  const std::vector<long long> ones(xlen, 1);
  comm.neighbor_bcast(Dims.data(), ones.data());
  comm.neighbor_bcast(ParentSequenceNum.data(), ones.data());

  std::vector<long long> Qoffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), elementsOnRow.begin(), [](const long long d) { return d * d; });
  std::inclusive_scan(elementsOnRow.begin(), elementsOnRow.end(), Qoffsets.begin() + 1);
  Qoffsets[0] = 0;
  Qdata = std::vector<std::complex<double>>(Qoffsets[xlen], std::complex<double>(0., 0.));
  std::transform(Qoffsets.begin(), Qoffsets.begin() + xlen, Q.begin(), [&](const long long d) { return &Qdata[d]; });

  std::vector<long long> Ssizes(xlen), Soffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), Ssizes.begin(), [](const long long d) { return 3 * d; });
  std::inclusive_scan(Ssizes.begin(), Ssizes.end(), Soffsets.begin() + 1);
  Soffsets[0] = 0;
  Sdata = std::vector<double>(Soffsets[xlen], 0.);
  std::transform(Soffsets.begin(), Soffsets.begin() + xlen, S.begin(), [&](const long long d) { return &Sdata[d]; });

  for (long long i = 0; i < nodes; i++) {
    long long ci = i + ybegin;
    long long childi = localChildIndex + localChildOffsets[i];
    long long cend = localChildIndex + localChildOffsets[i + 1];
    double* ske_i = &Sdata[Soffsets[i + ibegin]];

    if (cend <= childi)
      std::copy(&bodies[3 * cells[ci].Body[0]], &bodies[3 * cells[ci].Body[1]], ske_i);
    for (long long j = childi; j < cend; j++) {
      long long offset = prev_basis.copyOffset(j);
      long long len = prev_basis.DimsLr[j];
      std::copy(prev_basis.S[j], prev_basis.S[j] + (len * 3), &ske_i[offset * 3]);
    }
  }
  comm.neighbor_bcast(Sdata.data(), Ssizes.data());

  ARows = std::vector<long long>(&Near.RowIndex[ybegin], &Near.RowIndex[ybegin + nodes + 1]);
  ACols = std::vector<long long>(&Near.ColIndex[ARows[0]], &Near.ColIndex[ARows[nodes]]);
  long long offset = ARows[0];
  std::for_each(ARows.begin(), ARows.end(), [=](long long& i) { i = i - offset; });
  std::for_each(ACols.begin(), ACols.end(), [&](long long& col) { col = comm.iLocal(col); });

  std::vector<long long> Asizes(ARows[nodes]), Aoffsets(ARows[nodes] + 1);
  for (long long i = 0; i < nodes; i++)
    std::transform(&ACols[ARows[i]], &ACols[ARows[i + 1]], &Asizes[ARows[i]], 
      [&](long long col) { return Dims[i + ibegin] * Dims[col]; });
  std::inclusive_scan(Asizes.begin(), Asizes.end(), Aoffsets.begin() + 1);
  Aoffsets[0] = 0;

  A = std::vector<const std::complex<double>*>(ARows[nodes]);
  Adata = std::vector<std::complex<double>>(Aoffsets.back());
  std::transform(Aoffsets.begin(), Aoffsets.begin() + ARows[nodes], A.begin(), [&](const long long d) { return &Adata[d]; });

  for (long long i = 0; i < nodes; i++)
    for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
      long long j = ACols[ij];
      long long M = Dims[i + ibegin];
      long long N = Dims[j];
      gen_matrix(eval, M, N, S[i + ibegin], S[j], &Adata[Aoffsets[ij]]);
    }

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    std::complex<double>* matrix = &Qdata[Qoffsets[i + ibegin]];
    double* ske_i = &Sdata[Soffsets[i + ibegin]];

    long long fsize = wsa.fbodies_size_at_i(i);
    const double* fbodies = wsa.fbodies_at_i(i);
    long long rank = compute_basis(eval, epi, M, fsize, ske_i, fbodies, matrix);
    DimsLr[i + ibegin] = rank;
  }

  comm.neighbor_bcast(DimsLr.data(), ones.data());
  comm.neighbor_bcast(Sdata.data(), Ssizes.data());
  comm.neighbor_bcast(Qdata.data(), elementsOnRow.data());

  CRows = std::vector<long long>(&Far.RowIndex[ybegin], &Far.RowIndex[ybegin + nodes + 1]);
  CCols = std::vector<long long>(&Far.ColIndex[CRows[0]], &Far.ColIndex[CRows[nodes]]);
  offset = CRows[0];
  std::for_each(CRows.begin(), CRows.end(), [=](long long& i) { i = i - offset; });
  std::for_each(CCols.begin(), CCols.end(), [&](long long& col) { col = comm.iLocal(col); });

  std::vector<long long> Csizes(CRows[nodes]), Coffsets(CRows[nodes] + 1);
  for (long long i = 0; i < nodes; i++)
    std::transform(&CCols[CRows[i]], &CCols[CRows[i + 1]], &Csizes[CRows[i]], 
      [&](long long col) { return DimsLr[i + ibegin] * DimsLr[col]; });
  std::inclusive_scan(Csizes.begin(), Csizes.end(), Coffsets.begin() + 1);
  Coffsets[0] = 0;

  C = std::vector<const std::complex<double>*>(CRows[nodes]);
  Cdata = std::vector<std::complex<double>>(Coffsets.back());
  std::transform(Coffsets.begin(), Coffsets.begin() + CRows[nodes], C.begin(), [&](const long long d) { return &Cdata[d]; });

  for (long long i = 0; i < nodes; i++)
    for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
      long long j = CCols[ij];
      long long M = DimsLr[i + ibegin], N = DimsLr[j];
      gen_matrix(eval, M, N, S[i + ibegin], S[j], &Cdata[Coffsets[ij]]);
    }
}

long long H2Matrix::copyOffset(long long i) const {
  return std::reduce(&DimsLr[std::max((long long)0, i - ParentSequenceNum[i])], &DimsLr[i], (long long)0);
}

MatVec::MatVec(const H2Matrix basis[], const Cell cells[], const CellComm comm[], long long levels) :
  offsets(levels + 1), upperIndex(levels + 1), upperOffsets(levels + 1), rhsX(levels + 1), rhsY(levels + 1), Basis(basis), Comm(comm), Levels(levels) {
  
  for (long long l = levels; l >= 0; l--) {
    long long xlen = comm[l].lenNeighbors();
    offsets[l] = std::vector<long long>(xlen + 1, 0);
    upperIndex[l] = std::vector<long long>(xlen, 0);
    upperOffsets[l] = std::vector<long long>(xlen, 0);
    std::inclusive_scan(basis[l].Dims.begin(), basis[l].Dims.end(), offsets[l].begin() + 1);
    rhsX[l] = std::vector<std::complex<double>>(offsets[l][xlen], std::complex<double>(0., 0.));
    rhsY[l] = std::vector<std::complex<double>>(offsets[l][xlen], std::complex<double>(0., 0.));

    if (l < levels)
      for (long long i = 0; i < xlen; i++) {
        long long ci = comm[l].iGlobal(i);
        long long child = comm[l + 1].iLocal(cells[ci].Child[0]);
        long long clen = cells[ci].Child[1] - cells[ci].Child[0];

        if (child >= 0 && clen > 0) {
          std::fill(&upperIndex[l + 1][child], &upperIndex[l + 1][child + clen], i);
          std::exclusive_scan(&basis[l + 1].DimsLr[child], &basis[l + 1].DimsLr[child + clen], &upperOffsets[l + 1][child], 0);
        }
      }
  }
}

void MatVec::operator() (std::complex<double> X[]) {
  long long lbegin = Comm[Levels].oLocal();
  long long llen = Comm[Levels].lenLocal();
  long long lenX = offsets[Levels][lbegin + llen] - offsets[Levels][lbegin];

  const std::complex<double> one(1., 0.), zero(0., 0.);
  std::copy(X, &X[lenX], rhsX[Levels].data() + offsets[Levels][lbegin]);

  for (long long i = Levels; i >= 0; i--) {
    long long ibegin = Comm[i].oLocal();
    long long iboxes = Comm[i].lenLocal();
    long long xlen = Comm[i].lenNeighbors();
    Comm[i].level_merge(rhsX[i].data(), offsets[i][xlen]);

    if (0 < i) {
      std::fill(rhsX[i - 1].begin(), rhsX[i - 1].end(), zero);
      for (long long y = 0; y < iboxes; y++) {
        long long M = Basis[i].Dims[y + ibegin];
        long long N = Basis[i].DimsLr[y + ibegin];
        long long U = upperIndex[i][y + ibegin];

        const std::complex<double>* Xptr = rhsX[i].data() + offsets[i][ibegin + y];
        std::complex<double>* XOptr = rhsX[i - 1].data() + offsets[i - 1][U] + upperOffsets[i][ibegin + y];
        if (0 < N)
          cblas_zgemv(CblasColMajor, CblasTrans, M, N, &one, Basis[i].Q[y + ibegin], M, Xptr, 1, &zero, XOptr, 1);
      }
    }
  }

  for (long long i = Levels; i >= 0; i--)
    Comm[i].neighbor_bcast(rhsX[i].data(), Basis[i].Dims.data());
  std::fill(rhsY[0].begin(), rhsY[0].end(), zero);

  for (long long i = 1; i <= Levels; i++) {
    long long ibegin = Comm[i].oLocal();
    long long iboxes = Comm[i].lenLocal();
    std::fill(rhsY[i].begin(), rhsY[i].end(), zero);

    for (long long y = 0; y < iboxes; y++) {
      long long M = Basis[i].Dims[y + ibegin];
      long long K = Basis[i].DimsLr[y + ibegin];
      long long UY = upperIndex[i][y + ibegin];

      std::complex<double>* Yptr = rhsY[i].data() + offsets[i][ibegin + y];
      std::complex<double>* YOptr = rhsY[i - 1].data() + offsets[i - 1][UY] + upperOffsets[i][ibegin + y];

      if (0 < K) {
        for (long long yx = Basis[i].CRows[y]; yx < Basis[i].CRows[y + 1]; yx++) {
          long long x = Basis[i].CCols[yx];
          long long N = Basis[i].DimsLr[x];
          long long UX = upperIndex[i][x];

          const std::complex<double>* XOptr = rhsX[i - 1].data() + offsets[i - 1][UX] + upperOffsets[i][x];
          cblas_zgemv(CblasColMajor, CblasNoTrans, K, N, &one, Basis[i].C[yx], K, XOptr, 1, &one, YOptr, 1);
        }
        cblas_zgemv(CblasColMajor, CblasNoTrans, M, K, &one, Basis[i].Q[y + ibegin], M, YOptr, 1, &zero, Yptr, 1);
      }
    }
  }

  for (long long y = 0; y < llen; y++)
    for (long long yx = Basis[Levels].ARows[y]; yx < Basis[Levels].ARows[y + 1]; yx++) {
      long long x = Basis[Levels].ACols[yx];
      long long M = Basis[Levels].Dims[lbegin + y];
      long long N = Basis[Levels].Dims[x];

      std::complex<double>* Yptr = rhsY[Levels].data() + offsets[Levels][lbegin + y];
      const std::complex<double>* Xptr = rhsX[Levels].data() + offsets[Levels][x];
      cblas_zgemv(CblasColMajor, CblasNoTrans, M, N, &one, Basis[Levels].A[yx], M, Xptr, 1, &one, Yptr, 1);
    }
  
  std::copy(rhsY[Levels].data() + offsets[Levels][lbegin], rhsY[Levels].data() + offsets[Levels][lbegin + llen], X);
}

double residual(long long N, std::complex<double> R[], const std::complex<double> X[], const std::complex<double> B[], MatVec& A, const Preconditioner& M, const CellComm& comm) {
  std::fill(R, &R[N], std::complex<double>(0., 0.));
  const std::complex<double> one(1., 0.), minus_one(-1., 0.);

  cblas_zaxpy(N, &minus_one, X, 1, R, 1);
  A(R);
  cblas_zaxpy(N, &one, B, 1, R, 1);
  M.solve(R); // r = M^(-1) * (b - A * x)

  std::complex<double> beta(0., 0.);
  cblas_zdotc_sub(N, R, 1, R, 1, &beta);
  comm.level_sum(&beta, 1);
  return std::sqrt(beta.real()); // beta = || r ||_2
}

void smoother_gmres(long long N, long long iters, double beta, std::complex<double> X[], const std::complex<double> R[], MatVec& A, const Preconditioner& M, const CellComm& comm) {
  long long ld = iters + 1;
  std::vector<std::complex<double>> H(iters * ld, std::complex<double>(0., 0.));
  std::vector<std::complex<double>> v(N * ld, std::complex<double>(0., 0.));
  std::vector<std::complex<double>> s(ld, std::complex<double>(0., 0.));
  std::vector<std::complex<double>> w(N, std::complex<double>(0., 0.));

  std::vector<std::complex<double>> dotp(ld, std::complex<double>(0., 0.));
  std::vector<long long> jpvt(iters, 0);
  std::complex<double> scale(1. / beta, 0.);
  cblas_zaxpy(N, &scale, R, 1, &v[0], 1);
  s[0] = beta;
  
  for (long long i = 0; i < iters; i++) {
    std::copy(&v[i * N], &v[(i + 1) * N], w.begin());
    A(&w[0]);
    M.solve(&w[0]);

    for (long long k = 0; k <= i; k++)
      cblas_zdotc_sub(N, &v[k * N], 1, &w[0], 1, &dotp[k]);
    comm.level_sum(&dotp[0], i + 1);

    for (long long k = 0; k <= i; k++) {
      scale = -dotp[k];
      H[k + i * ld] = dotp[k];
      cblas_zaxpy(N, &scale, &v[k * N], 1, &w[0], 1);
    }

    cblas_zdotc_sub(N, &w[0], 1, &w[0], 1, &dotp[0]);
    comm.level_sum(&dotp[0], 1);

    double normw = std::sqrt(dotp[0].real());
    H[(i + 1) + i * ld] = std::complex<double>(normw, 0.);
    scale = std::complex<double>(1. / normw, 0.);
    cblas_zaxpy(N, &scale, &w[0], 1, &v[(i + 1) * N], 1);
  }

  long long rank = 0;
  const double machine_epsilon = std::numeric_limits<double>::epsilon();
  const std::complex<double> one(1., 0.);
  LAPACKE_zgelsy(LAPACK_COL_MAJOR, ld, iters, 1, &H[0], ld, &s[0], ld, &jpvt[0], machine_epsilon, &rank);
  cblas_zgemv(CblasColMajor, CblasNoTrans, N, iters, &one, &v[0], N, &s[0], 1, &one, X, 1);
}

std::pair<double, long long> MatVec::solveGMRES(double tol, const Preconditioner& M, std::complex<double> X[], const std::complex<double> B[], long long inner_iters, long long outer_iters) {
  long long lbegin = Comm[Levels].oLocal();
  long long llen = Comm[Levels].lenLocal();
  long long lenX = offsets[Levels][lbegin + llen] - offsets[Levels][lbegin];

  std::vector<std::complex<double>> r(B, &B[lenX]);
  std::complex<double> dotp(0., 0.);
  M.solve(&r[0]);
  cblas_zdotc_sub(lenX, &r[0], 1, &r[0], 1, &dotp);
  Comm[Levels].level_sum(&dotp, 1);
  double normb = std::sqrt(dotp.real());
  if (normb == 0.)
    normb = 1.;

  for (long long i = 0; i < outer_iters; i++) {
    double beta = residual(lenX, &r[0], X, B, *this, M, Comm[Levels]);
    double resid = beta / normb;
    if (resid < tol)
      return std::make_pair(resid, i);

    smoother_gmres(lenX, inner_iters, beta, X, &r[0], *this, M, Comm[Levels]);
  }

  double beta = residual(lenX, &r[0], X, B, *this, M, Comm[Levels]);
  double resid = beta / normb;
  return std::make_pair(resid, outer_iters);
}
