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

long long compute_basis(const MatrixAccessor& eval, double epi, long long M, long long N, double Xbodies[], const double Fbodies[], std::complex<double> A[], std::complex<double> C[]) {
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
    MKL_Zomatcopy('C', 'T', rank, M, one, &B[0], N, C, M);

    for (long long i = 0; i < M; i++) {
      long long piv = std::distance(jpiv.begin(), std::find(jpiv.begin() + i, jpiv.end(), i + 1));
      jpiv[piv] = jpiv[i];
      jpiv[i] = piv + 1;
    }
    LAPACKE_zlaswp(LAPACK_COL_MAJOR, rank, C, M, 1, M, &jpiv[0], 1);
    LAPACKE_dlaswp(LAPACK_ROW_MAJOR, 3, Xbodies, 3, 1, M, &jpiv[0], -1);

    cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, M, rank, &one, A, M, C, M);
    LAPACKE_zgeqrf(LAPACK_COL_MAJOR, M, rank, C, M, &TAU[0]);
    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'L', M, rank, C, M, A, M);
    LAPACKE_zungqr(LAPACK_COL_MAJOR, M, rank, rank, A, M, &TAU[0]);
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', rank - 1, rank - 1, zero, zero, &C[1], M);
  }
  else {
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'F', M, M, zero, one, A, M);
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'F', M, M, zero, one, C, M);
  }
  return rank;
}

H2Matrix::H2Matrix(const MatrixAccessor& eval, double epi, const Cell cells[], const CSR& Near, const CSR& Far, const double bodies[], const WellSeparatedApproximation& wsa, const CellComm& comm, const H2Matrix& lowerA, const CellComm& lowerComm) {
  long long xlen = comm.lenNeighbors();
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long ybegin = comm.oGlobal();

  long long ychild = cells[ybegin].Child[0];
  long long localChildIndex = lowerComm.iLocal(ychild);
  std::vector<long long> localChildOffsets(nodes + 1);
  localChildOffsets[0] = 0;
  std::transform(&cells[ybegin], &cells[ybegin + nodes], localChildOffsets.begin() + 1, [=](const Cell& c) { return c.Child[1] - ychild; });

  Dims = std::vector<long long>(xlen, 0);
  DimsLr = std::vector<long long>(xlen, 0);
  elementsOnRow = std::vector<long long>(xlen, 0);
  Q = std::vector<const std::complex<double>*>(xlen, nullptr);
  R = std::vector<std::complex<double>*>(xlen, nullptr);
  S = std::vector<double*>(xlen, nullptr);

  if (localChildOffsets.back() == 0)
    std::transform(&cells[ybegin], &cells[ybegin + nodes], &Dims[ibegin], [](const Cell& c) { return c.Body[1] - c.Body[0]; });
  else {
    std::vector<long long>::const_iterator iter = lowerA.DimsLr.begin() + localChildIndex;
    std::transform(localChildOffsets.begin(), localChildOffsets.begin() + nodes, localChildOffsets.begin() + 1, &Dims[ibegin],
      [&](long long start, long long end) { return std::reduce(iter + start, iter + end); });
  }

  const std::vector<long long> ones(xlen, 1);
  comm.neighbor_bcast(Dims.data(), ones.data());

  std::vector<long long> Qoffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), elementsOnRow.begin(), [](const long long d) { return d * d; });
  std::inclusive_scan(elementsOnRow.begin(), elementsOnRow.end(), Qoffsets.begin() + 1);
  Qoffsets[0] = 0;
  Qdata = std::vector<std::complex<double>>(Qoffsets[xlen], std::complex<double>(0., 0.));
  Rdata = std::vector<std::complex<double>>(Qoffsets[xlen], std::complex<double>(0., 0.));

  std::vector<long long> Ssizes(xlen), Soffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), Ssizes.begin(), [](const long long d) { return 3 * d; });
  std::inclusive_scan(Ssizes.begin(), Ssizes.end(), Soffsets.begin() + 1);
  Soffsets[0] = 0;
  Sdata = std::vector<double>(Soffsets[xlen], 0.);

  ARows = std::vector<long long>(Near.RowIndex.begin() + ybegin, Near.RowIndex.begin() + ybegin + nodes + 1);
  ACols = std::vector<long long>(Near.ColIndex.begin() + ARows[0], Near.ColIndex.begin() + ARows[nodes]);
  long long offset = ARows[0];
  std::for_each(ARows.begin(), ARows.end(), [=](long long& i) { i = i - offset; });
  std::for_each(ACols.begin(), ACols.end(), [&](long long& col) { col = comm.iLocal(col); });

  CRows = std::vector<long long>(Far.RowIndex.begin() + ybegin, Far.RowIndex.begin() + ybegin + nodes + 1);
  CCols = std::vector<long long>(Far.ColIndex.begin() + CRows[0], Far.ColIndex.begin() + CRows[nodes]);
  offset = CRows[0];
  std::for_each(CRows.begin(), CRows.end(), [=](long long& i) { i = i - offset; });
  std::for_each(CCols.begin(), CCols.end(), [&](long long& col) { col = comm.iLocal(col); });

  std::vector<long long> Asizes(ARows[nodes]), Aoffsets(ARows[nodes] + 1);
  std::vector<long long> Csizes(CRows[nodes]), Coffsets(CRows[nodes] + 1);
  for (long long i = 0; i < nodes; i++)
    std::transform(ACols.begin() + ARows[i], ACols.begin() + ARows[i + 1], Asizes.begin() + ARows[i],
      [&](long long col) { return Dims[i + ibegin] * Dims[col]; });
  std::inclusive_scan(Asizes.begin(), Asizes.end(), Aoffsets.begin() + 1);
  Aoffsets[0] = 0;

  A = std::vector<const std::complex<double>*>(ARows[nodes]);
  Adata = std::vector<std::complex<double>>(Aoffsets.back());

  if (Qoffsets.back()) {
    std::transform(Qoffsets.begin(), Qoffsets.begin() + xlen, Q.begin(), [&](const long long d) { return &Qdata[d]; });
    std::transform(Qoffsets.begin(), Qoffsets.begin() + xlen, R.begin(), [&](const long long d) { return &Rdata[d]; });
    std::transform(Soffsets.begin(), Soffsets.begin() + xlen, S.begin(), [&](const long long d) { return &Sdata[d]; });
    std::transform(Aoffsets.begin(), Aoffsets.begin() + ARows[nodes], A.begin(), [&](const long long d) { return &Adata[d]; });

    const std::complex<double> one(1., 0.);
    for (long long i = 0; i < nodes; i++) {
      long long M = Dims[i + ibegin];
      long long ci = i + ybegin;
      long long childi = localChildIndex + localChildOffsets[i];
      long long cend = localChildIndex + localChildOffsets[i + 1];
      std::complex<double>* matQ = &Qdata[Qoffsets[i + ibegin]];

      if (cend <= childi) {
        std::copy(&bodies[3 * cells[ci].Body[0]], &bodies[3 * cells[ci].Body[1]], S[i + ibegin]);
        for (long long j = 0; j < M; j++)
          matQ[j * (M + 1)] = one;
      }
      for (long long j = childi; j < cend; j++) {
        long long offset = std::reduce(&lowerA.DimsLr[childi], &lowerA.DimsLr[j]);
        long long len = lowerA.DimsLr[j];
        std::copy(lowerA.S[j], lowerA.S[j] + (len * 3), &(S[i + ibegin])[offset * 3]);
        MKL_Zomatcopy('C', 'N', len, len, one, lowerA.R[j], lowerA.Dims[j], &matQ[offset * (M + 1)], M);
      }
    }
    comm.neighbor_bcast(Sdata.data(), Ssizes.data());
    comm.neighbor_bcast(Qdata.data(), elementsOnRow.data());

    for (long long i = 0; i < nodes; i++)
      for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
        long long j = ACols[ij];
        long long M = Dims[i + ibegin], N = Dims[j];
        if (0 < M && 0 < N) {
          std::complex<double>* Aij = &Adata[Aoffsets[ij]];
          gen_matrix(eval, M, N, S[i + ibegin], S[j], Aij);
          cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, M, N, &one, Q[i + ibegin], M, Aij, M);
          cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, M, N, &one, Q[j], N, Aij, M);
        }
      }

    for (long long i = 0; i < nodes; i++) {
      long long fsize = wsa.fbodies_size_at_i(i);
      const double* fbodies = wsa.fbodies_at_i(i);
      long long rank = compute_basis(eval, epi, Dims[i + ibegin], fsize, S[i + ibegin], fbodies, &Qdata[Qoffsets[i + ibegin]], R[i + ibegin]);
      DimsLr[i + ibegin] = rank;
    }

    comm.neighbor_bcast(DimsLr.data(), ones.data());
    comm.neighbor_bcast(Sdata.data(), Ssizes.data());
    comm.neighbor_bcast(Qdata.data(), elementsOnRow.data());
    comm.neighbor_bcast(Rdata.data(), elementsOnRow.data());
  }

  for (long long i = 0; i < nodes; i++)
    std::transform(CCols.begin() + CRows[i], CCols.begin() + CRows[i + 1], Csizes.begin() + CRows[i],
      [&](long long col) { return DimsLr[i + ibegin] * DimsLr[col]; });
  std::inclusive_scan(Csizes.begin(), Csizes.end(), Coffsets.begin() + 1);
  Coffsets[0] = 0;

  C = std::vector<const std::complex<double>*>(CRows[nodes]);
  Cdata = std::vector<std::complex<double>>(Coffsets.back());

  if (Coffsets.back()) {
    std::transform(Coffsets.begin(), Coffsets.begin() + CRows[nodes], C.begin(), [&](const long long d) { return &Cdata[d]; });

    const std::complex<double> one(1., 0.);
    for (long long i = 0; i < nodes; i++)
      for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
        long long j = CCols[ij];
        long long M = DimsLr[i + ibegin], N = DimsLr[j];
        if (0 < M && 0 < N) {
          std::complex<double>* Cij = &Cdata[Coffsets[ij]];
          gen_matrix(eval, M, N, S[i + ibegin], S[j], Cij);
          cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, M, N, &one, R[i + ibegin], Dims[i + ibegin], Cij, M);
          cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, M, N, &one, R[j], Dims[j], Cij, M);
        }
      }
  }
}

H2MatrixSolver::H2MatrixSolver(const H2Matrix A[], const Cell cells[], const CellComm comm[], long long levels) :
  Levels(levels), offsets(levels + 1), upperIndex(levels + 1), upperOffsets(levels + 1), A(A), Comm(comm) {
  
  for (long long l = levels; l >= 0; l--) {
    long long xlen = comm[l].lenNeighbors();
    offsets[l] = std::vector<long long>(xlen + 1, 0);
    upperIndex[l] = std::vector<long long>(xlen, 0);
    upperOffsets[l] = std::vector<long long>(xlen, 0);
    std::inclusive_scan(A[l].Dims.begin(), A[l].Dims.end(), offsets[l].begin() + 1);

    if (l < levels)
      for (long long i = 0; i < xlen; i++) {
        long long ci = comm[l].iGlobal(i);
        long long child = comm[l + 1].iLocal(cells[ci].Child[0]);
        long long clen = cells[ci].Child[1] - cells[ci].Child[0];

        if (child >= 0 && clen > 0) {
          std::fill(upperIndex[l + 1].begin() + child, upperIndex[l + 1].begin() + child + clen, i);
          std::exclusive_scan(A[l + 1].DimsLr.begin() + child, A[l + 1].DimsLr.begin() + child + clen, upperOffsets[l + 1].begin() + child, 0ll);
        }
      }
  }
}

void H2MatrixSolver::matVecMul(std::complex<double> X[], long long levels) const {
  levels = 0 <= levels ? std::min(levels, Levels) : Levels;
  long long lbegin = Comm[levels].oLocal();
  long long llen = Comm[levels].lenLocal();
  long long lenX = offsets[levels][lbegin + llen] - offsets[levels][lbegin];

  std::vector<std::vector<std::complex<double>>> rhsX(levels + 1);
  std::vector<std::vector<std::complex<double>>> rhsY(levels + 1);
  const std::complex<double> one(1., 0.), zero(0., 0.);

  for (long long l = levels; l >= 0; l--) {
    long long xlen = Comm[l].lenNeighbors();
    rhsX[l] = std::vector<std::complex<double>>(offsets[l][xlen], zero);
    rhsY[l] = std::vector<std::complex<double>>(offsets[l][xlen], zero);
  }
  if (X)
    std::copy(X, &X[lenX], rhsX[levels].data() + offsets[levels][lbegin]);

  for (long long l = levels; l >= 0; l--) {
    long long ibegin = Comm[l].oLocal();
    long long iboxes = Comm[l].lenLocal();
    long long xlen = Comm[l].lenNeighbors();
    Comm[l].level_merge(rhsX[l].data(), offsets[l][xlen]);
    Comm[l].neighbor_bcast(rhsX[l].data(), A[l].Dims.data());

    if (0 < l)
      for (long long y = 0; y < iboxes; y++) {
        long long M = A[l].Dims[y + ibegin];
        long long N = A[l].DimsLr[y + ibegin];
        long long U = upperIndex[l][y + ibegin];

        const std::complex<double>* Xptr = rhsX[l].data() + offsets[l][ibegin + y];
        std::complex<double>* XOptr = rhsX[l - 1].data() + offsets[l - 1][U] + upperOffsets[l][ibegin + y];
        if (0 < N)
          cblas_zgemv(CblasColMajor, CblasTrans, M, N, &one, A[l].Q[y + ibegin], M, Xptr, 1, &zero, XOptr, 1);
      }
  }

  for (long long l = 1; l <= levels; l++) {
    long long ibegin = Comm[l].oLocal();
    long long iboxes = Comm[l].lenLocal();

    for (long long y = 0; y < iboxes; y++) {
      long long M = A[l].Dims[y + ibegin];
      long long K = A[l].DimsLr[y + ibegin];
      long long UY = upperIndex[l][y + ibegin];

      if (0 < K) {
        std::complex<double>* Yptr = rhsY[l].data() + offsets[l][ibegin + y];
        std::complex<double>* YOptr = rhsY[l - 1].data() + offsets[l - 1][UY] + upperOffsets[l][ibegin + y];

        for (long long yx = A[l].CRows[y]; yx < A[l].CRows[y + 1]; yx++) {
          long long x = A[l].CCols[yx];
          long long N = A[l].DimsLr[x];
          long long UX = upperIndex[l][x];

          const std::complex<double>* XOptr = rhsX[l - 1].data() + offsets[l - 1][UX] + upperOffsets[l][x];
          cblas_zgemv(CblasColMajor, CblasNoTrans, K, N, &one, A[l].C[yx], K, XOptr, 1, &one, YOptr, 1);
        }
        cblas_zgemv(CblasColMajor, CblasNoTrans, M, K, &one, A[l].Q[y + ibegin], M, YOptr, 1, &zero, Yptr, 1);
      }
    }
  }

  for (long long y = 0; y < llen; y++) {
    long long M = A[levels].Dims[lbegin + y];
    if (0 < M)
      for (long long yx = A[levels].ARows[y]; yx < A[levels].ARows[y + 1]; yx++) {
        long long x = A[levels].ACols[yx];
        long long N = A[levels].Dims[x];

        std::complex<double>* Yptr = rhsY[levels].data() + offsets[levels][lbegin + y];
        const std::complex<double>* Xptr = rhsX[levels].data() + offsets[levels][x];
        cblas_zgemv(CblasColMajor, CblasNoTrans, M, N, &one, A[levels].A[yx], M, Xptr, 1, &one, Yptr, 1);
      }
  }
  if (X)
    std::copy(rhsY[levels].data() + offsets[levels][lbegin], rhsY[levels].data() + offsets[levels][lbegin + llen], X);
}

void H2MatrixSolver::solvePrecondition(std::complex<double>[], long long) const {
  // Default preconditioner = I
}

double residual(long long N, std::complex<double> R[], const std::complex<double> X[], const std::complex<double> B[], long long levels, const H2MatrixSolver& A, const CellComm& comm) {
  std::fill(R, &R[N], std::complex<double>(0., 0.));
  const std::complex<double> one(1., 0.), minus_one(-1., 0.);

  cblas_zaxpy(N, &minus_one, X, 1, R, 1);
  A.matVecMul(R, levels);
  cblas_zaxpy(N, &one, B, 1, R, 1);
  A.solvePrecondition(R, levels); // r = M^(-1) * (b - A * x)

  std::complex<double> beta(0., 0.);
  cblas_zdotc_sub(N, R, 1, R, 1, &beta);
  comm.level_sum(&beta, 1);
  return std::sqrt(beta.real()); // beta = || r ||_2
}

void smoother_gmres(long long N, long long iters, double beta, std::complex<double> X[], const std::complex<double> R[], long long levels, const H2MatrixSolver& A, const CellComm& comm) {
  long long ld = iters + 1;
  std::vector<std::complex<double>> H(iters * ld, std::complex<double>(0., 0.));
  std::vector<std::complex<double>> v(N * ld, std::complex<double>(0., 0.));
  std::vector<std::complex<double>> s(ld, std::complex<double>(0., 0.));

  std::vector<long long> jpvt(iters, 0);
  std::complex<double> scale(1. / beta, 0.);
  cblas_zaxpy(N, &scale, R, 1, &v[0], 1);
  
  for (long long i = 0; i < iters; i++) {
    std::complex<double>* w = &v[(i + 1) * N];
    std::copy(&v[i * N], w, w);
    A.matVecMul(w, levels);
    A.solvePrecondition(w, levels);

    for (long long k = 0; k <= i; k++)
      cblas_zdotc_sub(N, &v[k * N], 1, w, 1, &s[k]);
    comm.level_sum(&s[0], i + 1);

    for (long long k = 0; k <= i; k++) {
      scale = -s[k];
      H[k + i * ld] = s[k];
      cblas_zaxpy(N, &scale, &v[k * N], 1, w, 1);
    }

    cblas_zdotc_sub(N, w, 1, w, 1, &s[0]);
    comm.level_sum(&s[0], 1);

    double normw = std::sqrt(s[0].real());
    H[(i + 1) + i * ld] = std::complex<double>(normw, 0.);
    scale = std::complex<double>(1. / normw, 0.);
    cblas_zscal(N, &scale, w, 1);
  }

  long long rank = 0;
  const double machine_epsilon = std::numeric_limits<double>::epsilon();
  std::fill(s.begin() + 1, s.end(), std::complex<double>(0., 0.));
  s[0] = beta;
  scale = std::complex<double>(1., 0.);

  LAPACKE_zgelsy(LAPACK_COL_MAJOR, ld, iters, 1, &H[0], ld, &s[0], ld, &jpvt[0], machine_epsilon, &rank);
  if (0 < N)
    cblas_zgemv(CblasColMajor, CblasNoTrans, N, iters, &scale, &v[0], N, &s[0], 1, &scale, X, 1);
}

std::pair<double, long long> H2MatrixSolver::solveGMRES(double tol, std::complex<double> X[], const std::complex<double> B[], long long inner_iters, long long outer_iters) const {
  long long lbegin = Comm[Levels].oLocal();
  long long llen = Comm[Levels].lenLocal();
  long long lenX = offsets[Levels][lbegin + llen] - offsets[Levels][lbegin];

  std::vector<std::complex<double>> r(B, &B[lenX]);
  std::complex<double> dotp(0., 0.);
  solvePrecondition(&r[0], Levels);
  cblas_zdotc_sub(lenX, &r[0], 1, &r[0], 1, &dotp);
  Comm[Levels].level_sum(&dotp, 1);
  double normb = std::sqrt(dotp.real());
  if (normb == 0.)
    normb = 1.;

  for (long long i = 0; i < outer_iters; i++) {
    double beta = residual(lenX, &r[0], X, B, Levels, *this, Comm[Levels]);
    double resid = beta / normb;
    if (resid < tol)
      return std::make_pair(resid, i);

    smoother_gmres(lenX, inner_iters, beta, X, &r[0], Levels, *this, Comm[Levels]);
  }

  double beta = residual(lenX, &r[0], X, B, Levels, *this, Comm[Levels]);
  double resid = beta / normb;
  return std::make_pair(resid, outer_iters);
}

std::pair<double, long long> H2MatrixSolver::VcycleMG(double tol, std::complex<double> X[], const std::complex<double> B[], long long inner_iters, long long outer_iters) const {
  long long lbegin = Comm[Levels].oLocal();
  long long llen = Comm[Levels].lenLocal();
  long long lenX = offsets[Levels][lbegin + llen] - offsets[Levels][lbegin];

  std::vector<std::vector<std::complex<double>>> rhsX(Levels + 1);
  std::vector<std::vector<std::complex<double>>> rhsB(Levels + 1);
  std::vector<std::vector<std::complex<double>>> R(Levels + 1);
  const std::complex<double> one(1., 0.), zero(0., 0.);

  for (long long l = Levels; l >= 0; l--) {
    long long xlen = Comm[l].lenNeighbors();
    rhsX[l] = std::vector<std::complex<double>>(offsets[l][xlen], zero);
    rhsB[l] = std::vector<std::complex<double>>(offsets[l][xlen], zero);
    R[l] = std::vector<std::complex<double>>(offsets[l][xlen], zero);
  }

  std::complex<double>* leafX = rhsX[Levels].data() + offsets[Levels][lbegin];
  std::complex<double>* leafB = rhsB[Levels].data() + offsets[Levels][lbegin];
  std::complex<double>* leafR = R[Levels].data() + offsets[Levels][lbegin];
  std::copy(B, &B[lenX], leafB);
  std::copy(X, &X[lenX], leafX);

  std::complex<double> dotp(0., 0.);
  solvePrecondition(leafB, Levels);
  cblas_zdotc_sub(lenX, leafB, 1, leafB, 1, &dotp);
  Comm[Levels].level_sum(&dotp, 1);
  double normb = std::sqrt(dotp.real());
  if (normb == 0.)
    normb = 1.;

  for (long long i = 0; i < outer_iters; i++) {
    double beta = residual(lenX, leafR, leafX, B, Levels, *this, Comm[Levels]);
    double resid = beta / normb;
    if (resid < tol) {
      std::copy(leafX, &leafX[lenX], X);
      return std::make_pair(resid, i);
    }

    for (long long l = Levels; l >= 0; l--) {
      long long ibegin = Comm[l].oLocal();
      long long iboxes = Comm[l].lenLocal();
      long long xlen = Comm[l].lenNeighbors();
      long long dlen = offsets[l][ibegin + iboxes] - offsets[l][ibegin];
      Comm[l].level_merge(rhsB[l].data(), offsets[l][xlen]);
      Comm[l].neighbor_bcast(rhsB[l].data(), A[l].Dims.data());

      std::complex<double>* Rptr = R[l].data() + offsets[l][ibegin];
      std::complex<double>* Bptr = rhsB[l].data() + offsets[l][ibegin];
      std::complex<double>* Xptr = rhsX[l].data() + offsets[l][ibegin];
      
      double beta = residual(dlen, Rptr, Xptr, Bptr, l, *this, Comm[l]);
      smoother_gmres(dlen, inner_iters, beta, Xptr, Rptr, l, *this, Comm[l]);
      residual(dlen, Rptr, Xptr, Bptr, l, *this, Comm[l]);

      if (0 < l)
        for (long long y = 0; y < iboxes; y++) {
          long long M = A[l].Dims[y + ibegin];
          long long N = A[l].DimsLr[y + ibegin];
          long long U = upperIndex[l][y + ibegin];

          Rptr = R[l].data() + offsets[l][ibegin + y];
          Bptr = rhsB[l - 1].data() + offsets[l - 1][U] + upperOffsets[l][ibegin + y];
          if (0 < N)
            cblas_zgemv(CblasColMajor, CblasConjTrans, M, N, &one, A[l].Q[y + ibegin], M, Rptr, 1, &zero, Bptr, 1);
        }
    }

    for (long long l = 1; l <= Levels; l++) {
      long long ibegin = Comm[l].oLocal();
      long long iboxes = Comm[l].lenLocal();
      long long dlen = offsets[l][ibegin + iboxes] - offsets[l][ibegin];

      for (long long y = 0; y < iboxes; y++) {
        long long M = A[l].Dims[y + ibegin];
        long long N = A[l].DimsLr[y + ibegin];
        long long U = upperIndex[l][y + ibegin];

        std::complex<double>* Xptr = rhsX[l].data() + offsets[l][ibegin + y];
        const std::complex<double>* XOptr = rhsX[l - 1].data() + offsets[l - 1][U] + upperOffsets[l][ibegin + y];
        if (0 < N)
          cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, 1, M, N, &one, XOptr, 1, A[l].Q[y + ibegin], M, &one, Xptr, 1);
      }

      std::complex<double>* Rptr = R[l].data() + offsets[l][ibegin];
      std::complex<double>* Bptr = rhsB[l].data() + offsets[l][ibegin];
      std::complex<double>* Xptr = rhsX[l].data() + offsets[l][ibegin];

      double beta = residual(dlen, Rptr, Xptr, Bptr, l, *this, Comm[l]);
      smoother_gmres(dlen, inner_iters, beta, Xptr, Rptr, l, *this, Comm[l]);
    }
  }

  std::copy(leafX, &leafX[lenX], X);
  double beta = residual(lenX, leafR, leafX, B, Levels, *this, Comm[Levels]);
  double resid = beta / normb;
  return std::make_pair(resid, outer_iters);
}
