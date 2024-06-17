#include <h2matrix.hpp>
#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <kernel.hpp>

#include <mkl.h>
#include <eigen3/Eigen/Dense>
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
  std::vector<int> jpiv(M, 0);

  if (0 < K) {
    gen_matrix(eval, N, M, Fbodies, Xbodies, &B[0]);
    if (K < N) {
      LAPACKE_zgeqrf(LAPACK_COL_MAJOR, N, M, &B[0], N, &TAU[0]);
      LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, zero, zero, &B[1], N);
    }

    LAPACKE_zgeqp3(LAPACK_COL_MAJOR, K, M, &B[0], N, &jpiv[0], &TAU[0]);
    rank = std::min(K, (long long)std::floor(epi));
    double s0 = epi * std::abs(B[0]);
    if (std::numeric_limits<double>::min() < s0 && epi < 1.)
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

long long lookupIJ(const std::vector<long long>& RowIndex, const std::vector<long long>& ColIndex, long long i, long long j) {
  if (i < 0 || RowIndex.size() <= (1ull + i))
    return -1;
  long long k = std::distance(ColIndex.begin(), std::find(ColIndex.begin() + RowIndex[i], ColIndex.begin() + RowIndex[i + 1], j));
  return (k < RowIndex[i + 1]) ? k : -1;
}

H2Matrix::H2Matrix(const MatrixAccessor& eval, double epi, const Cell cells[], const CSR& Near, const CSR& Far, const double bodies[], const WellSeparatedApproximation& wsa, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm, bool use_near_bodies) {
  long long xlen = comm.lenNeighbors();
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long ybegin = comm.oGlobal();
  long long pbegin = lowerComm.oLocal();

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

  ARows = std::vector<long long>(Near.RowIndex.begin() + ybegin, Near.RowIndex.begin() + ybegin + nodes + 1);
  ACols = std::vector<long long>(Near.ColIndex.begin() + ARows[0], Near.ColIndex.begin() + ARows[nodes]);
  long long offset = ARows[0];
  std::for_each(ARows.begin(), ARows.end(), [=](long long& i) { i = i - offset; });
  std::for_each(ACols.begin(), ACols.end(), [&](long long& col) { col = comm.iLocal(col); });
  A = std::vector<const std::complex<double>*>(ARows[nodes]);
  NXT = std::vector<const std::complex<double>*>(ARows[nodes], nullptr);
  Nstride = std::vector<long long>(ARows[nodes], 0);

  CRows = std::vector<long long>(Far.RowIndex.begin() + ybegin, Far.RowIndex.begin() + ybegin + nodes + 1);
  CCols = std::vector<long long>(Far.ColIndex.begin() + CRows[0], Far.ColIndex.begin() + CRows[nodes]);
  offset = CRows[0];
  std::for_each(CRows.begin(), CRows.end(), [=](long long& i) { i = i - offset; });
  std::for_each(CCols.begin(), CCols.end(), [&](long long& col) { col = comm.iLocal(col); });
  C = std::vector<const std::complex<double>*>(CRows[nodes], nullptr);
  Cstride = std::vector<long long>(CRows[nodes], 0);

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

  std::vector<long long> Asizes(ARows[nodes]), Aoffsets(ARows[nodes] + 1);
  std::vector<long long> Csizes(CRows[nodes]), Coffsets(CRows[nodes] + 1);
  for (long long i = 0; i < nodes; i++)
    std::transform(ACols.begin() + ARows[i], ACols.begin() + ARows[i + 1], Asizes.begin() + ARows[i],
      [&](long long col) { return Dims[i + ibegin] * Dims[col]; });
  std::inclusive_scan(Asizes.begin(), Asizes.end(), Aoffsets.begin() + 1);
  Aoffsets[0] = 0;
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

    for (long long i = 0; i < nodes; i++) {
      long long childi = localChildIndex + localChildOffsets[i];
      long long cendi = localChildIndex + localChildOffsets[i + 1];

      for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
        long long j = ACols[ij];
        long long M = Dims[i + ibegin], N = Dims[j];

        long long j_global = Near.ColIndex[ij + Near.RowIndex[ybegin]];
        long long childj = lowerComm.iLocal(cells[j_global].Child[0]);
        long long cendj = (0 <= childj) ? (childj + cells[j_global].Child[1] - cells[j_global].Child[0]) : -1;

        if (0 < M && 0 < N) {
          std::complex<double>* Aij = &Adata[Aoffsets[ij]];
          gen_matrix(eval, M, N, S[i + ibegin], S[j], Aij);
          cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, M, N, &one, Q[i + ibegin], M, Aij, M);
          cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, M, N, &one, Q[j], N, Aij, M);
          
          for (long long y = childi; y < cendi; y++) {
            long long offset_y = std::reduce(&lowerA.DimsLr[childi], &lowerA.DimsLr[y]);
            for (long long x = childj; x < cendj; x++) {
              long long offset_x = std::reduce(&lowerA.DimsLr[childj], &lowerA.DimsLr[x]);
              long long lowN = lookupIJ(lowerA.ARows, lowerA.ACols, y - pbegin, x);
              long long lowC = lookupIJ(lowerA.CRows, lowerA.CCols, y - pbegin, x);
              Aij = &Adata[Aoffsets[ij] + offset_y + offset_x * M];

              if (0 <= lowN) {
                lowerA.NXT[lowN] = Aij;
                lowerA.Nstride[lowN] = M;
              }
              if (0 <= lowC) {
                lowerA.C[lowC] = Aij;
                lowerA.Cstride[lowC] = M;
              }
            }
          }
        }
      }
    }

    for (long long i = 0; i < nodes; i++) {
      long long fsize = wsa.fbodies_size_at_i(i);
      const double* fbodies = wsa.fbodies_at_i(i);
      std::vector<double> cbodies;

      if (use_near_bodies) {
        std::vector<long long>::iterator neighbors = ACols.begin() + ARows[i];
        std::vector<long long>::iterator neighbors_end = ACols.begin() + ARows[i + 1];
        long long csize = std::transform_reduce(neighbors, neighbors_end, -Dims[i + ibegin], std::plus<long long>(), [&](long long col) { return Dims[col]; });

        cbodies = std::vector<double>(3 * (fsize + csize));
        long long loc = 0;
        for (long long n = 0; n < (ARows[i + 1] - ARows[i]); n++) {
          long long col = neighbors[n];
          long long len = 3 * Dims[col];
          if (col != (i + ibegin)) {
            std::copy(S[col], S[col] + len, cbodies.begin() + loc);
            loc += len;
          }
        }
        std::copy(fbodies, &fbodies[3 * fsize], cbodies.begin() + loc);

        fsize += csize;
        fbodies = cbodies.data();
      }
      
      long long rank = compute_basis(eval, epi, Dims[i + ibegin], fsize, S[i + ibegin], fbodies, &Qdata[Qoffsets[i + ibegin]], R[i + ibegin]);
      DimsLr[i + ibegin] = rank;
    }

    comm.neighbor_bcast(DimsLr.data(), ones.data());
    comm.neighbor_bcast(Sdata.data(), Ssizes.data());
    comm.neighbor_bcast(Qdata.data(), elementsOnRow.data());
    comm.neighbor_bcast(Rdata.data(), elementsOnRow.data());
  }
}

H2MatrixSolver::H2MatrixSolver(const H2Matrix A[], const Cell cells[], const ColCommMPI comm[], long long levels) :
  levels(levels), offsets(levels + 1), upperIndex(levels + 1), upperOffsets(levels + 1), A(A), Comm(comm) {
  
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

void H2MatrixSolver::matVecMul(std::complex<double> X[]) const {
  typedef Eigen::Map<Eigen::VectorXcd> Vector_t;
  typedef Eigen::Stride<Eigen::Dynamic, 1> Stride_t;
  typedef Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Stride_t> Matrix_t;

  long long lbegin = Comm[levels].oLocal();
  long long llen = Comm[levels].lenLocal();
  long long lenX = offsets[levels][lbegin + llen] - offsets[levels][lbegin];

  std::vector<std::vector<std::complex<double>>> rhsX(levels + 1);
  std::vector<std::vector<std::complex<double>>> rhsY(levels + 1);

  for (long long l = levels; l >= 0; l--) {
    long long xlen = Comm[l].lenNeighbors();
    rhsX[l] = std::vector<std::complex<double>>(offsets[l][xlen], std::complex<double>(0., 0.));
    rhsY[l] = std::vector<std::complex<double>>(offsets[l][xlen], std::complex<double>(0., 0.));
  }

  Vector_t X_in(X, lenX);
  Vector_t X_leaf(rhsX[levels].data() + offsets[levels][lbegin], lenX);
  Vector_t Y_leaf(rhsY[levels].data() + offsets[levels][lbegin], lenX);

  if (X)
    X_leaf = X_in;

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

        if (0 < N) {
          Vector_t X(rhsX[l].data() + offsets[l][ibegin + y], M);
          Vector_t Xo(rhsX[l - 1].data() + offsets[l - 1][U] + upperOffsets[l][ibegin + y], N);
          Matrix_t Q(A[l].Q[y + ibegin], M, N, Stride_t(M, 1));
          Xo = Q.transpose() * X;
        }
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
        Vector_t Y(rhsY[l].data() + offsets[l][ibegin + y], M);
        Vector_t Yo(rhsY[l - 1].data() + offsets[l - 1][UY] + upperOffsets[l][ibegin + y], K);

        for (long long yx = A[l].CRows[y]; yx < A[l].CRows[y + 1]; yx++) {
          long long x = A[l].CCols[yx];
          long long N = A[l].DimsLr[x];
          long long UX = upperIndex[l][x];

          Vector_t Xo(rhsX[l - 1].data() + offsets[l - 1][UX] + upperOffsets[l][x], N);
          Matrix_t C(A[l].C[yx], K, N, Stride_t(A[l].Cstride[yx], 1));
          Yo += C * Xo;
        }
        Matrix_t Q(A[l].Q[y + ibegin], M, K, Stride_t(M, 1));
        Y = Q * Yo;
      }
    }
  }

  for (long long y = 0; y < llen; y++) {
    long long M = A[levels].Dims[lbegin + y];
    Vector_t Y(rhsY[levels].data() + offsets[levels][lbegin + y], M);

    if (0 < M)
      for (long long yx = A[levels].ARows[y]; yx < A[levels].ARows[y + 1]; yx++) {
        long long x = A[levels].ACols[yx];
        long long N = A[levels].Dims[x];

        Vector_t X(rhsX[levels].data() + offsets[levels][x], N);
        Matrix_t C(A[levels].A[yx], M, N, Stride_t(M, 1));
        Y += C * X;
      }
  }
  if (X)
    X_in = Y_leaf;
}

void H2MatrixSolver::solvePrecondition(std::complex<double>[]) const {
  // Default preconditioner = I
}

std::pair<double, long long> H2MatrixSolver::solveGMRES(double tol, std::complex<double> x[], const std::complex<double> b[], long long inner_iters, long long outer_iters) const {
  using Eigen::VectorXcd, Eigen::MatrixXcd;

  long long lbegin = Comm[levels].oLocal();
  long long llen = Comm[levels].lenLocal();
  long long N = offsets[levels][lbegin + llen] - offsets[levels][lbegin];
  long long ld = inner_iters + 1;

  Eigen::Map<const Eigen::VectorXcd> B(b, N);
  Eigen::Map<Eigen::VectorXcd> X(x, N);
  VectorXcd R = B;
  solvePrecondition(R.data());

  std::complex<double> normr = R.adjoint() * R;
  Comm[levels].level_sum(&normr, 1);
  double normb = std::sqrt(normr.real());
  if (normb == 0.)
    normb = 1.;

  for (long long j = 0; j < outer_iters; j++) {
    R = -X;
    matVecMul(R.data());
    R += B;
    solvePrecondition(R.data());

    normr = R.adjoint() * R;
    Comm[levels].level_sum(&normr, 1);
    double beta = std::sqrt(normr.real());
    double resid = beta / normb;
    if (resid < tol)
      return std::make_pair(resid, j);

    MatrixXcd H = MatrixXcd::Zero(ld, inner_iters);
    MatrixXcd v = MatrixXcd::Zero(N, ld);
    v.col(0) = R * (1. / beta);
    
    for (long long i = 0; i < inner_iters; i++) {
      VectorXcd w = v.col(i);
      matVecMul(w.data());
      solvePrecondition(w.data());

      for (long long k = 0; k <= i; k++)
        H(k, i) = v.col(k).adjoint() * w;
      Comm[levels].level_sum(H.col(i).data(), i + 1);

      for (long long k = 0; k <= i; k++)
        w -= H(k, i) * v.col(k);

      std::complex<double> normw = w.adjoint() * w;
      Comm[levels].level_sum(&normw, 1);
      H(i + 1, i) = std::sqrt(normw.real());
      v.col(i + 1) = w * (1. / H(i + 1, i));
    }

    VectorXcd s = VectorXcd::Zero(ld);
    s(0) = beta;

    VectorXcd y = H.colPivHouseholderQr().solve(s);
    X += v.leftCols(inner_iters) * y;
  }

  R = -X;
  matVecMul(R.data());
  R += B;
  solvePrecondition(R.data());

  normr = R.adjoint() * R;
  Comm[levels].level_sum(&normr, 1);
  double beta = std::sqrt(normr.real());
  double resid = beta / normb;
  return std::make_pair(resid, outer_iters);
}

