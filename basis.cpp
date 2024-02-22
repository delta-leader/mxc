#include <basis.hpp>
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
  std::vector<std::complex<double>> B(M * N), TAU(M);
  std::vector<long long> jpiv(M, 0);
  std::complex<double> one(1., 0.), zero(0., 0.);

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

  if (rank == M)
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'F', M, M, zero, one, A, M);
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
  return rank;
}

ClusterBasis::ClusterBasis(const MatrixAccessor& eval, double epi, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& wsa, const CellComm& comm, const ClusterBasis& prev_basis, const CellComm& prev_comm) {
  long long xlen = comm.lenNeighbors();
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long ybegin = comm.oGlobal();

  localChildIndex = prev_comm.iLocal(cells[ybegin].Child[0]);
  localChildOffsets = std::vector<long long>(nodes + 1);
  localChildOffsets[0] = 0;
  std::transform(&cells[ybegin], &cells[ybegin + nodes], localChildOffsets.begin() + 1, [&](const Cell& c) { return c.Child[1] - cells[ybegin].Child[0]; });

  localChildLrDims = std::vector<long long>(localChildOffsets.back());
  std::copy(&prev_basis.DimsLr[localChildIndex], &prev_basis.DimsLr[localChildIndex + localChildLrDims.size()], localChildLrDims.begin());

  Dims = std::vector<long long>(xlen, 0);
  DimsLr = std::vector<long long>(xlen, 0);
  ParentSequenceNum = std::vector<long long>(xlen);
  elementsOnRow = std::vector<long long>(xlen);
  S = std::vector<const double*>(xlen);
  Q = std::vector<const std::complex<double>*>(xlen);
  R = std::vector<std::complex<double>*>(xlen);

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
  comm.dup_bcast(Dims.data(), xlen);
  comm.dup_bcast(ParentSequenceNum.data(), xlen);

  std::vector<long long> Qoffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), elementsOnRow.begin(), [](const long long d) { return d * d; });
  std::inclusive_scan(elementsOnRow.begin(), elementsOnRow.end(), Qoffsets.begin() + 1);
  Qoffsets[0] = 0;
  Qdata = std::vector<std::complex<double>>(Qoffsets[xlen], std::complex<double>(0., 0.));
  Rdata = std::vector<std::complex<double>>(Qoffsets[xlen], std::complex<double>(0., 0.));
  std::transform(Qoffsets.begin(), Qoffsets.end(), Q.begin(), [&](const long long d) { return &Qdata[d]; });
  std::transform(Qoffsets.begin(), Qoffsets.end(), R.begin(), [&](const long long d) { return &Rdata[d]; });

  std::vector<long long> Ssizes(xlen), Soffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), Ssizes.begin(), [](const long long d) { return 3 * d; });
  std::inclusive_scan(Ssizes.begin(), Ssizes.end(), Soffsets.begin() + 1);
  Soffsets[0] = 0;
  Sdata = std::vector<double>(Soffsets[xlen], 0.);
  std::transform(Soffsets.begin(), Soffsets.end(), S.begin(), [&](const long long d) { return &Sdata[d]; });

  for (long long i = 0; i < nodes; i++) {
    long long dim = Dims[i + ibegin];
    std::complex<double>* matrix = &Qdata[Qoffsets[i + ibegin]];
    double* ske = &Sdata[Soffsets[i + ibegin]];

    long long ci = i + ybegin;
    long long childi = localChildIndex + localChildOffsets[i];
    long long cend = localChildIndex + localChildOffsets[i + 1];

    if (cend <= childi)
      std::copy(&bodies[3 * cells[ci].Body[0]], &bodies[3 * cells[ci].Body[1]], ske);
    for (long long j = childi; j < cend; j++) {
      long long offset = prev_basis.copyOffset(j);
      long long len = prev_basis.DimsLr[j];
      std::copy(prev_basis.S[j], prev_basis.S[j] + (len * 3), &ske[offset * 3]);
    }

    long long fsize = wsa.fbodies_size_at_i(i);
    const double* fbodies = wsa.fbodies_at_i(i);
    long long rank = (dim > 0 && fsize > 0) ? compute_basis(eval, epi, dim, fsize, ske, fbodies, matrix) : 0;
    DimsLr[i + ibegin] = rank;
  }

  comm.neighbor_bcast(DimsLr.data(), ones.data());
  comm.neighbor_bcast(Sdata.data(), Ssizes.data());
  comm.neighbor_bcast(Qdata.data(), elementsOnRow.data());
  comm.dup_bcast(DimsLr.data(), xlen);
  comm.dup_bcast(Sdata.data(), Soffsets[xlen]);
  comm.dup_bcast(Qdata.data(), Qoffsets[xlen]);

  CRows = std::vector<long long>(&Far.RowIndex[ybegin], &Far.RowIndex[ybegin + nodes + 1]);
  CCols = std::vector<long long>(&Far.ColIndex[CRows[0]], &Far.ColIndex[CRows[nodes]]);
  long long offset = CRows[0];
  std::for_each(CRows.begin(), CRows.end(), [=](long long& i) { i = i - offset; });

  std::vector<long long> Csizes(CRows[nodes]), Coffsets(CRows[nodes] + 1);
  for (long long i = 0; i < nodes; i++)
    std::transform(&CCols[CRows[i]], &CCols[CRows[i + 1]], &Csizes[CRows[i]], 
      [&](long long col) { return DimsLr[i + ibegin] * DimsLr[comm.iLocal(col)]; });
  std::inclusive_scan(Csizes.begin(), Csizes.end(), Coffsets.begin() + 1);
  Coffsets[0] = 0;

  C = std::vector<const std::complex<double>*>(CRows[nodes]);
  Cdata = std::vector<std::complex<double>>(Coffsets.back());
  std::transform(Coffsets.begin(), Coffsets.begin() + CRows[nodes], C.begin(), [&](const long long d) { return &Cdata[d]; });

  for (long long i = 0; i < nodes; i++)
    for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
      long long j = comm.iLocal(CCols[ij]);
      long long m = DimsLr[i + ibegin], n = DimsLr[j];
      gen_matrix(eval, m, n, S[i + ibegin], S[j], &Cdata[Coffsets[ij]]);
    }
}

long long ClusterBasis::copyOffset(long long i) const {
  long long start = std::max((long long)0, i - ParentSequenceNum[i]);
  return std::reduce(&DimsLr[start], &DimsLr[i], (long long)0);
}

long long compute_recompression(double epi, long long DimQ, long long RankQ, std::complex<double> Q[], std::complex<double> R[]) {
  if (0 < RankQ && RankQ < DimQ) {
    long long M = DimQ + RankQ;
    std::vector<std::complex<double>> A(M * DimQ);
    std::vector<double> S(DimQ * 2);
    std::complex<double> one(1., 0.), zero(0., 0.);

    MKL_Zomatcopy('C', 'N', DimQ, RankQ, one, Q, DimQ, &A[0], DimQ);
    MKL_Zomatcopy('C', 'T', DimQ, DimQ, one, R, DimQ, &A[DimQ * RankQ], DimQ);
    double nrmQ = cblas_dznrm2(DimQ * RankQ, &A[0], 1);
    double nrmR = cblas_dznrm2(DimQ * DimQ, &A[DimQ * RankQ], 1);
    if (std::numeric_limits<double>::min() < nrmQ && nrmQ < nrmR) {
      double scaleQ = nrmR / nrmQ;
      cblas_zscal(DimQ * RankQ, &scaleQ, &A[0], 1);
    }
    else if (std::numeric_limits<double>::min() < nrmR && nrmR < nrmQ) {
      double scaleR = nrmQ / nrmR;
      cblas_zscal(DimQ * DimQ, &scaleR, &A[DimQ * RankQ], 1);
    }

    LAPACKE_zgesvd(LAPACK_COL_MAJOR, 'O', 'N', DimQ, M, &A[0], DimQ, &S[0], &A[0], DimQ, nullptr, M, &S[DimQ]);
    long long rank = 0;
    double s0 = epi * S[0];
    if (std::numeric_limits<double>::min() < s0)
      while (rank < DimQ && s0 <= S[rank])
        ++rank;
        
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'F', DimQ, DimQ, zero, zero, R, DimQ);
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, rank, RankQ, DimQ, &one, &A[0], DimQ, Q, DimQ, &zero, R, DimQ);
    MKL_Zomatcopy('C', 'N', DimQ, DimQ, one, &A[0], DimQ, Q, DimQ);
    return rank;
  }
  else if (RankQ == DimQ) {
    std::vector<std::complex<double>> TAU(DimQ);
    std::complex<double> zero(0., 0.);
    LAPACKE_zgeqrf(LAPACK_COL_MAJOR, DimQ, DimQ, Q, DimQ, &TAU[0]);
    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'U', DimQ, DimQ, Q, DimQ, R, DimQ);
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', DimQ - 1, DimQ - 1, zero, zero, &R[1], DimQ);
    LAPACKE_zungqr(LAPACK_COL_MAJOR, DimQ, DimQ, DimQ, Q, DimQ, &TAU[0]);
  }
  return RankQ;
}

void ClusterBasis::recompressR(double epi, const CellComm& comm) {
  long long xlen = comm.lenNeighbors();
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();

  std::vector<std::vector<std::complex<double>>> CdataOld(CRows[nodes]);
  std::vector<long long> DimsLrOld(DimsLr.begin(), DimsLr.end());
  for (long long i = 0; i < CRows[nodes] - 1; i++)
    CdataOld[i] = std::vector<std::complex<double>>(C[i], C[i + 1]);
  if (0 < CRows[nodes])
    CdataOld[CRows[nodes] - 1] = std::vector<std::complex<double>>(C[CRows[nodes] - 1], const_cast<const std::complex<double>*>(&Cdata[Cdata.size()]));

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin], N = DimsLr[i + ibegin];
    std::vector<std::complex<double>> c(N * N);
    std::complex<double>* Qptr = const_cast<std::complex<double>*>(Q[i + ibegin]);
    long long rank = compute_recompression(epi, M, N, Qptr, R[i + ibegin]);
    DimsLr[i + ibegin] = rank;
  }

  const std::vector<long long> ones(xlen, 1);
  long long lenQ = std::reduce(elementsOnRow.begin(), elementsOnRow.end());
  comm.neighbor_bcast(DimsLr.data(), ones.data());
  comm.neighbor_bcast(Qdata.data(), elementsOnRow.data());
  comm.neighbor_bcast(Rdata.data(), elementsOnRow.data());
  comm.dup_bcast(DimsLr.data(), xlen);
  comm.dup_bcast(Qdata.data(), lenQ);
  comm.dup_bcast(Rdata.data(), lenQ);

  std::vector<long long> Csizes(CRows[nodes]), Coffsets(CRows[nodes] + 1);
  for (long long i = 0; i < nodes; i++)
    std::transform(&CCols[CRows[i]], &CCols[CRows[i + 1]], &Csizes[CRows[i]], 
      [&](long long col) { return DimsLr[i + ibegin] * DimsLr[comm.iLocal(col)]; });
  std::inclusive_scan(Csizes.begin(), Csizes.end(), Coffsets.begin() + 1);
  Coffsets[0] = 0;

  Cdata.resize(Coffsets.back());
  std::fill(Cdata.begin(), Cdata.end(), std::complex<double>(0., 0.));
  std::transform(Coffsets.begin(), Coffsets.begin() + CRows[nodes], C.begin(), [&](const long long d) { return &Cdata[d]; });

  long long dim_max = *std::max_element(DimsLr.begin(), DimsLr.end());
  std::vector<std::complex<double>> Tmp(dim_max * dim_max);
  std::complex<double> one(1., 0.), zero(0., 0.);

  for (long long i = 0; i < nodes; i++) {
    long long m1 = DimsLrOld[i + ibegin];
    long long m2 = DimsLr[i + ibegin];
    long long ldl = Dims[i + ibegin];
    for (long long ij = CRows[i]; ij < CRows[i + 1] && 0 < m1; ij++) {
      long long j = comm.iLocal(CCols[ij]);
      long long n1 = DimsLrOld[j];
      long long n2 = DimsLr[j];
      long long ldr = Dims[j];
      cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, m1, n2, n1, &one, CdataOld[ij].data(), m1, R[j], ldr, &zero, &Tmp[0], m1);
      cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m2, n2, m1, &one, R[i + ibegin], ldl, &Tmp[0], m1, &zero, &Cdata[Coffsets[ij]], m2);
    }
  }
}

void ClusterBasis::adjustLowerRankGrowth(const ClusterBasis& prev_basis, const CellComm& comm) {
  long long xlen = comm.lenNeighbors();
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();

  std::vector<long long> oldDims(nodes);
  std::vector<std::vector<std::complex<double>>> oldQ(nodes);
  std::vector<long long> newLocalChildLrDims(localChildLrDims.size());
  std::copy(&prev_basis.DimsLr[localChildIndex], &prev_basis.DimsLr[localChildIndex + localChildLrDims.size()], newLocalChildLrDims.begin());

  for (long long i = 0; i < nodes; i++) {
    oldDims[i] = Dims[i + ibegin];
    oldQ[i] = std::vector<std::complex<double>>(Q[i + ibegin], &(Q[i + ibegin])[oldDims[i] * DimsLr[i + ibegin]]);
    Dims[i + ibegin] = std::reduce(&newLocalChildLrDims[localChildOffsets[i]], &newLocalChildLrDims[localChildOffsets[i + 1]]);
  }

  const std::vector<long long> ones(xlen, 1);
  comm.neighbor_bcast(Dims.data(), ones.data());
  comm.dup_bcast(Dims.data(), xlen);

  std::vector<long long> Qoffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), elementsOnRow.begin(), [](const long long d) { return d * d; });
  std::inclusive_scan(elementsOnRow.begin(), elementsOnRow.end(), Qoffsets.begin() + 1);
  Qoffsets[0] = 0;
  Qdata = std::vector<std::complex<double>>(Qoffsets[xlen], std::complex<double>(0., 0.));
  Rdata = std::vector<std::complex<double>>(Qoffsets[xlen], std::complex<double>(0., 0.));
  std::transform(Qoffsets.begin(), Qoffsets.end(), Q.begin(), [&](const long long d) { return &Qdata[d]; });
  std::transform(Qoffsets.begin(), Qoffsets.end(), R.begin(), [&](const long long d) { return &Rdata[d]; });

  std::complex<double> one(1., 0.), zero(0., 0.);
  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    long long N = DimsLr[i + ibegin];
    long long childi = localChildOffsets[i];
    long long cend = localChildOffsets[i + 1];
    for (long long j = childi; j < cend && 0 < N; j++) {
      long long m1 = localChildLrDims[j], m2 = newLocalChildLrDims[j];
      long long lj = localChildIndex + j;
      long long offsetOld = std::reduce(&localChildLrDims[childi], &localChildLrDims[j]);
      long long offsetNew = prev_basis.copyOffset(lj);
      cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m2, N, m1, &one, 
        prev_basis.R[lj], prev_basis.Dims[lj], &(oldQ[i])[offsetOld], oldDims[i], &zero, &Qdata[Qoffsets[i + ibegin] + offsetNew], M);
    }
  }

  std::copy(newLocalChildLrDims.begin(), newLocalChildLrDims.end(), localChildLrDims.begin());
  comm.neighbor_bcast(Qdata.data(), elementsOnRow.data());
  comm.dup_bcast(Qdata.data(), Qoffsets[xlen]);
}

MatVec::MatVec(const MatrixAccessor& eval, const ClusterBasis basis[], const double bodies[], const Cell cells[], const CSR& near, const CellComm comm[], long long levels) :
  EvalFunc(&eval), Basis(basis), Bodies(bodies), Cells(cells), Near(&near), Comm(comm), Levels(levels) {
}

void MatVec::operator() (long long nrhs, std::complex<double> X[]) const {
  long long lbegin = Comm[Levels].oLocal();
  long long llen = Comm[Levels].lenLocal();

  std::vector<std::vector<std::complex<double>>> rhsX(Levels + 1), rhsY(Levels + 1);
  std::vector<std::vector<std::complex<double>*>> rhsXptr(Levels + 1), rhsYptr(Levels + 1);
  std::vector<std::vector<std::pair<std::complex<double>*, long long>>> rhsXoptr(Levels + 1), rhsYoptr(Levels + 1);

  for (long long l = Levels; l >= 0; l--) {
    long long xlen = Comm[l].lenNeighbors();
    std::vector<long long> offsets(xlen + 1, 0);
    std::inclusive_scan(Basis[l].Dims.begin(), Basis[l].Dims.end(), offsets.begin() + 1);

    rhsX[l] = std::vector<std::complex<double>>(offsets[xlen] * nrhs, std::complex<double>(0., 0.));
    rhsY[l] = std::vector<std::complex<double>>(offsets[xlen] * nrhs, std::complex<double>(0., 0.));
    rhsXptr[l] = std::vector<std::complex<double>*>(xlen, nullptr);
    rhsYptr[l] = std::vector<std::complex<double>*>(xlen, nullptr);
    rhsXoptr[l] = std::vector<std::pair<std::complex<double>*, long long>>(xlen, std::make_pair(nullptr, 0));
    rhsYoptr[l] = std::vector<std::pair<std::complex<double>*, long long>>(xlen, std::make_pair(nullptr, 0));

    std::transform(offsets.begin(), offsets.begin() + xlen, rhsXptr[l].begin(), [&](const long long d) { return &rhsX[l][0] + d * nrhs; });
    std::transform(offsets.begin(), offsets.begin() + xlen, rhsYptr[l].begin(), [&](const long long d) { return &rhsY[l][0] + d * nrhs; });

    if (l < Levels)
      for (long long i = 0; i < xlen; i++) {
        long long ci = Comm[l].iGlobal(i);
        long long child = Comm[l + 1].iLocal(Cells[ci].Child[0]);
        long long clen = Cells[ci].Child[1] - Cells[ci].Child[0];

        if (child >= 0 && clen > 0) {
          std::vector<long long> offsets_child(clen + 1, 0);
          std::inclusive_scan(&Basis[l + 1].DimsLr[child], &Basis[l + 1].DimsLr[child + clen], offsets_child.begin() + 1);
          long long ldi = Basis[l].Dims[i];
          std::transform(offsets_child.begin(), offsets_child.begin() + clen, &rhsXoptr[l + 1][child], 
            [&](const long long d) { return std::make_pair(rhsXptr[l][i] + d, ldi); });
          std::transform(offsets_child.begin(), offsets_child.begin() + clen, &rhsYoptr[l + 1][child], 
            [&](const long long d) { return std::make_pair(rhsYptr[l][i] + d, ldi); });
        }
      }
  }

  long long Y = 0, lenX = std::reduce(&Basis[Levels].Dims[lbegin], &Basis[Levels].Dims[lbegin + llen]);
  for (long long i = 0; i < llen; i++) {
    long long M = Basis[Levels].Dims[lbegin + i];
    MKL_Zomatcopy('C', 'N', M, nrhs, std::complex<double>(1., 0.), &X[Y], lenX, rhsXptr[Levels][lbegin + i], M);
    Y = Y + M;
  }

  const std::complex<double> one(1., 0.), zero(0., 0.);
  for (long long i = Levels; i > 0; i--) {
    long long ibegin = Comm[i].oLocal();
    long long iboxes = Comm[i].lenLocal();
    long long xlen = Comm[i].lenNeighbors();

    std::vector<long long> lens(xlen);
    std::transform(Basis[i].Dims.begin(), Basis[i].Dims.end(), lens.begin(), [=](const long long& i) { return i * nrhs; });
    long long lenI = nrhs * std::reduce(&Basis[i].Dims[0], &Basis[i].Dims[xlen]);
    Comm[i].level_merge(rhsX[i].data(), lenI);
    Comm[i].neighbor_bcast(rhsX[i].data(), lens.data());
    Comm[i].dup_bcast(rhsX[i].data(), lenI);

    for (long long y = 0; y < iboxes; y++) {
      long long M = Basis[i].Dims[y + ibegin];
      long long N = Basis[i].DimsLr[y + ibegin];
      if (0 < N)
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, N, nrhs, M, &one, Basis[i].Q[y + ibegin], M, 
          rhsXptr[i][y + ibegin], M, &zero, rhsXoptr[i][y + ibegin].first, rhsXoptr[i][y + ibegin].second);
    }
  }

  if (Basis[0].Dims[0] > 0) {
    Comm[0].level_merge(rhsX[0].data(), Basis[0].Dims[0] * nrhs);
    Comm[0].dup_bcast(rhsX[0].data(), Basis[0].Dims[0] * nrhs);
  }

  for (long long i = 1; i <= Levels; i++) {
    long long ibegin = Comm[i].oLocal();
    long long iboxes = Comm[i].lenLocal();

    for (long long y = 0; y < iboxes; y++) {
      long long M = Basis[i].Dims[y + ibegin];
      long long K = Basis[i].DimsLr[y + ibegin];

      if (0 < K) {
        for (long long yx = Basis[i].CRows[y]; yx < Basis[i].CRows[y + 1]; yx++) {
          long long x = Comm[i].iLocal(Basis[i].CCols[yx]);
          long long N = Basis[i].DimsLr[x];
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, K, nrhs, N, &one, Basis[i].C[yx], K, 
              rhsXoptr[i][x].first, rhsXoptr[i][x].second, &one, rhsYoptr[i][y + ibegin].first, rhsYoptr[i][y + ibegin].second);
        }
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, nrhs, K, &one, Basis[i].Q[y + ibegin], M, 
          rhsYoptr[i][y + ibegin].first, rhsYoptr[i][y + ibegin].second, &zero, rhsYptr[i][y + ibegin], M);
      }
    }
  }

  long long gbegin = Comm[Levels].oGlobal();
  for (long long y = 0; y < llen; y++)
    for (long long yx = Near->RowIndex[y + gbegin]; yx < Near->RowIndex[y + gbegin + 1]; yx++) {
      long long x = Near->ColIndex[yx];
      long long x_loc = Comm[Levels].iLocal(x);
      long long M = Cells[y + gbegin].Body[1] - Cells[y + gbegin].Body[0];
      long long N = Cells[x].Body[1] - Cells[x].Body[0];
      mat_vec_reference(*EvalFunc, M, N, nrhs, rhsYptr[Levels][y + lbegin], rhsXptr[Levels][x_loc], &Bodies[3 * Cells[y + gbegin].Body[0]], &Bodies[3 * Cells[x].Body[0]]);
    }
  Y = 0;
  for (long long i = 0; i < llen; i++) {
    long long M = Basis[Levels].Dims[lbegin + i];
    MKL_Zomatcopy('C', 'N', M, nrhs, std::complex<double>(1., 0.), rhsYptr[Levels][lbegin + i], M, &X[Y], lenX);
    Y = Y + M;
  }
}
