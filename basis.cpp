#include <basis.hpp>
#include <build_tree.hpp>
#include <comm.hpp>
#include <kernel.hpp>

#include <mkl.h>
#include <algorithm>
#include <numeric>
#include <cmath>

WellSeparatedApproximation::WellSeparatedApproximation(const MatrixAccessor& eval, double epi, int64_t rank, int64_t lbegin, int64_t len, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& upper) :
  lbegin(lbegin), lend(lbegin + len), M(len) {
  std::vector<std::vector<double>> Fbodies(len);
  for (int64_t i = upper.lbegin; i < upper.lend; i++)
    for (int64_t c = cells[i].Child[0]; c < cells[i].Child[1]; c++)
      if (lbegin <= c && c < lend)
        M[c - lbegin] = std::vector<double>(upper.M[i - upper.lbegin].begin(), upper.M[i - upper.lbegin].end());

  for (int64_t y = lbegin; y < lend; y++) {
    for (int64_t yx = Far.RowIndex[y]; yx < Far.RowIndex[y + 1]; yx++) {
      int64_t x = Far.ColIndex[yx];
      int64_t m = cells[y].Body[1] - cells[y].Body[0];
      int64_t n = cells[x].Body[1] - cells[x].Body[0];
      const double* Xbodies = &bodies[3 * cells[x].Body[0]];
      const double* Ybodies = &bodies[3 * cells[y].Body[0]];

      int64_t k = std::min(rank, std::min(m, n));
      std::vector<int64_t> ipiv(k);
      std::vector<std::complex<double>> U(n * k);
      int64_t iters = interpolative_decomp_aca(epi, eval, n, m, k, Xbodies, Ybodies, &ipiv[0], &U[0], n);
      std::vector<double> Fbodies(3 * iters);
      for (int64_t i = 0; i < iters; i++)
        std::copy(&Xbodies[3 * ipiv[i]], &Xbodies[3 * (ipiv[i] + 1)], &Fbodies[3 * i]);
      M[y - lbegin].insert(M[y - lbegin].end(), Fbodies.begin(), Fbodies.end());
    }
  }
}

int64_t WellSeparatedApproximation::fbodies_size_at_i(int64_t i) const {
  return 0 <= i && i < (int64_t)M.size() ? M[i].size() / 3 : 0;
}

const double* WellSeparatedApproximation::fbodies_at_i(int64_t i) const {
  return 0 <= i && i < (int64_t)M.size() ? M[i].data() : nullptr;
}

int64_t compute_basis(const MatrixAccessor& eval, double epi, int64_t M, int64_t N, double Xbodies[], const double Fbodies[], std::complex<double> A[], int64_t LDA) {
  int64_t K = std::max(M, N);
  std::vector<std::complex<double>> B(M * K), TAU(M);
  std::vector<MKL_INT> ipiv(M), jpiv(M, 0);
  std::complex<double> one(1., 0.), zero(0., 0.);

  gen_matrix(eval, N, M, Fbodies, Xbodies, &B[0], K);
  LAPACKE_zgeqrf(LAPACK_COL_MAJOR, N, M, &B[0], K, &TAU[0]);
  LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, zero, zero, &B[1], K);
  LAPACKE_zgeqp3(LAPACK_COL_MAJOR, M, M, &B[0], K, &jpiv[0], &TAU[0]);
  int64_t rank = 0;
  double s0 = epi * std::abs(B[0]);
  if (std::numeric_limits<double>::min() < s0)
    while (rank < M && s0 <= std::abs(B[rank * (K + 1)]))
      ++rank;

  if (rank == M)
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'F', M, M, zero, one, A, LDA);
  if (0 < rank && rank < M) {
    cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, rank, M - rank, &one, &B[0], K, &B[rank * K], K);
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'F', rank, rank, zero, one, &B[0], K);
    MKL_Zomatcopy('C', 'T', rank, M, one, &B[0], K, A, LDA);

    for (int64_t i = 0; i < M; i++) {
      int64_t piv = std::distance(&jpiv[0], std::find(&jpiv[i], &jpiv[M], i + 1));
      ipiv[i] = piv + 1;
      if (piv != i)
        std::iter_swap(&jpiv[i], &jpiv[piv]);
    }
    LAPACKE_zlaswp(LAPACK_COL_MAJOR, rank, A, LDA, 1, M, &ipiv[0], 1);
    LAPACKE_dlaswp(LAPACK_ROW_MAJOR, 3, Xbodies, 3, 1, M, &ipiv[0], -1);
  }
  return rank;
}

ClusterBasis::ClusterBasis(const MatrixAccessor& eval, double epi, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& wsa, const CellComm& comm, const ClusterBasis& prev_basis, const CellComm& prev_comm) {
  int64_t xlen = comm.lenNeighbors();
  int64_t ibegin = comm.oLocal();
  int64_t nodes = comm.lenLocal();
  int64_t ybegin = comm.oGlobal();

  localChildOffsets = std::vector<int64_t>(nodes + 1);
  localChildLrDims = std::vector<int64_t>(cells[ybegin + nodes - 1].Child[1] - cells[ybegin].Child[0]);
  localChildIndex = prev_comm.iLocal(cells[ybegin].Child[0]);
  std::transform(&cells[ybegin], &cells[ybegin + nodes], localChildOffsets.begin() + 1, [&](const Cell& c) { return c.Child[1] - cells[ybegin].Child[0]; });
  std::copy(&prev_basis.DimsLr[localChildIndex], &prev_basis.DimsLr[localChildIndex + localChildLrDims.size()], localChildLrDims.begin());
  localChildOffsets[0] = 0;

  Dims = std::vector<int64_t>(xlen, 0);
  DimsLr = std::vector<int64_t>(xlen, 0);
  elementsOnRow = std::vector<int64_t>(xlen);
  S = std::vector<const double*>(xlen);
  Q = std::vector<const std::complex<double>*>(xlen);
  R = std::vector<std::complex<double>*>(xlen);

  for (int64_t i = 0; i < nodes; i++)
    Dims[i + ibegin] = localChildOffsets[i] == localChildOffsets[i + 1] ? (cells[i + ybegin].Body[1] - cells[i + ybegin].Body[0]) :
      std::reduce(&localChildLrDims[localChildOffsets[i]], &localChildLrDims[localChildOffsets[i + 1]]);

  const std::vector<int64_t> ones(xlen, 1);
  comm.neighbor_bcast(Dims.data(), ones.data());
  comm.dup_bcast(Dims.data(), xlen);

  std::vector<int64_t> Qoffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), elementsOnRow.begin(), [](const int64_t d) { return d * d; });
  std::inclusive_scan(elementsOnRow.begin(), elementsOnRow.end(), Qoffsets.begin() + 1);
  Qoffsets[0] = 0;
  Qdata = std::vector<std::complex<double>>(Qoffsets[xlen], std::complex<double>(0., 0.));
  Rdata = std::vector<std::complex<double>>(Qoffsets[xlen], std::complex<double>(0., 0.));
  std::transform(Qoffsets.begin(), Qoffsets.end(), Q.begin(), [&](const int64_t d) { return &Qdata[d]; });
  std::transform(Qoffsets.begin(), Qoffsets.end(), R.begin(), [&](const int64_t d) { return &Rdata[d]; });

  std::vector<int64_t> Ssizes(xlen), Soffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), Ssizes.begin(), [](const int64_t d) { return 3 * d; });
  std::inclusive_scan(Ssizes.begin(), Ssizes.end(), Soffsets.begin() + 1);
  Soffsets[0] = 0;
  Sdata = std::vector<double>(Soffsets[xlen], 0.);
  std::transform(Soffsets.begin(), Soffsets.end(), S.begin(), [&](const int64_t d) { return &Sdata[d]; });

  for (int64_t i = 0; i < nodes; i++) {
    int64_t dim = Dims[i + ibegin];
    std::complex<double>* matrix = &Qdata[Qoffsets[i + ibegin]];
    double* ske = &Sdata[Soffsets[i + ibegin]];

    int64_t ci = i + ybegin;
    int64_t childi = localChildIndex + localChildOffsets[i];
    int64_t cend = localChildIndex + localChildOffsets[i + 1];

    if (cend <= childi)
      std::copy(&bodies[3 * cells[ci].Body[0]], &bodies[3 * cells[ci].Body[1]], ske);
    for (int64_t j = childi; j < cend; j++) {
      int64_t offset = std::reduce(&prev_basis.DimsLr[childi], &prev_basis.DimsLr[j]);
      int64_t len = prev_basis.DimsLr[j];
      std::copy(prev_basis.S[j], prev_basis.S[j] + (len * 3), &ske[offset * 3]);
    }

    int64_t fsize = wsa.fbodies_size_at_i(i);
    const double* fbodies = wsa.fbodies_at_i(i);
    int64_t rank = (dim > 0 && fsize > 0) ? compute_basis(eval, epi, dim, fsize, ske, fbodies, matrix, dim) : 0;
    DimsLr[i + ibegin] = rank;
  }

  comm.neighbor_bcast(DimsLr.data(), ones.data());
  comm.neighbor_bcast(Sdata.data(), Ssizes.data());
  comm.neighbor_bcast(Qdata.data(), elementsOnRow.data());
  comm.dup_bcast(DimsLr.data(), xlen);
  comm.dup_bcast(Sdata.data(), Soffsets[xlen]);
  comm.dup_bcast(Qdata.data(), Qoffsets[xlen]);

  CRows = std::vector<int64_t>(&Far.RowIndex[ybegin], &Far.RowIndex[ybegin + nodes + 1]);
  CCols = std::vector<int64_t>(&Far.ColIndex[CRows[0]], &Far.ColIndex[CRows[nodes]]);
  int64_t offset = CRows[0];
  std::for_each(CRows.begin(), CRows.end(), [=](int64_t& i) { i = i - offset; });

  std::vector<int64_t> Csizes(CRows[nodes]), Coffsets(CRows[nodes] + 1);
  for (int64_t i = 0; i < nodes; i++)
    std::transform(&CCols[CRows[i]], &CCols[CRows[i + 1]], &Csizes[CRows[i]], 
      [&](int64_t col) { return DimsLr[i + ibegin] * DimsLr[comm.iLocal(col)]; });
  std::inclusive_scan(Csizes.begin(), Csizes.end(), Coffsets.begin() + 1);
  Coffsets[0] = 0;

  C = std::vector<const std::complex<double>*>(CRows[nodes]);
  Cdata = std::vector<std::complex<double>>(Coffsets.back());
  std::transform(Coffsets.begin(), Coffsets.begin() + CRows[nodes], C.begin(), [&](const int64_t d) { return &Cdata[d]; });

  for (int64_t i = 0; i < nodes; i++)
    for (int64_t ij = CRows[i]; ij < CRows[i + 1]; ij++) {
      int64_t j = comm.iLocal(CCols[ij]);
      int64_t m = DimsLr[i + ibegin], n = DimsLr[j];
      gen_matrix(eval, m, n, S[i + ibegin], S[j], &Cdata[Coffsets[ij]], m);
    }
}

int64_t compute_recompression(double epi, int64_t DimQ, int64_t RankQ, std::complex<double> Q[], int64_t LDQ, std::complex<double> R[], int64_t LDR) {
  if (0 < RankQ && RankQ < DimQ) {
    int64_t M = DimQ + RankQ;
    std::vector<std::complex<double>> A(M * DimQ);
    std::vector<double> S(DimQ * 2);
    std::complex<double> one(1., 0.), zero(0., 0.);

    MKL_Zomatcopy('C', 'N', DimQ, RankQ, one, Q, LDQ, &A[0], DimQ);
    MKL_Zomatcopy('C', 'T', DimQ, DimQ, one, R, LDR, &A[DimQ * RankQ], DimQ);
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
    int64_t rank = 0;
    double s0 = epi * S[0];
    if (std::numeric_limits<double>::min() < s0)
      while (rank < DimQ && s0 <= S[rank])
        ++rank;
        
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'F', DimQ, DimQ, zero, zero, R, LDR);
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, rank, RankQ, DimQ, &one, &A[0], DimQ, Q, LDQ, &zero, R, LDR);
    MKL_Zomatcopy('C', 'N', DimQ, DimQ, one, &A[0], DimQ, Q, LDQ);
    return rank;
  }
  else if (RankQ == DimQ) {
    std::vector<std::complex<double>> TAU(DimQ);
    std::complex<double> zero(0., 0.);
    LAPACKE_zgeqrf(LAPACK_COL_MAJOR, DimQ, DimQ, Q, LDQ, &TAU[0]);
    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'U', DimQ, DimQ, Q, LDQ, R, LDR);
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', DimQ - 1, DimQ - 1, zero, zero, &R[1], LDR);
    LAPACKE_zungqr(LAPACK_COL_MAJOR, DimQ, DimQ, DimQ, Q, LDQ, &TAU[0]);
  }
  return RankQ;
}

void ClusterBasis::recompressR(double epi, const CellComm& comm) {
  int64_t xlen = comm.lenNeighbors();
  int64_t ibegin = comm.oLocal();
  int64_t nodes = comm.lenLocal();

  std::vector<std::vector<std::complex<double>>> CdataOld(CRows[nodes]);
  std::vector<int64_t> DimsLrOld(DimsLr.begin(), DimsLr.end());
  for (int64_t i = 0; i < CRows[nodes] - 1; i++)
    CdataOld[i] = std::vector<std::complex<double>>(C[i], C[i + 1]);
  if (0 < CRows[nodes])
    CdataOld[CRows[nodes] - 1] = std::vector<std::complex<double>>(C[CRows[nodes] - 1], const_cast<const std::complex<double>*>(&Cdata[Cdata.size()]));

  for (int64_t i = 0; i < nodes; i++) {
    int64_t M = Dims[i + ibegin], N = DimsLr[i + ibegin];
    std::vector<std::complex<double>> c(N * N);
    std::complex<double>* Qptr = const_cast<std::complex<double>*>(Q[i + ibegin]);
    int64_t rank = compute_recompression(epi, M, N, Qptr, M, R[i + ibegin], M);
    DimsLr[i + ibegin] = rank;
  }

  const std::vector<int64_t> ones(xlen, 1);
  int64_t lenQ = std::reduce(elementsOnRow.begin(), elementsOnRow.end());
  comm.neighbor_bcast(DimsLr.data(), ones.data());
  comm.neighbor_bcast(Qdata.data(), elementsOnRow.data());
  comm.neighbor_bcast(Rdata.data(), elementsOnRow.data());
  comm.dup_bcast(DimsLr.data(), xlen);
  comm.dup_bcast(Qdata.data(), lenQ);
  comm.dup_bcast(Rdata.data(), lenQ);

  std::vector<int64_t> Csizes(CRows[nodes]), Coffsets(CRows[nodes] + 1);
  for (int64_t i = 0; i < nodes; i++)
    std::transform(&CCols[CRows[i]], &CCols[CRows[i + 1]], &Csizes[CRows[i]], 
      [&](int64_t col) { return DimsLr[i + ibegin] * DimsLr[comm.iLocal(col)]; });
  std::inclusive_scan(Csizes.begin(), Csizes.end(), Coffsets.begin() + 1);
  Coffsets[0] = 0;

  Cdata.resize(Coffsets.back());
  std::fill(Cdata.begin(), Cdata.end(), std::complex<double>(0., 0.));
  std::transform(Coffsets.begin(), Coffsets.begin() + CRows[nodes], C.begin(), [&](const int64_t d) { return &Cdata[d]; });

  int64_t dim_max = *std::max_element(DimsLr.begin(), DimsLr.end());
  std::vector<std::complex<double>> Tmp(dim_max * dim_max);
  std::complex<double> one(1., 0.), zero(0., 0.);

  for (int64_t i = 0; i < nodes; i++)
    for (int64_t ij = CRows[i]; ij < CRows[i + 1]; ij++) {
      int64_t j = comm.iLocal(CCols[ij]);
      int64_t m1 = DimsLrOld[i + ibegin], n1 = DimsLrOld[j];
      int64_t m2 = DimsLr[i + ibegin], n2 = DimsLr[j];
      int64_t ldl = Dims[i + ibegin], ldr = Dims[j];
      cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, m1, n2, n1, &one, CdataOld[ij].data(), m1, R[j], ldr, &zero, &Tmp[0], m1);
      cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m2, n2, m1, &one, R[i + ibegin], ldl, &Tmp[0], m1, &zero, &Cdata[Coffsets[ij]], m2);
    }
}

void ClusterBasis::adjustLowerRankGrowth(const ClusterBasis& prev_basis, const CellComm& comm) {
  int64_t xlen = comm.lenNeighbors();
  int64_t ibegin = comm.oLocal();
  int64_t nodes = comm.lenLocal();

  std::vector<int64_t> oldDims(nodes);
  std::vector<std::vector<std::complex<double>>> oldQ(nodes);
  std::vector<int64_t> newLocalChildLrDims(localChildLrDims.size());
  std::copy(&prev_basis.DimsLr[localChildIndex], &prev_basis.DimsLr[localChildIndex + localChildLrDims.size()], newLocalChildLrDims.begin());

  for (int64_t i = 0; i < nodes; i++) {
    oldDims[i] = Dims[i + ibegin];
    oldQ[i] = std::vector<std::complex<double>>(Q[i + ibegin], &(Q[i + ibegin])[oldDims[i] * DimsLr[i + ibegin]]);
    Dims[i + ibegin] = std::reduce(&newLocalChildLrDims[localChildOffsets[i]], &newLocalChildLrDims[localChildOffsets[i + 1]]);
  }

  const std::vector<int64_t> ones(xlen, 1);
  comm.neighbor_bcast(Dims.data(), ones.data());
  comm.dup_bcast(Dims.data(), xlen);

  std::vector<int64_t> Qoffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), elementsOnRow.begin(), [](const int64_t d) { return d * d; });
  std::inclusive_scan(elementsOnRow.begin(), elementsOnRow.end(), Qoffsets.begin() + 1);
  Qoffsets[0] = 0;
  Qdata = std::vector<std::complex<double>>(Qoffsets[xlen], std::complex<double>(0., 0.));
  Rdata = std::vector<std::complex<double>>(Qoffsets[xlen], std::complex<double>(0., 0.));
  std::transform(Qoffsets.begin(), Qoffsets.end(), Q.begin(), [&](const int64_t d) { return &Qdata[d]; });
  std::transform(Qoffsets.begin(), Qoffsets.end(), R.begin(), [&](const int64_t d) { return &Rdata[d]; });

  std::complex<double> one(1., 0.), zero(0., 0.);
  for (int64_t i = 0; i < nodes; i++) {
    int64_t M = Dims[i + ibegin];
    int64_t N = DimsLr[i + ibegin];
    int64_t childi = localChildOffsets[i];
    int64_t cend = localChildOffsets[i + 1];
    for (int64_t j = childi; j < cend && 0 < N; j++) {
      int64_t offsetOld = std::reduce(&localChildLrDims[childi], &localChildLrDims[j]);
      int64_t offsetNew = std::reduce(&newLocalChildLrDims[childi], &newLocalChildLrDims[j]);
      int64_t m1 = localChildLrDims[j], m2 = newLocalChildLrDims[j];
      int64_t lj = localChildIndex + j;
      cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m2, N, m1, &one, 
        prev_basis.R[lj], prev_basis.Dims[lj], &(oldQ[i])[offsetOld], oldDims[i], &zero, &Qdata[Qoffsets[i + ibegin] + offsetNew], M);
    }
  }

  std::copy(newLocalChildLrDims.begin(), newLocalChildLrDims.end(), localChildLrDims.begin());
  comm.neighbor_bcast(Qdata.data(), elementsOnRow.data());
  comm.dup_bcast(Qdata.data(), Qoffsets[xlen]);
}

MatVec::MatVec(const MatrixAccessor& eval, const ClusterBasis basis[], const double bodies[], const Cell cells[], const CSR& near, const CellComm comm[], int64_t levels) :
  EvalFunc(&eval), Basis(basis), Bodies(bodies), Cells(cells), Near(&near), Comm(comm), Levels(levels) {
}

void MatVec::operator() (int64_t nrhs, std::complex<double> X[], int64_t ldX) const {
  int64_t lbegin = Comm[Levels].oLocal();
  int64_t llen = Comm[Levels].lenLocal();

  std::vector<std::vector<std::complex<double>>> rhsX(Levels + 1), rhsY(Levels + 1);
  std::vector<std::vector<std::complex<double>*>> rhsXptr(Levels + 1), rhsYptr(Levels + 1);
  std::vector<std::vector<std::pair<std::complex<double>*, int64_t>>> rhsXoptr(Levels + 1), rhsYoptr(Levels + 1);

  for (int64_t l = Levels; l >= 0; l--) {
    int64_t xlen = Comm[l].lenNeighbors();
    std::vector<int64_t> offsets(xlen + 1, 0);
    std::inclusive_scan(Basis[l].Dims.begin(), Basis[l].Dims.end(), offsets.begin() + 1);

    rhsX[l] = std::vector<std::complex<double>>(offsets[xlen] * nrhs, std::complex<double>(0., 0.));
    rhsY[l] = std::vector<std::complex<double>>(offsets[xlen] * nrhs, std::complex<double>(0., 0.));
    rhsXptr[l] = std::vector<std::complex<double>*>(xlen, nullptr);
    rhsYptr[l] = std::vector<std::complex<double>*>(xlen, nullptr);
    rhsXoptr[l] = std::vector<std::pair<std::complex<double>*, int64_t>>(xlen, std::make_pair(nullptr, 0));
    rhsYoptr[l] = std::vector<std::pair<std::complex<double>*, int64_t>>(xlen, std::make_pair(nullptr, 0));

    std::transform(offsets.begin(), offsets.begin() + xlen, rhsXptr[l].begin(), [&](const int64_t d) { return &rhsX[l][0] + d * nrhs; });
    std::transform(offsets.begin(), offsets.begin() + xlen, rhsYptr[l].begin(), [&](const int64_t d) { return &rhsY[l][0] + d * nrhs; });

    if (l < Levels)
      for (int64_t i = 0; i < xlen; i++) {
        int64_t ci = Comm[l].iGlobal(i);
        int64_t child = Comm[l + 1].iLocal(Cells[ci].Child[0]);
        int64_t clen = Cells[ci].Child[1] - Cells[ci].Child[0];

        if (child >= 0 && clen > 0) {
          std::vector<int64_t> offsets_child(clen + 1, 0);
          std::inclusive_scan(&Basis[l + 1].DimsLr[child], &Basis[l + 1].DimsLr[child + clen], offsets_child.begin() + 1);
          int64_t ldi = Basis[l].Dims[i];
          std::transform(offsets_child.begin(), offsets_child.begin() + clen, &rhsXoptr[l + 1][child], 
            [&](const int64_t d) { return std::make_pair(rhsXptr[l][i] + d, ldi); });
          std::transform(offsets_child.begin(), offsets_child.begin() + clen, &rhsYoptr[l + 1][child], 
            [&](const int64_t d) { return std::make_pair(rhsYptr[l][i] + d, ldi); });
        }
      }
  }

  int64_t Y = 0;
  for (int64_t i = 0; i < llen; i++) {
    int64_t M = Basis[Levels].Dims[lbegin + i];
    MKL_Zomatcopy('C', 'N', M, nrhs, std::complex<double>(1., 0.), &X[Y], ldX, rhsXptr[Levels][lbegin + i], M);
    Y = Y + M;
  }

  const std::complex<double> one(1., 0.), zero(0., 0.);
  for (int64_t i = Levels; i > 0; i--) {
    int64_t ibegin = Comm[i].oLocal();
    int64_t iboxes = Comm[i].lenLocal();
    int64_t xlen = Comm[i].lenNeighbors();

    std::vector<int64_t> lens(xlen);
    std::transform(Basis[i].Dims.begin(), Basis[i].Dims.end(), lens.begin(), [=](const int64_t& i) { return i * nrhs; });
    int64_t lenI = nrhs * std::reduce(&Basis[i].Dims[0], &Basis[i].Dims[xlen]);
    Comm[i].level_merge(rhsX[i].data(), lenI);
    Comm[i].neighbor_bcast(rhsX[i].data(), lens.data());
    Comm[i].dup_bcast(rhsX[i].data(), lenI);

    for (int64_t y = 0; y < iboxes; y++) {
      int64_t M = Basis[i].Dims[y + ibegin];
      int64_t N = Basis[i].DimsLr[y + ibegin];
      if (M > 0 && N > 0)
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, N, nrhs, M, &one, Basis[i].Q[y + ibegin], M, 
          rhsXptr[i][y + ibegin], M, &zero, rhsXoptr[i][y + ibegin].first, rhsXoptr[i][y + ibegin].second);
    }
  }

  if (Basis[0].Dims[0] > 0) {
    Comm[0].level_merge(rhsX[0].data(), Basis[0].Dims[0] * nrhs);
    Comm[0].dup_bcast(rhsX[0].data(), Basis[0].Dims[0] * nrhs);
  }

  for (int64_t i = 1; i <= Levels; i++) {
    int64_t ibegin = Comm[i].oLocal();
    int64_t iboxes = Comm[i].lenLocal();

    for (int64_t y = 0; y < iboxes; y++) {
      int64_t M = Basis[i].Dims[y + ibegin];
      int64_t K = Basis[i].DimsLr[y + ibegin];

      for (int64_t yx = Basis[i].CRows[y]; yx < Basis[i].CRows[y + 1]; yx++) {
        int64_t x = Comm[i].iLocal(Basis[i].CCols[yx]);
        int64_t N = Basis[i].DimsLr[x];
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, K, nrhs, N, &one, Basis[i].C[yx], K, 
          rhsXoptr[i][x].first, rhsXoptr[i][x].second, &one, rhsYoptr[i][y + ibegin].first, rhsYoptr[i][y + ibegin].second);
      }
      if (M > 0 && K > 0)
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, nrhs, K, &one, Basis[i].Q[y + ibegin], M, 
          rhsYoptr[i][y + ibegin].first, rhsYoptr[i][y + ibegin].second, &zero, rhsYptr[i][y + ibegin], M);
    }
  }

  int64_t gbegin = Comm[Levels].oGlobal();
  for (int64_t y = 0; y < llen; y++)
    for (int64_t yx = Near->RowIndex[y + gbegin]; yx < Near->RowIndex[y + gbegin + 1]; yx++) {
      int64_t x = Near->ColIndex[yx];
      int64_t x_loc = Comm[Levels].iLocal(x);
      int64_t M = Cells[y + gbegin].Body[1] - Cells[y + gbegin].Body[0];
      int64_t N = Cells[x].Body[1] - Cells[x].Body[0];
      mat_vec_reference(*EvalFunc, M, N, nrhs, rhsYptr[Levels][y + lbegin], M, rhsXptr[Levels][x_loc], N, &Bodies[3 * Cells[y + gbegin].Body[0]], &Bodies[3 * Cells[x].Body[0]]);
    }
  Y = 0;
  for (int64_t i = 0; i < llen; i++) {
    int64_t M = Basis[Levels].Dims[lbegin + i];
    MKL_Zomatcopy('C', 'N', M, nrhs, std::complex<double>(1., 0.), rhsYptr[Levels][lbegin + i], M, &X[Y], ldX);
    Y = Y + M;
  }
}
