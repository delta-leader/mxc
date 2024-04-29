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

ClusterBasis::ClusterBasis(const MatrixAccessor& eval, double epi, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& wsa, const CellComm& comm, const ClusterBasis& prev_basis, const CellComm& prev_comm) {
  long long xlen = comm.lenNeighbors();
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long ybegin = comm.oGlobal();

  long long ychild = cells[ybegin].Child[0];
  localChildIndex = prev_comm.iLocal(ychild);
  selfChildIndex = 0 <= ychild ? (prev_comm.oGlobal() - ychild) : 0;
  localChildOffsets = std::vector<long long>(nodes + 1);
  localChildOffsets[0] = 0;
  std::transform(&cells[ybegin], &cells[ybegin + nodes], localChildOffsets.begin() + 1, [=](const Cell& c) { return c.Child[1] - ychild; });

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

  std::vector<long long> Qoffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), elementsOnRow.begin(), [](const long long d) { return d * d; });
  std::inclusive_scan(elementsOnRow.begin(), elementsOnRow.end(), Qoffsets.begin() + 1);
  Qoffsets[0] = 0;
  Qdata = std::vector<std::complex<double>>(Qoffsets[xlen], std::complex<double>(0., 0.));
  Rdata = std::vector<std::complex<double>>(Qoffsets[xlen], std::complex<double>(0., 0.));
  std::transform(Qoffsets.begin(), Qoffsets.begin() + xlen, Q.begin(), [&](const long long d) { return &Qdata[d]; });
  std::transform(Qoffsets.begin(), Qoffsets.begin() + xlen, R.begin(), [&](const long long d) { return &Rdata[d]; });

  std::vector<long long> Ssizes(xlen), Soffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), Ssizes.begin(), [](const long long d) { return 3 * d; });
  std::inclusive_scan(Ssizes.begin(), Ssizes.end(), Soffsets.begin() + 1);
  Soffsets[0] = 0;
  Sdata = std::vector<double>(Soffsets[xlen], 0.);
  std::transform(Soffsets.begin(), Soffsets.begin() + xlen, S.begin(), [&](const long long d) { return &Sdata[d]; });

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
    long long rank = compute_basis(eval, epi, dim, fsize, ske, fbodies, matrix);
    DimsLr[i + ibegin] = rank;
  }

  comm.neighbor_bcast(DimsLr.data(), ones.data());
  comm.neighbor_bcast(Sdata.data(), Ssizes.data());
  comm.neighbor_bcast(Qdata.data(), elementsOnRow.data());

  CRows = std::vector<long long>(&Far.RowIndex[ybegin], &Far.RowIndex[ybegin + nodes + 1]);
  CCols = std::vector<long long>(&Far.ColIndex[CRows[0]], &Far.ColIndex[CRows[nodes]]);
  CColsLocal = std::vector<long long>(CRows.back());
  long long offset = CRows[0];
  std::for_each(CRows.begin(), CRows.end(), [=](long long& i) { i = i - offset; });
  std::transform(CCols.begin(), CCols.end(), CColsLocal.begin(), [&](long long col) { return comm.iLocal(col); });

  std::vector<long long> Csizes(CRows[nodes]), Coffsets(CRows[nodes] + 1);
  for (long long i = 0; i < nodes; i++)
    std::transform(&CColsLocal[CRows[i]], &CColsLocal[CRows[i + 1]], &Csizes[CRows[i]], 
      [&](long long col) { return DimsLr[i + ibegin] * DimsLr[col]; });
  std::inclusive_scan(Csizes.begin(), Csizes.end(), Coffsets.begin() + 1);
  Coffsets[0] = 0;

  C = std::vector<const std::complex<double>*>(CRows[nodes]);
  Cdata = std::vector<std::complex<double>>(Coffsets.back());
  std::transform(Coffsets.begin(), Coffsets.begin() + CRows[nodes], C.begin(), [&](const long long d) { return &Cdata[d]; });

  for (long long i = 0; i < nodes; i++)
    for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
      long long j = CColsLocal[ij];
      long long m = DimsLr[i + ibegin], n = DimsLr[j];
      gen_matrix(eval, m, n, S[i + ibegin], S[j], &Cdata[Coffsets[ij]]);
    }
}

long long ClusterBasis::copyOffset(long long i) const {
  return std::reduce(&DimsLr[std::max((long long)0, i - ParentSequenceNum[i])], &DimsLr[i], (long long)0);
}

long long ClusterBasis::childWriteOffset() const {
  return std::reduce(localChildLrDims.begin(), localChildLrDims.begin() + selfChildIndex, (long long)0);
}

MatVec::MatVec(const MatrixAccessor& eval, const ClusterBasis basis[], const double bodies[], const Cell cells[], const CSR& near, const CellComm comm[], long long levels) :
  EvalFunc(&eval), Basis(basis), Bodies(bodies), Cells(cells), Near(&near), Comm(comm), Levels(levels) {
}

void MatVec::operator() (long long nrhs, std::complex<double> X[]) const {
  long long lbegin = Comm[Levels].oLocal();
  long long llen = Comm[Levels].lenLocal();

  std::vector<std::vector<std::complex<double>>> rhsX(Levels + 1), rhsY(Levels + 1);
  std::vector<std::vector<long long>> offsets(Levels + 1);
  std::vector<std::vector<std::pair<std::complex<double>*, long long>>> rhsXoptr(Levels + 1), rhsYoptr(Levels + 1);

  for (long long l = Levels; l >= 0; l--) {
    long long xlen = Comm[l].lenNeighbors();
    offsets[l] = std::vector<long long>(xlen + 1, 0);
    std::inclusive_scan(Basis[l].Dims.begin(), Basis[l].Dims.end(), offsets[l].begin() + 1);

    rhsX[l] = std::vector<std::complex<double>>(offsets[l][xlen] * nrhs, std::complex<double>(0., 0.));
    rhsY[l] = std::vector<std::complex<double>>(offsets[l][xlen] * nrhs, std::complex<double>(0., 0.));
    rhsXoptr[l] = std::vector<std::pair<std::complex<double>*, long long>>(xlen, std::make_pair(nullptr, 0));
    rhsYoptr[l] = std::vector<std::pair<std::complex<double>*, long long>>(xlen, std::make_pair(nullptr, 0));

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
            [&](const long long d) { return std::make_pair(rhsX[l].data() + nrhs * offsets[l][i] + d, ldi); });
          std::transform(offsets_child.begin(), offsets_child.begin() + clen, &rhsYoptr[l + 1][child], 
            [&](const long long d) { return std::make_pair(rhsY[l].data() + nrhs * offsets[l][i] + d, ldi); });
        }
      }
  }

  const std::complex<double> one(1., 0.), zero(0., 0.);
  long long Y = 0, lenX = std::reduce(&Basis[Levels].Dims[lbegin], &Basis[Levels].Dims[lbegin + llen]);
  for (long long i = 0; i < llen; i++) {
    long long M = Basis[Levels].Dims[lbegin + i];
    std::complex<double>* Xptr = rhsX[Levels].data() + nrhs * offsets[Levels][lbegin + i];
    MKL_Zomatcopy('C', 'N', M, nrhs, one, &X[Y], lenX, Xptr, M);
    Y = Y + M;
  }

  for (long long i = Levels; i >= 0; i--) {
    long long ibegin = Comm[i].oLocal();
    long long iboxes = Comm[i].lenLocal();
    long long xlen = Comm[i].lenNeighbors();

    long long lenI = nrhs * std::reduce(&Basis[i].Dims[0], &Basis[i].Dims[xlen]);
    Comm[i].level_merge(rhsX[i].data(), lenI);

    for (long long y = 0; y < iboxes; y++) {
      long long M = Basis[i].Dims[y + ibegin];
      long long N = Basis[i].DimsLr[y + ibegin];
      std::complex<double>* Xptr = rhsX[i].data() + nrhs * offsets[i][ibegin + y];
      if (0 < N)
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, N, nrhs, M, &one, Basis[i].Q[y + ibegin], M, 
          Xptr, M, &zero, rhsXoptr[i][y + ibegin].first, rhsXoptr[i][y + ibegin].second);
    }
  }

  for (long long i = Levels; i >= 0; i--) {
    long long xlen = Comm[i].lenNeighbors();
    std::vector<long long> lens(xlen);
    std::transform(Basis[i].Dims.begin(), Basis[i].Dims.end(), lens.begin(), [=](long long d) { return d * nrhs; });
    Comm[i].neighbor_bcast(rhsX[i].data(), lens.data());
  }

  for (long long i = 1; i <= Levels; i++) {
    long long ibegin = Comm[i].oLocal();
    long long iboxes = Comm[i].lenLocal();

    for (long long y = 0; y < iboxes; y++) {
      long long M = Basis[i].Dims[y + ibegin];
      long long K = Basis[i].DimsLr[y + ibegin];
      std::complex<double>* Yptr = rhsY[i].data() + nrhs * offsets[i][ibegin + y];

      if (0 < K) {
        for (long long yx = Basis[i].CRows[y]; yx < Basis[i].CRows[y + 1]; yx++) {
          long long x = Basis[i].CColsLocal[yx];
          long long N = Basis[i].DimsLr[x];
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, K, nrhs, N, &one, Basis[i].C[yx], K, 
              rhsXoptr[i][x].first, rhsXoptr[i][x].second, &one, rhsYoptr[i][y + ibegin].first, rhsYoptr[i][y + ibegin].second);
        }
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, nrhs, K, &one, Basis[i].Q[y + ibegin], M, 
          rhsYoptr[i][y + ibegin].first, rhsYoptr[i][y + ibegin].second, &zero, Yptr, M);
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
      std::complex<double>* Yptr = rhsY[Levels].data() + nrhs * offsets[Levels][lbegin + y];
      const std::complex<double>* Xptr = rhsX[Levels].data() + nrhs * offsets[Levels][x_loc];
      mat_vec_reference(*EvalFunc, M, N, nrhs, Yptr, Xptr, &Bodies[3 * Cells[y + gbegin].Body[0]], &Bodies[3 * Cells[x].Body[0]]);
    }
  Y = 0;
  for (long long i = 0; i < llen; i++) {
    long long M = Basis[Levels].Dims[lbegin + i];
    const std::complex<double>* Yptr = rhsY[Levels].data() + nrhs * offsets[Levels][lbegin + i];
    MKL_Zomatcopy('C', 'N', M, nrhs, one, Yptr, M, &X[Y], lenX);
    Y = Y + M;
  }
}
