#include <basis.hpp>
#include <build_tree.hpp>
#include <comm.hpp>
#include <kernel.hpp>

#include <cblas.h>
#include <lapacke.h>
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

int64_t compute_basis(const MatrixAccessor& eval, double epi, int64_t M, int64_t N, double Xbodies[], const double Fbodies[], std::complex<double> A[], int64_t LDA, std::complex<double> R[], int64_t LDR) {
  int64_t K = std::max(M, N);
  std::complex<double> one(1., 0.), zero(0., 0.);
  std::vector<std::complex<double>> B(M * K);
  std::vector<double> S(M * 3);
  std::vector<int32_t> jpiv(M, 0);

  lapack_complex_double* Aptr = reinterpret_cast<lapack_complex_double*>(&A[0]);
  lapack_complex_double* Bptr = reinterpret_cast<lapack_complex_double*>(&B[0]);
  lapack_complex_double* Rptr = reinterpret_cast<lapack_complex_double*>(&R[0]);
  lapack_complex_double* Tptr = reinterpret_cast<lapack_complex_double*>(&S[0]);
  lapack_complex_double* One = reinterpret_cast<lapack_complex_double*>(&one);
  lapack_complex_double* Zero = reinterpret_cast<lapack_complex_double*>(&zero);

  gen_matrix(eval, N, M, Fbodies, Xbodies, &B[0], K);
  LAPACKE_zgeqrf(LAPACK_COL_MAJOR, N, M, Bptr, K, Tptr);
  LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, *Zero, *Zero, &Bptr[1], K);
  LAPACKE_zgeqp3(LAPACK_COL_MAJOR, M, M, Bptr, K, &jpiv[0], Tptr);
  int64_t rank = 0;
  double s0 = epi * std::abs(B[0]);
  while (rank < M && s0 <= std::abs(B[rank * (K + 1)]))
    ++rank;
  
  if (rank > 0) {
    if (rank < M)
      cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, rank, M - rank, &one, &B[0], K, &B[rank * K], K);
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'F', rank, rank, *Zero, *One, &Bptr[0], K);

    for (int64_t i = 0; i < M; i++) {
      int64_t piv = (int64_t)jpiv[i] - 1;
      std::copy(&B[i * K], &B[i * K + rank], &R[piv * LDR]);
      std::copy(&Xbodies[piv * 3], &Xbodies[piv * 3 + 3], &S[i * 3]);
    }
    std::copy(&S[0], &S[M * 3], Xbodies);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, rank, M, &one, A, LDA, R, LDR, &zero, &B[0], M);
    LAPACKE_zgeqrf(LAPACK_COL_MAJOR, M, rank, Bptr, M, Tptr);
    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'L', M, rank, Bptr, M, Aptr, LDA);
    LAPACKE_zungqr(LAPACK_COL_MAJOR, M, M, rank, Aptr, LDA, Tptr);
    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'U', rank, rank, Bptr, M, Rptr, LDR);
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', rank - 1, rank - 1, *Zero, *Zero, &Rptr[1], LDR);
  }
  return rank;
}

template <typename T>
void memcpy2d(T* dst, const T* src, int64_t rows, int64_t cols, int64_t ld_dst, int64_t ld_src) {
  if (rows == ld_dst && rows == ld_src)
    std::copy(src, src + rows * cols, dst);
  else 
    for (int64_t i = 0; i < cols; i++)
      std::copy(&src[i * ld_src], &src[i * ld_src + rows], &dst[i * ld_dst]);
}

ClusterBasis::ClusterBasis(const MatrixAccessor& eval, double epi, const Cell cells[], const double bodies[], const WellSeparatedApproximation& wsa, const CellComm& comm, const ClusterBasis& prev_basis, const CellComm& prev_comm) {
  int64_t xlen = comm.lenNeighbors();
  int64_t ibegin = comm.oLocal();
  int64_t nodes = comm.lenLocal();
  Dims = std::vector<int64_t>(xlen, 0);
  DimsLr = std::vector<int64_t>(xlen, 0);
  S = std::vector<const double*>(xlen);
  Q = std::vector<const std::complex<double>*>(xlen);
  R = std::vector<const std::complex<double>*>(xlen);

  for (int64_t i = 0; i < nodes; i++) {
    int64_t ci = comm.iGlobal(i + ibegin);
    int64_t childi = prev_comm.iLocal(cells[ci].Child[0]);
    int64_t clen = cells[ci].Child[1] - cells[ci].Child[0];
    Dims[i + ibegin] = (0 <= childi && 0 < clen) ? 
      std::reduce(&prev_basis.DimsLr[childi], &prev_basis.DimsLr[childi + clen]) : cells[ci].Body[1] - cells[ci].Body[0];
  }

  const std::vector<int64_t> ones(xlen, 1);
  comm.neighbor_bcast(Dims.data(), ones.data());
  comm.dup_bcast(Dims.data(), xlen);

  std::vector<int64_t> Qsizes(xlen), Qoffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), Qsizes.begin(), [](const int64_t d) { return d * d; });
  std::inclusive_scan(Qsizes.begin(), Qsizes.end(), Qoffsets.begin() + 1);
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

    int64_t ci = comm.iGlobal(i + ibegin);
    int64_t childi = prev_comm.iLocal(cells[ci].Child[0]);
    int64_t clen = cells[ci].Child[1] - cells[ci].Child[0];

    if (clen <= 0) {
      std::copy(&bodies[3 * cells[ci].Body[0]], &bodies[3 * cells[ci].Body[1]], ske);
      for (int64_t j = 0; j < dim; j++)
        matrix[j * (dim + 1)] = std::complex<double>(1., 0.);
    }
    for (int64_t j = 0; j < clen; j++) {
      int64_t offset = std::reduce(&prev_basis.DimsLr[childi], &prev_basis.DimsLr[childi + j]);
      int64_t len = prev_basis.DimsLr[childi + j];
      const double* mbegin = prev_basis.S[childi + j];
      std::copy(mbegin, &mbegin[len * 3], &ske[offset * 3]);
      memcpy2d(&matrix[offset * (dim + 1)], prev_basis.R[childi + j], len, len, dim, prev_basis.Dims[childi + j]);
    }

    int64_t fsize = wsa.fbodies_size_at_i(i);
    const double* fbodies = wsa.fbodies_at_i(i);
    int64_t rank = (dim > 0 && fsize > 0) ? compute_basis(eval, epi, dim, fsize, ske, fbodies, matrix, dim, &Rdata[Qoffsets[i + ibegin]], dim) : 0;
    DimsLr[i + ibegin] = rank;
  }

  comm.neighbor_bcast(DimsLr.data(), ones.data());
  comm.dup_bcast(DimsLr.data(), xlen);
  comm.neighbor_bcast(Sdata.data(), Ssizes.data());
  comm.dup_bcast(Sdata.data(), Soffsets[xlen]);
  comm.neighbor_bcast(Qdata.data(), Qsizes.data());
  comm.dup_bcast(Qdata.data(), Qoffsets[xlen]);
  comm.neighbor_bcast(Rdata.data(), Qsizes.data());
  comm.dup_bcast(Rdata.data(), Qoffsets[xlen]);
}

MatVec::MatVec(const MatrixAccessor& eval, const ClusterBasis basis[], const double bodies[], const Cell cells[], const CSR& near, const CSR& far, const CellComm comm[], int64_t levels) :
  EvalFunc(&eval), Basis(basis), Bodies(bodies), Cells(cells), Near(&near), Far(&far), Comm(comm), Levels(levels) {
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
    memcpy2d(rhsXptr[Levels][lbegin + i], &X[Y], M, nrhs, M, ldX);
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
    int64_t gbegin = Comm[i].oGlobal();

    for (int64_t y = 0; y < iboxes; y++) {
      int64_t K = Basis[i].DimsLr[y + ibegin];
      for (int64_t yx = Far->RowIndex[y + gbegin]; yx < Far->RowIndex[y + gbegin + 1]; yx++) {
        int64_t x = Comm[i].iLocal(Far->ColIndex[yx]);
        int64_t N = Basis[i].DimsLr[x];

        std::vector<std::complex<double>> TMPX(N * nrhs, std::complex<double>(0., 0.));
        std::vector<std::complex<double>> TMPB(K * nrhs, std::complex<double>(0., 0.));
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, N, nrhs, N, &one, Basis[i].R[x], Basis[i].Dims[x], rhsXoptr[i][x].first, rhsXoptr[i][x].second, &zero, &TMPX[0], N);
        mat_vec_reference(*EvalFunc, K, N, nrhs, &TMPB[0], K, &TMPX[0], N, Basis[i].S[y + ibegin], Basis[i].S[x]);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, K, nrhs, K, &one, Basis[i].R[y + ibegin], Basis[i].Dims[y + ibegin], &TMPB[0], K, &one, rhsYoptr[i][y + ibegin].first, rhsYoptr[i][y + ibegin].second);
      }
      int64_t M = Basis[i].Dims[y + ibegin];
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
    memcpy2d(&X[Y], rhsYptr[Levels][lbegin + i], M, nrhs, ldX, M);
    Y = Y + M;
  }
}
