
#include <basis.hpp>
#include <build_tree.hpp>
#include <comm.hpp>
#include <kernel.hpp>
#include <lowrank.hpp>

#include <cblas.h>
#include <lapacke.h>
#include <algorithm>
#include <numeric>
#include <cmath>

template <typename T>
void memcpy2d(T* dst, const T* src, int64_t rows, int64_t cols, int64_t ld_dst, int64_t ld_src) {
  if (rows == ld_dst && rows == ld_src)
    std::copy(src, src + rows * cols, dst);
  else 
    for (int64_t i = 0; i < cols; i++)
      std::copy(&src[i * ld_src], &src[i * ld_src + rows], &dst[i * ld_dst]);
}

template <typename T>
void matrixLaset(char uplo, int64_t M, int64_t N, T alpha, T beta, T A[], int64_t LDA) {
  for (int64_t x = 0; x < N; x++)
    for (int64_t y = 0; y < M; y++) {
      if ((y < x && uplo != 'L') || (y > x && uplo != 'U'))
        A[y + x * LDA] = alpha;
      else if (y == x)
        A[y + x * LDA] = beta;
    }
}

void compute_AallT(const Eval& eval, int64_t M, const double Xbodies[], std::vector<std::pair<const double*, int64_t>>& Fbodies, std::complex<double> Aall[], int64_t LDA) {
  if (M > 0) {
    int64_t N = std::max(M, (int64_t)(1 << 11)), B2 = N + M;
    std::vector<std::complex<double>> B(M * B2, 0.), tau(M);
    std::complex<double> zero(0., 0.);

    int64_t loc = 0;
    for (int64_t i = 0; i < (int64_t)Fbodies.size(); i++) {
      int64_t loc_i = 0;
      while(loc_i < Fbodies[i].second) {
        int64_t len = std::min(Fbodies[i].second - loc_i, N - loc);
        gen_matrix(eval, len, M, Fbodies[i].first + (loc_i * 3), Xbodies, &B[M + loc], B2);
        loc_i = loc_i + len;
        loc = loc + len;
        if (loc == N) {
          LAPACKE_zgeqrf(LAPACK_COL_MAJOR, M + N, M, reinterpret_cast<lapack_complex_double*>(&B[0]), B2, reinterpret_cast<lapack_complex_double*>(&tau[0]));
          matrixLaset('L', M - 1, M - 1, zero, zero, &B[1], B2);
          loc = 0;
        }
      }
    }

    if (loc > 0)
      LAPACKE_zgeqrf(LAPACK_COL_MAJOR, M + loc, M, reinterpret_cast<lapack_complex_double*>(&B[0]), B2, reinterpret_cast<lapack_complex_double*>(&tau[0]));
    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'U', M, M, reinterpret_cast<lapack_complex_double*>(&B[0]), B2, reinterpret_cast<lapack_complex_double*>(Aall), LDA);
    matrixLaset('L', M - 1, M - 1, zero, zero, &Aall[1], LDA);
  }
}

std::vector<std::pair<const double*, int64_t>> getRemote(int64_t ci, const Cell cells[], const CSR& Near, const double bodies[], int64_t nbodies) {
  std::vector<std::pair<const double*, int64_t>> remote;
  int64_t loc = 0;
  for (int64_t c = Near.RowIndex[ci]; c < Near.RowIndex[ci + 1]; c++) {
    int64_t cj = Near.ColIndex[c];
    int64_t len = cells[cj].Body[0] - loc;
    if (len > 0)
      remote.emplace_back(&bodies[loc * 3], len);
    loc = cells[cj].Body[1];
  }
  if (loc < nbodies)
    remote.emplace_back(&bodies[loc * 3], nbodies - loc);
  return remote;
}

ClusterBasis::ClusterBasis(const Eval& eval, double epi, const Cell cells[], const CSR& Near, const double bodies[], int64_t nbodies, const CellComm& comm) {
  int64_t xlen = comm.lenNeighbors();
  int64_t ibegin = comm.oLocal();
  int64_t nodes = comm.lenLocal();
  Dims = std::vector<int64_t>(xlen, 0);
  DimsLr = std::vector<int64_t>(xlen, 0);
  V = std::vector<std::complex<double>*>(xlen);

  for (int64_t i = 0; i < xlen; i++) {
    int64_t ci = comm.iGlobal(i);
    Dims[i] = cells[ci].Body[1] - cells[ci].Body[0];
  }

  std::vector<int64_t> Vsizes(xlen), Voffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), Vsizes.begin(), [](const int64_t d) { return d * d; });
  std::inclusive_scan(Vsizes.begin(), Vsizes.end(), Voffsets.begin() + 1);
  Voffsets[0] = 0;
  Vdata = std::vector<std::complex<double>>(Voffsets[xlen], std::complex<double>(0., 0.));
  std::transform(Voffsets.begin(), Voffsets.end(), V.begin(), [&](const int64_t d) { return &Vdata[d]; });

  std::vector<int64_t> Msizes(xlen), Moffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), Msizes.begin(), [](const int64_t d) { return 3 * d; });
  std::inclusive_scan(Msizes.begin(), Msizes.end(), Moffsets.begin() + 1);
  Moffsets[0] = 0;
  Mdata = std::vector<double>(Moffsets[xlen], 0.);

  for (int64_t i = 0; i < nodes; i++) {
    int64_t dim = Dims[i + ibegin];
    std::complex<double>* matrix = &Vdata[Voffsets[i + ibegin]];
    double* ske = &Mdata[Moffsets[i + ibegin]];

    int64_t ci = comm.iGlobal(i + ibegin);
    std::vector<std::pair<const double*, int64_t>> remote = getRemote(ci, cells, Near, bodies, nbodies);

    std::copy(&bodies[3 * cells[ci].Body[0]], &bodies[3 * cells[ci].Body[1]], ske);
    if (remote.size() > 0) {
      compute_AallT(eval, dim, ske, remote, matrix, dim);
      LowRank lr(epi, dim, dim, matrix, dim, ske, 3);
      int64_t rank = lr.Rank;
      if (rank > 0) {
        memcpy2d(matrix, &lr.V[0], rank, dim, dim, rank);
        std::copy(lr.BodiesJ.begin(), lr.BodiesJ.end(), ske);
      }
      DimsLr[i + ibegin] = rank;
    }
  }

  const std::vector<int64_t> ones(xlen, 1);
  comm.neighbor_bcast(DimsLr.data(), ones.data());
  comm.dup_bcast(DimsLr.data(), xlen);
  comm.neighbor_bcast(Mdata.data(), Msizes.data());
  comm.dup_bcast(Mdata.data(), Moffsets[xlen]);
  comm.neighbor_bcast(Vdata.data(), Vsizes.data());
  comm.dup_bcast(Vdata.data(), Voffsets[xlen]);
}

ClusterBasis::ClusterBasis(const Eval& eval, double epi, const ClusterBasis& prev_basis, const Cell cells[], const CSR& Near, const double bodies[], int64_t nbodies, const CellComm& comm, const CellComm& prev_comm) {
  int64_t xlen = comm.lenNeighbors();
  int64_t ibegin = comm.oLocal();
  int64_t nodes = comm.lenLocal();
  Dims = std::vector<int64_t>(xlen, 0);
  DimsLr = std::vector<int64_t>(xlen, 0);
  V = std::vector<std::complex<double>*>(xlen);

  for (int64_t i = 0; i < nodes; i++) {
    int64_t ci = comm.iGlobal(i + ibegin);
    int64_t childi = prev_comm.iLocal(cells[ci].Child[0]);
    int64_t clen = cells[ci].Child[1] - cells[ci].Child[0];
    Dims[i + ibegin] = std::accumulate(&prev_basis.DimsLr[childi], &prev_basis.DimsLr[childi + clen], 0);
  }

  const std::vector<int64_t> ones(xlen, 1);
  comm.neighbor_bcast(Dims.data(), ones.data());
  comm.dup_bcast(Dims.data(), xlen);

  std::vector<int64_t> Vsizes(xlen), Voffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), Vsizes.begin(), [](const int64_t d) { return d * d; });
  std::inclusive_scan(Vsizes.begin(), Vsizes.end(), Voffsets.begin() + 1);
  Voffsets[0] = 0;
  Vdata = std::vector<std::complex<double>>(Voffsets[xlen], std::complex<double>(0., 0.));
  std::transform(Voffsets.begin(), Voffsets.end(), V.begin(), [&](const int64_t d) { return &Vdata[d]; });

  std::vector<int64_t> Msizes(xlen), Moffsets(xlen + 1);
  std::transform(Dims.begin(), Dims.end(), Msizes.begin(), [](const int64_t d) { return 3 * d; });
  std::inclusive_scan(Msizes.begin(), Msizes.end(), Moffsets.begin() + 1);
  Moffsets[0] = 0;
  Mdata = std::vector<double>(Moffsets[xlen], 0.);

  for (int64_t i = 0; i < nodes; i++) {
    int64_t dim = Dims[i + ibegin];
    std::complex<double>* matrix = &Vdata[Voffsets[i + ibegin]];
    double* ske = &Mdata[Moffsets[i + ibegin]];

    int64_t ci = comm.iGlobal(i + ibegin);
    int64_t childi = prev_comm.iLocal(cells[ci].Child[0]);
    int64_t clen = cells[ci].Child[1] - cells[ci].Child[0];
    std::vector<std::pair<const double*, int64_t>> remote = getRemote(ci, cells, Near, bodies, nbodies);

    for (int64_t j = 0; j < clen; j++) {
      int64_t offset = std::accumulate(&prev_basis.DimsLr[childi], &prev_basis.DimsLr[childi + j], 0);
      int64_t len = prev_basis.DimsLr[childi + j];
      const double* mbegin = prev_basis.ske_at_i(childi + j);
      std::copy(mbegin, &mbegin[len * 3], &ske[offset * 3]);
    }

    if (remote.size() > 0) {
      compute_AallT(eval, dim, ske, remote, matrix, dim);
      LowRank lr(epi, dim, dim, matrix, dim, ske, 3);
      int64_t rank = lr.Rank;
      if (rank > 0) {
        memcpy2d(matrix, &lr.V[0], rank, dim, dim, rank);
        std::copy(lr.BodiesJ.begin(), lr.BodiesJ.end(), ske);
      }
      DimsLr[i + ibegin] = rank;
    }
  }

  comm.neighbor_bcast(DimsLr.data(), ones.data());
  comm.dup_bcast(DimsLr.data(), xlen);
  comm.neighbor_bcast(Mdata.data(), Msizes.data());
  comm.dup_bcast(Mdata.data(), Moffsets[xlen]);
  comm.neighbor_bcast(Vdata.data(), Vsizes.data());
  comm.dup_bcast(Vdata.data(), Voffsets[xlen]);
}

const double* ClusterBasis::ske_at_i(int64_t i) const {
  return Mdata.data() + 3 * std::accumulate(Dims.begin(), Dims.begin() + i, 0);
}

MatVec::MatVec(const Eval& eval, const ClusterBasis basis[], const double bodies[], const Cell cells[], const CSR& near, const CSR& far, const CellComm comm[], int64_t levels) :
  EvalFunc(&eval), Basis(basis), Bodies(bodies), Cells(cells), Near(&near), Far(&far), Comm(comm), Levels(levels) {
}

void MatVec::operator() (int64_t nrhs, std::complex<double> X[], int64_t ldX) const {
  int64_t lbegin = Comm[Levels].oLocal();
  int64_t llen = Comm[Levels].lenLocal();

  std::vector<std::vector<std::complex<double>>> rhsX(Levels + 1), rhsB(Levels + 1);
  std::vector<std::vector<std::complex<double>*>> rhsXptr(Levels + 1), rhsBptr(Levels + 1);
  std::vector<std::vector<std::pair<std::complex<double>*, int64_t>>> rhsXoptr(Levels + 1), rhsBoptr(Levels + 1);

  for (int64_t l = Levels; l >= 0; l--) {
    int64_t xlen = Comm[l].lenNeighbors();
    std::vector<int64_t> offsets(xlen + 1, 0);
    std::inclusive_scan(Basis[l].Dims.begin(), Basis[l].Dims.end(), offsets.begin() + 1);

    rhsX[l] = std::vector<std::complex<double>>(offsets[xlen] * nrhs, std::complex<double>(0., 0.));
    rhsB[l] = std::vector<std::complex<double>>(offsets[xlen] * nrhs, std::complex<double>(0., 0.));
    rhsXptr[l] = std::vector<std::complex<double>*>(xlen, nullptr);
    rhsBptr[l] = std::vector<std::complex<double>*>(xlen, nullptr);
    rhsXoptr[l] = std::vector<std::pair<std::complex<double>*, int64_t>>(xlen, std::make_pair(nullptr, 0));
    rhsBoptr[l] = std::vector<std::pair<std::complex<double>*, int64_t>>(xlen, std::make_pair(nullptr, 0));

    std::transform(offsets.begin(), offsets.begin() + xlen, rhsXptr[l].begin(), [&](const int64_t d) { return &rhsX[l][0] + d * nrhs; });
    std::transform(offsets.begin(), offsets.begin() + xlen, rhsBptr[l].begin(), [&](const int64_t d) { return &rhsB[l][0] + d * nrhs; });

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
          std::transform(offsets_child.begin(), offsets_child.begin() + clen, &rhsBoptr[l + 1][child], 
            [&](const int64_t d) { return std::make_pair(rhsBptr[l][i] + d, ldi); });
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
    int64_t lenI = nrhs * std::accumulate(&Basis[i].Dims[0], &Basis[i].Dims[xlen], 0);
    Comm[i].level_merge(rhsX[i].data(), lenI);
    Comm[i].neighbor_bcast(rhsX[i].data(), lens.data());
    Comm[i].dup_bcast(rhsX[i].data(), lenI);

    for (int64_t y = 0; y < iboxes; y++) {
      int64_t M = Basis[i].Dims[y + ibegin];
      int64_t N = Basis[i].DimsLr[y + ibegin];
      cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, nrhs, M, &one, Basis[i].V[y + ibegin], M, 
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
        mat_vec_reference(*EvalFunc, K, N, nrhs, rhsBoptr[i][y + ibegin].first, rhsBoptr[i][y + ibegin].second, 
          rhsXoptr[i][x].first, rhsXoptr[i][x].second, Basis[i].ske_at_i(y + ibegin), Basis[i].ske_at_i(x));
      }
      int64_t M = Basis[i].Dims[y + ibegin];
      cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, M, nrhs, K, &one, Basis[i].V[y + ibegin], M, 
        rhsBoptr[i][y + ibegin].first, rhsBoptr[i][y + ibegin].second, &one, rhsBptr[i][y + ibegin], M);
    }
  }

  int64_t gbegin = Comm[Levels].oGlobal();
  for (int64_t y = 0; y < llen; y++)
    for (int64_t yx = Near->RowIndex[y + gbegin]; yx < Near->RowIndex[y + gbegin + 1]; yx++) {
      int64_t x = Near->ColIndex[yx];
      int64_t x_loc = Comm[Levels].iLocal(x);
      int64_t M = Cells[y + gbegin].Body[1] - Cells[y + gbegin].Body[0];
      int64_t N = Cells[x].Body[1] - Cells[x].Body[0];
      mat_vec_reference(*EvalFunc, M, N, nrhs, rhsBptr[Levels][y + lbegin], M, rhsXptr[Levels][x_loc], N, &Bodies[3 * Cells[y + gbegin].Body[0]], &Bodies[3 * Cells[x].Body[0]]);
    }

  Y = 0;
  for (int64_t i = 0; i < llen; i++) {
    int64_t M = Basis[Levels].Dims[lbegin + i];
    memcpy2d(&X[Y], rhsBptr[Levels][lbegin + i], M, nrhs, ldX, M);
    Y = Y + M;
  }
}

