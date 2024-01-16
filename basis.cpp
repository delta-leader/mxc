
#include <basis.hpp>
#include <build_tree.hpp>
#include <comm.hpp>
#include <linalg.hpp>

#include <cblas.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <tuple>

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
      if ((x < y && uplo != 'L') || (x > y && uplo != 'U'))
        A[y + x * LDA] = alpha;
      else if (y == x)
        A[y + x * LDA] = beta;
    }
}

const double* Base::ske_at_i(int64_t i) const {
  return Mdata.data() + 3 * std::accumulate(Dims.begin(), Dims.begin() + i, 0);
}

void Base::mulUcLeft(int64_t i, int64_t N, std::complex<double>* A, int64_t LDA) const {
  int64_t M = Dims[i], K = DimsLr[i];
  std::vector<std::complex<double>> B(N * K);
  std::complex<double> one(1., 0.), zero(0., 0.), minus_one(-1., 0.);
  cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, K, N, M, &one, Uo[i], M, A, LDA, &zero, &B[0], K);
  cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, &minus_one, Uo[i], M, &B[0], K, &one, A, LDA);
}

void Base::mulUcRight(int64_t i, int64_t M, std::complex<double>* A, int64_t LDA) const {
  int64_t N = Dims[i], K = DimsLr[i];
  std::vector<std::complex<double>> B(M * K);
  std::complex<double> one(1., 0.), zero(0., 0.), minus_one(-1., 0.);
  cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, K, N, &one, A, LDA, Uo[i], N, &zero, &B[0], M);
  cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, M, N, K, &minus_one, &B[0], M, Uo[i], N, &one, A, LDA);
}

MatVec::MatVec(const Eval& eval, const Base basis[], const double bodies[], const Cell cells[], const CSR& near, const CSR& far, const CellComm comm[], int64_t levels) :
  EvalFunc(&eval), Basis(basis), Bodies(bodies), Cells(cells), Near(&near), Far(&far), Comm(comm), Levels(levels) {
}

void MatVec::operator() (int64_t nrhs, std::complex<double> X[], int64_t ldX) const {
  const Eval& eval = *EvalFunc;
  const CSR& near = *Near;
  const CSR& far = *Far;

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

    for (int64_t j = 0; j < iboxes; j++) {
      int64_t M = Basis[i].Dims[j + ibegin];
      int64_t N = Basis[i].DimsLr[j + ibegin];
      std::complex<double> one(1., 0.), zero(0., 0.);
      cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, N, nrhs, M, &one, Basis[i].Uo[j + ibegin], M, rhsXptr[i][j + ibegin], M, &zero, rhsXoptr[i][j + ibegin].first, rhsXoptr[i][j + ibegin].second);
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

    for (int64_t y = 0; y < iboxes; y++)
      for (int64_t yx = far.RowIndex[y + gbegin]; yx < far.RowIndex[y + gbegin + 1]; yx++) {
        int64_t x = Comm[i].iLocal(far.ColIndex[yx]);
        int64_t M = Basis[i].DimsLr[y + ibegin];
        int64_t N = Basis[i].DimsLr[x];

        std::vector<std::complex<double>> TMPX(N * nrhs, std::complex<double>(0., 0.));
        std::vector<std::complex<double>> TMPB(M * nrhs, std::complex<double>(0., 0.));
        std::complex<double> one(1., 0.), zero(0., 0.);
        cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, N, nrhs, N, &one, Basis[i].R[x], Basis[i].Dims[x], rhsXoptr[i][x].first, rhsXoptr[i][x].second, &zero, &TMPX[0], N);
        mat_vec_reference(eval, M, N, nrhs, &TMPB[0], M, &TMPX[0], N, Basis[i].ske_at_i(y + ibegin), Basis[i].ske_at_i(x));
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, nrhs, M, &one, Basis[i].R[y + ibegin], Basis[i].Dims[y + ibegin], &TMPB[0], M, &one, rhsBoptr[i][y + ibegin].first, rhsBoptr[i][y + ibegin].second);
      }
  }
  
  for (int64_t i = 1; i <= Levels; i++) {
    int64_t ibegin = Comm[i].oLocal();
    int64_t iboxes = Comm[i].lenLocal();
    for (int64_t j = 0; j < iboxes; j++) {
      int64_t M = Basis[i].Dims[j + ibegin];
      int64_t N = Basis[i].DimsLr[j + ibegin];
      std::complex<double> one(1., 0.);
      cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, nrhs, N, &one, Basis[i].Uo[j + ibegin], M, rhsBoptr[i][j + ibegin].first, rhsBoptr[i][j + ibegin].second, &one, rhsBptr[i][j + ibegin], M);
    }
  }

  int64_t gbegin = Comm[Levels].oGlobal();
  for (int64_t y = 0; y < llen; y++)
    for (int64_t yx = near.RowIndex[y + gbegin]; yx < near.RowIndex[y + gbegin + 1]; yx++) {
      int64_t x = near.ColIndex[yx];
      int64_t x_loc = Comm[Levels].iLocal(x);
      int64_t M = Cells[y + gbegin].Body[1] - Cells[y + gbegin].Body[0];
      int64_t N = Cells[x].Body[1] - Cells[x].Body[0];
      mat_vec_reference(eval, M, N, nrhs, rhsBptr[Levels][y + lbegin], M, rhsXptr[Levels][x_loc], N, &Bodies[3 * Cells[y + gbegin].Body[0]], &Bodies[3 * Cells[x].Body[0]]);
    }

  Y = 0;
  for (int64_t i = 0; i < llen; i++) {
    int64_t M = Basis[Levels].Dims[lbegin + i];
    memcpy2d(&X[Y], rhsBptr[Levels][lbegin + i], M, nrhs, ldX, M);
    Y = Y + M;
  }
}

void buildBasis(const Eval& eval, double epi, Base basis[], const Cell* cells, const CSR& rel_near, int64_t levels, const CellComm* comm, const double* bodies, int64_t nbodies) {

  for (int64_t l = levels; l >= 0; l--) {
    int64_t xlen = comm[l].lenNeighbors();
    int64_t ibegin = comm[l].oLocal();
    int64_t nodes = comm[l].lenLocal();
    basis[l].Dims = std::vector<int64_t>(xlen, 0);
    basis[l].DimsLr = std::vector<int64_t>(xlen, 0);
    basis[l].Uo = std::vector<std::complex<double>*>(xlen);
    basis[l].R = std::vector<std::complex<double>*>(xlen);
    std::vector<std::tuple<int64_t, int64_t, int64_t>> celli(nodes);

    for (int64_t i = 0; i < nodes; i++) {
      int64_t gi = comm[l].iGlobal(i + ibegin);
      int64_t child = l < levels ? comm[l + 1].iLocal(cells[gi].Child[0]) : -1;
      int64_t clen = cells[gi].Child[1] - cells[gi].Child[0];
      celli[i] = std::make_tuple(gi, child, clen);

      int64_t dim = l < levels ? 
        std::accumulate(&basis[l + 1].DimsLr[child], &basis[l + 1].DimsLr[child + clen], 0) : (cells[gi].Body[1] - cells[gi].Body[0]);
      basis[l].Dims[i + ibegin] = dim;
    }

    const std::vector<int64_t> ones(xlen, 1);
    comm[l].neighbor_bcast(basis[l].Dims.data(), ones.data());
    comm[l].dup_bcast(basis[l].Dims.data(), xlen);

    std::vector<int64_t> Usizes(xlen), Uoffsets(xlen + 1);
    std::transform(basis[l].Dims.begin(), basis[l].Dims.end(), Usizes.begin(), [](const int64_t d) { return d * d; });
    std::inclusive_scan(Usizes.begin(), Usizes.end(), Uoffsets.begin() + 1);
    Uoffsets[0] = 0;
    basis[l].Udata = std::vector<std::complex<double>>(Uoffsets[xlen], std::complex<double>(0., 0.));
    basis[l].Rdata = std::vector<std::complex<double>>(Uoffsets[xlen], std::complex<double>(0., 0.));
    std::transform(Uoffsets.begin(), Uoffsets.end(), basis[l].Uo.begin(), [&](const int64_t d) { return &basis[l].Udata[d]; });
    std::transform(Uoffsets.begin(), Uoffsets.end(), basis[l].R.begin(), [&](const int64_t d) { return &basis[l].Rdata[d]; });

    std::vector<int64_t> Msizes(xlen), Moffsets(xlen + 1);
    std::transform(basis[l].Dims.begin(), basis[l].Dims.end(), Msizes.begin(), [](const int64_t d) { return 3 * d; });
    std::inclusive_scan(Msizes.begin(), Msizes.end(), Moffsets.begin() + 1);
    Moffsets[0] = 0;
    basis[l].Mdata = std::vector<double>(Moffsets[xlen], 0.);

    for (int64_t i = 0; i < nodes; i++) {
      int64_t dim = basis[l].Dims[i + ibegin];
      std::complex<double>* matrixU = &basis[l].Udata[Uoffsets[i + ibegin]];
      std::complex<double>* matrixR = &basis[l].Rdata[Uoffsets[i + ibegin]];
      double* ske = &basis[l].Mdata[Moffsets[i + ibegin]];

      int64_t ci = std::get<0>(celli[i]);
      std::vector<const double*> remote;
      std::vector<int64_t> lens;

      int64_t loc = 0;
      for (int64_t c = rel_near.RowIndex[ci]; c < rel_near.RowIndex[ci + 1]; c++) {
        int64_t cj = rel_near.ColIndex[c];
        int64_t len = cells[cj].Body[0] - loc;
        if (len > 0) {
          remote.emplace_back(&bodies[loc * 3]);
          lens.emplace_back(len);
        }
        loc = cells[cj].Body[1];
      }
      if (loc < nbodies) {
        remote.emplace_back(&bodies[loc * 3]);
        lens.emplace_back(nbodies - loc);
      }

      int64_t childi = std::get<1>(celli[i]);
      int64_t clen = std::get<2>(celli[i]);
      if (l < levels)
        for (int64_t j = 0; j < clen; j++) {
          int64_t offset = std::accumulate(&basis[l + 1].DimsLr[childi], &basis[l + 1].DimsLr[childi + j], 0);
          int64_t len = basis[l + 1].DimsLr[childi + j];
          memcpy2d(&matrixU[offset * (dim + 1)], basis[l + 1].R[childi + j], len, len, dim, basis[l + 1].Dims[childi + j]);
          const double* mbegin = basis[l + 1].ske_at_i(childi + j);
          std::copy(mbegin, &mbegin[len * 3], &ske[offset * 3]);
        }
      else {
        matrixLaset('F', dim, dim, std::complex<double>(0., 0.), std::complex<double>(1., 0.), matrixU, dim);
        std::copy(&bodies[3 * cells[ci].Body[0]], &bodies[3 * cells[ci].Body[1]], ske);
      }
      
      compute_AallT(eval, dim, ske, remote.size(), &lens[0], &remote[0], matrixR, dim);
      int64_t rank = remote.size() > 0 ? compute_basis(epi, dim, matrixU, dim, matrixR, dim, ske) : 0;
      basis[l].DimsLr[i + ibegin] = rank;
    }

    comm[l].neighbor_bcast(basis[l].DimsLr.data(), ones.data());
    comm[l].dup_bcast(basis[l].DimsLr.data(), xlen);
    comm[l].neighbor_bcast(basis[l].Mdata.data(), Msizes.data());
    comm[l].dup_bcast(basis[l].Mdata.data(), Moffsets[xlen]);
    comm[l].neighbor_bcast(basis[l].Udata.data(), Usizes.data());
    comm[l].dup_bcast(basis[l].Udata.data(), Uoffsets[xlen]);
    comm[l].neighbor_bcast(basis[l].Rdata.data(), Usizes.data());
    comm[l].dup_bcast(basis[l].Rdata.data(), Uoffsets[xlen]);
  }
}


void solveRelErr(double* err_out, const std::complex<double>* X, const std::complex<double>* ref, int64_t lenX) {
  double err[2] = { 0., 0. };
  for (int64_t i = 0; i < lenX; i++) {
    std::complex<double> diff = X[i] - ref[i];
    err[0] = err[0] + (diff.real() * diff.real());
    err[1] = err[1] + (ref[i].real() * ref[i].real());
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  *err_out = std::sqrt(err[0] / err[1]);
}

