
#include <solver.hpp>
#include <basis.hpp>
#include <build_tree.hpp>
#include <comm.hpp>
#include <kernel.hpp>

#include <mkl.h>
#include <algorithm>
#include <numeric>
#include <map>

BlockSparseMatrix::BlockSparseMatrix(int64_t len, const std::pair<int64_t, int64_t> lil[], const std::pair<int64_t, int64_t> dim[], const CellComm& comm) {
  int64_t xlen = comm.lenNeighbors();
  int64_t ibegin = comm.oLocal();
  int64_t nodes = comm.lenLocal();

  blocksOnRow = std::vector<int64_t>(xlen);
  elementsOnRow = std::vector<int64_t>(xlen);
  const std::pair<int64_t, int64_t>* iter = lil;
  for (int64_t i = 0; i < nodes; i++) {
    int64_t cols = std::distance(iter, std::find_if_not(iter, &lil[len], [=](std::pair<int64_t, int64_t> l) { return l.first == i; }));
    blocksOnRow[i + ibegin] = cols;
    iter = &iter[cols];
  }

  const std::vector<int64_t> ones(xlen, 1);
  comm.neighbor_bcast(blocksOnRow.data(), ones.data());
  comm.dup_bcast(blocksOnRow.data(), xlen);

  RowIndex = std::vector<int64_t>(xlen + 1);
  RowIndex[0] = 0;
  std::inclusive_scan(blocksOnRow.begin(), blocksOnRow.end(), RowIndex.begin() + 1);

  M = std::vector<int64_t>(RowIndex[xlen]);
  N = std::vector<int64_t>(RowIndex[xlen]);
  ColIndex = std::vector<int64_t>(RowIndex[xlen]);

  std::transform(dim, &dim[len], &M[RowIndex[ibegin]], [](std::pair<int64_t, int64_t> d) { return d.first; });
  std::transform(dim, &dim[len], &N[RowIndex[ibegin]], [](std::pair<int64_t, int64_t> d) { return d.second; });
  std::transform(lil, &lil[len], &ColIndex[RowIndex[ibegin]], [](std::pair<int64_t, int64_t> l) { return l.second; });

  comm.neighbor_bcast(M.data(), blocksOnRow.data());
  comm.neighbor_bcast(N.data(), blocksOnRow.data());
  comm.neighbor_bcast(ColIndex.data(), blocksOnRow.data());
  comm.dup_bcast(M.data(), RowIndex[xlen]);
  comm.dup_bcast(N.data(), RowIndex[xlen]);
  comm.dup_bcast(ColIndex.data(), RowIndex[xlen]);

  RankM = std::vector<int64_t>(M.begin(), M.end());
  RankN = std::vector<int64_t>(N.begin(), N.end());
  std::vector<int64_t> DataSizes(RowIndex[xlen]);
  DataOffsets = std::vector<int64_t>(RowIndex[xlen] + 1);
  DataOffsets[0] = 0;

  std::transform(M.begin(), M.end(), N.begin(), DataSizes.begin(), [](int64_t m, int64_t n) { return m * n; });
  std::inclusive_scan(DataSizes.begin(), DataSizes.end(), DataOffsets.begin() + 1);
  Data = std::vector<std::complex<double>>(DataOffsets.back(), std::complex<double>(0., 0.));

  for (int64_t i = 0; i < xlen; i++)
    elementsOnRow[i] = std::reduce(&DataSizes[RowIndex[i]], &DataSizes[RowIndex[i + 1]]);
}

const std::complex<double>* BlockSparseMatrix::operator[](int64_t i) const {
  return &Data[DataOffsets[i]];
}

const std::complex<double>* BlockSparseMatrix::operator()(int64_t y, int64_t x) const {
  int64_t i = std::distance(&ColIndex[0], std::find(&ColIndex[RowIndex[y]], &ColIndex[RowIndex[y + 1]], x));
  return i < RowIndex[y + 1] ? &Data[DataOffsets[i]] : nullptr;
}

std::complex<double>* BlockSparseMatrix::operator[](int64_t i) {
  return &Data[DataOffsets[i]];
}

std::complex<double>* BlockSparseMatrix::operator()(int64_t y, int64_t x) {
  int64_t i = std::distance(&ColIndex[0], std::find(&ColIndex[RowIndex[y]], &ColIndex[RowIndex[y + 1]], x));
  return i < RowIndex[y + 1] ? &Data[DataOffsets[i]] : nullptr;
}

UlvSolver::UlvSolver(const int64_t Dims[], const CSR& csr, const CellComm& comm) {
  int64_t ibegin = comm.oLocal();
  int64_t ybegin = comm.oGlobal();
  int64_t nodes = comm.lenLocal();

  int64_t lenA = csr.RowIndex[ybegin + nodes] - csr.RowIndex[ybegin];
  std::vector<std::pair<int64_t, int64_t>> lil(lenA), dims(lenA);

  for (int64_t i = 0; i < nodes; i++) {
    int64_t xbegin = csr.RowIndex[ybegin + i];
    int64_t xend = csr.RowIndex[ybegin + i + 1];
    int64_t lbegin = xbegin - csr.RowIndex[ybegin];
    std::transform(&csr.ColIndex[xbegin], &csr.ColIndex[xend], &lil[lbegin], 
      [=](int64_t col) { return std::make_pair(i, col); });
    std::transform(&csr.ColIndex[xbegin], &csr.ColIndex[xend], &dims[lbegin], 
      [&](int64_t col) { return std::make_pair(Dims[i + ibegin], Dims[comm.iLocal(col)]); });
  }

  A = BlockSparseMatrix(lenA, &lil[0], &dims[0], comm);
  std::map<std::pair<int64_t, int64_t>, std::pair<int64_t, int64_t>> fills;

  for (int64_t y = 0; y < nodes; y++) {
    int64_t* ycols = &A.ColIndex[0] + A.RowIndex[y + ibegin];
    int64_t* ycols_end = &A.ColIndex[0] + A.RowIndex[y + ibegin + 1];
    int64_t d = std::distance(&A.ColIndex[0], std::find(ycols, ycols_end, ybegin + y));

    for (int64_t yk = d + 1; yk < A.RowIndex[y + ibegin + 1]; yk++) {
      int64_t k = comm.iLocal(A.ColIndex[yk]);
      for (int64_t kx = A.RowIndex[k]; kx < A.RowIndex[k + 1]; kx++)
        if (ycols_end == std::find(ycols, ycols_end, A.ColIndex[kx]))
          fills.emplace(std::make_pair(y, A.ColIndex[kx]), std::make_pair(yk, kx));
    }

    for (int64_t yk = A.RowIndex[y + ibegin]; yk < d; yk++) {
      int64_t k = comm.iLocal(A.ColIndex[yk]);
      for (int64_t kx = A.RowIndex[k]; kx < A.RowIndex[k + 1]; kx++)
        if (ycols_end == std::find(ycols, ycols_end, A.ColIndex[kx]))
          fills.emplace(std::make_pair(y, A.ColIndex[kx]), std::make_pair(yk, kx));
    }
  }

  int64_t lenC = fills.size();
  std::vector<std::pair<int64_t, int64_t>> lilC(lenC), dimsC(lenC);
  std::transform(fills.begin(), fills.end(), lilC.begin(), 
    [&](std::pair<std::pair<int64_t, int64_t>, std::pair<int64_t, int64_t>> f) { return f.first; });
  std::transform(fills.begin(), fills.end(), dimsC.begin(), 
    [&](std::pair<std::pair<int64_t, int64_t>, std::pair<int64_t, int64_t>> f) { return std::make_pair(A.M[f.second.first], A.N[f.second.second]); });

  C = BlockSparseMatrix(lenC, &lilC[0], &dimsC[0], comm);
  Ck = std::vector<int64_t>(C.RowIndex.back());
  std::transform(fills.begin(), fills.end(), &Ck[C.RowIndex[ibegin]], 
    [&](std::pair<std::pair<int64_t, int64_t>, std::pair<int64_t, int64_t>> f) { return A.ColIndex[f.second.first]; });
  comm.neighbor_bcast(Ck.data(), C.blocksOnRow.data());
  comm.dup_bcast(Ck.data(), C.RowIndex.back());
}

void UlvSolver::loadDataLeaf(const MatrixAccessor& eval, const Cell cells[], const double bodies[], const CellComm& comm) {
  int64_t xlen = comm.lenNeighbors();
  for (int64_t i = 0; i < xlen; i++) {
    int64_t y = comm.iGlobal(i);
    for (int64_t yx = A.RowIndex[i]; yx < A.RowIndex[i + 1]; yx++) {
      int64_t x = A.ColIndex[yx];
      int64_t m = cells[y].Body[1] - cells[y].Body[0];
      int64_t n = cells[x].Body[1] - cells[x].Body[0];
      const double* Ibodies = &bodies[3 * cells[y].Body[0]];
      const double* Jbodies = &bodies[3 * cells[x].Body[0]];
      gen_matrix(eval, m, n, Ibodies, Jbodies, A[yx], m);
    }
  }
}

void captureA(int64_t M, const int64_t N[], int64_t lenA, const std::complex<double>* A[], const int64_t LDA[], std::complex<double> C[], int64_t LDC) {
  constexpr int64_t block_size = 1 << 11;
  if (M > 0) {
    int64_t K = std::max(M, block_size), B2 = K + M;
    std::vector<std::complex<double>> B(M * B2, 0.), TAU(M);
    std::complex<double> zero(0., 0.);

    MKL_Zomatcopy('C', 'N', M, M, std::complex<double>(1., 0.), C, LDC, &B[0], B2);
    int64_t loc = 0;
    for (int64_t i = 0; i < lenA; i++) {
      int64_t loc_i = 0;
      while(loc_i < N[i]) {
        int64_t len = std::min(N[i] - loc_i, K - loc);
        MKL_Zomatcopy('C', 'T', M, len, std::complex<double>(1., 0.), A[i] + loc_i * LDA[i], LDA[i], &B[M + loc], B2);
        loc_i = loc_i + len;
        loc = loc + len;
        if (loc == K) {
          LAPACKE_zgeqrf(LAPACK_COL_MAJOR, B2, M, &B[0], B2, &TAU[0]);
          LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, zero, zero, &B[1], B2);
          loc = 0;
        }
      }
    }

    if (loc > 0)
      LAPACKE_zgeqrf(LAPACK_COL_MAJOR, M + loc, M, &B[0], B2, &TAU[0]);
    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'U', M, M, &B[0], B2, C, LDC);
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, zero, zero, &C[1], LDC);
  }
}

void captureAmulB(int64_t M, int64_t N, const int64_t K[], int64_t lenAB, const std::complex<double>* A[], const int64_t LDA[], const std::complex<double>* B[], const int64_t LDB[], std::complex<double> C[], int64_t LDC) {
  constexpr int64_t batch_size = 4;
  if (M > 0) {
    int64_t B2 = std::max(M, batch_size * N) + M;
    std::vector<std::complex<double>> Y(M * B2, 0.), TAU(M);
    std::complex<double> zero(0., 0.), one(1., 0.);

    MKL_Zomatcopy('C', 'N', M, M, one, C, LDC, &Y[0], B2);
    int64_t rem = lenAB % batch_size;
    if (rem > 0) {
      for (int64_t b = 0; b < rem; b++)
        cblas_zgemm(CblasColMajor, CblasTrans, CblasTrans, N, M, K[b], &one, B[b], LDB[b], A[b], LDA[b], &zero, &Y[M + b * N], B2);

      LAPACKE_zgeqrf(LAPACK_COL_MAJOR, M + rem * N, M, &Y[0], B2, &TAU[0]);
      LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, zero, zero, &Y[1], B2);
    }

    for (int64_t i = rem; i < lenAB; i += batch_size) {
      for (int64_t b = 0; b < batch_size; b++)
        cblas_zgemm(CblasColMajor, CblasTrans, CblasTrans, N, M, K[i + b], &one, B[i + b], LDB[i + b], A[i + b], LDA[i + b], &zero, &Y[M + b * N], B2);

      LAPACKE_zgeqrf(LAPACK_COL_MAJOR, M + batch_size * N, M, &Y[0], B2, &TAU[0]);
      LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, zero, zero, &Y[1], B2);
    }

    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'U', M, M, &Y[0], B2, C, LDC);
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, zero, zero, &C[1], LDC);
  }
}

/*void captureAmulB(int64_t M, int64_t N, const int64_t K[], int64_t lenAB, const std::complex<double>* A[], const int64_t LDA[], const std::complex<double>* B[], const int64_t LDB[], std::complex<double> C[], int64_t LDC) {
  if (M > 0) {
    int64_t B2 = N + M;
    std::vector<std::complex<double>> Y(M * B2, 0.), TAU(M);
    std::complex<double> zero(0., 0.), one(1., 0.);

    MKL_Zomatcopy('C', 'N', M, M, one, C, LDC, &Y[0], B2);
    for (int64_t b = 0; b < lenAB; b++)
      cblas_zgemm(CblasColMajor, CblasTrans, CblasTrans, N, M, K[b], &one, B[b], LDB[b], A[b], LDA[b], &one, &Y[M], B2);

    LAPACKE_zgeqrf(LAPACK_COL_MAJOR, B2, M, &Y[0], B2, &TAU[0]);
    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'U', M, M, &Y[0], B2, C, LDC);
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, zero, zero, &C[1], LDC);
  }
}*/

void UlvSolver::preCompressA2(double epi, ClusterBasis& basis, const CellComm& comm) {
  int64_t ibegin = comm.oLocal();
  int64_t nodes = comm.lenLocal();
  int64_t ybegin = comm.oGlobal();
  int64_t xlen = comm.lenNeighbors();

  int64_t lenC = C.RowIndex[ibegin + nodes] - C.RowIndex[ibegin];
  std::vector<const std::complex<double>*> Cptr(lenC);
  std::transform(&C.DataOffsets[C.RowIndex[ibegin]], &C.DataOffsets[C.RowIndex[ibegin + nodes]], 
    Cptr.begin(), [&](int64_t offset) { return &C.Data[offset]; });

  for (int64_t i = 0; i < nodes; i++) {
    int64_t m = basis.Dims[i + ibegin];
    std::fill(basis.R[i + ibegin], basis.R[i + ibegin] + m * m, std::complex<double>(0., 0.));

    int64_t offsetCi = C.RowIndex[i + ibegin];
    int64_t lenCi = C.RowIndex[i + ibegin + 1] - offsetCi;
    if (0 < lenCi)
      captureA(m, &C.N[offsetCi], lenCi, &Cptr[offsetCi - C.RowIndex[ibegin]], &C.M[offsetCi], basis.R[i + ibegin], m);
    for (int64_t ij = C.RowIndex[i + ibegin]; ij < C.RowIndex[i + ibegin + 1]; ij++) {
      std::vector<const std::complex<double>*> a, b;
      std::vector<int64_t> am, bm;

      for (int64_t ik = A.RowIndex[i + ibegin]; ik < A.RowIndex[i + ibegin + 1]; ik++) {
        int64_t k = comm.iLocal(A.ColIndex[ik]);
        int64_t kj = std::distance(&A.ColIndex[0], std::find(&A.ColIndex[A.RowIndex[k]], &A.ColIndex[A.RowIndex[k + 1]], C.ColIndex[ij]));
        if (k != i + ibegin && kj != A.RowIndex[k + 1]) {
          a.emplace_back(A[ik]);
          b.emplace_back(A[kj]);
          am.emplace_back(A.M[ik]);
          bm.emplace_back(A.M[kj]);
        }
      }
      captureAmulB(m, C.N[ij], &bm[0], a.size(), &a[0], &am[0], &b[0], &bm[0], basis.R[i + ibegin], m);
    }
  }
  
  basis.recompressR(epi, comm);
  for (int64_t i = 0; i < xlen; i++)
    for (int64_t ij = C.RowIndex[i]; ij < C.RowIndex[i + 1]; ij++) {
      int64_t j = comm.iLocal(C.ColIndex[ij]);
      if (ybegin <= Ck[ij] && Ck[ij] < ybegin + nodes) {
        C.RankM[ij] = basis.DimsLr[i];
        C.RankN[ij] = basis.DimsLr[j];
        // TODO
        //std::cout << comm.iGlobal(i) << ", " << C.ColIndex[ij] << ", " << Ck[ij] << std::endl;
      }
      else {
        int64_t M = C.M[ij], N = C.N[ij];
        std::complex<double>* Cptr = C[ij];
        std::fill(Cptr, &Cptr[M * N], std::complex<double>(0., 0.));
        C.RankM[ij] = 0;
        C.RankN[ij] = 0;
      }
    }

  comm.neighbor_reduce(C.Data.data(), C.elementsOnRow.data());
  comm.neighbor_reduce(C.RankM.data(), C.blocksOnRow.data());
  comm.neighbor_reduce(C.RankN.data(), C.blocksOnRow.data());
  comm.dup_bcast(C.Data.data(), C.DataOffsets.back());
  comm.dup_bcast(C.RankM.data(), C.RowIndex.back());
  comm.dup_bcast(C.RankN.data(), C.RowIndex.back());
}
