
#include <solver.hpp>
#include <basis.hpp>
#include <build_tree.hpp>
#include <comm.hpp>
#include <kernel.hpp>

#include <mkl.h>
#include <algorithm>
#include <numeric>
#include <set>

UlvSolver::UlvSolver(const int64_t Dims[], const CSR& csr, const CellComm& comm) {
  int64_t xlen = comm.lenNeighbors();
  blocksOnRow = std::vector<int64_t>(xlen);
  elementsOnRow = std::vector<int64_t>(xlen);
  ARows = std::vector<int64_t>(xlen + 1);
  ARows[0] = 0;

  std::for_each(blocksOnRow.begin(), blocksOnRow.end(), 
    [&](int64_t& y) { int64_t i = comm.iGlobal(std::distance(&blocksOnRow[0], &y)); y = csr.RowIndex[i + 1] - csr.RowIndex[i]; });
  std::inclusive_scan(blocksOnRow.begin(), blocksOnRow.end(), ARows.begin() + 1);

  M = std::vector<int64_t>(ARows[xlen]);
  N = std::vector<int64_t>(ARows[xlen]);
  A = std::vector<const std::complex<double>*>(ARows[xlen]);
  ACols = std::vector<int64_t>(ARows[xlen]);
  
  int64_t ylocal = comm.oLocal();
  int64_t nodes = comm.lenLocal();

  for (int64_t i = 0; i < xlen; i++) {
    int64_t y = comm.iGlobal(i);
    std::copy(&csr.ColIndex[csr.RowIndex[y]], &csr.ColIndex[csr.RowIndex[y + 1]], &ACols[ARows[i]]);
    std::fill(&M[ARows[i]], &M[ARows[i + 1]], Dims[i]);
  }

  std::transform(&ACols[ARows[ylocal]], &ACols[ARows[ylocal + nodes]], &N[ARows[ylocal]],
    [&](int64_t col) { return Dims[comm.iLocal(col)]; });
  comm.neighbor_bcast(&N[0], &blocksOnRow[0]);
  comm.dup_bcast(&N[0], ARows[xlen]);
  RankM = std::vector<int64_t>(M.begin(), M.end());
  RankN = std::vector<int64_t>(N.begin(), N.end());

  std::vector<int64_t> Asizes(ARows[xlen]), Aoffsets(ARows[xlen] + 1);
  std::transform(M.begin(), M.end(), N.begin(), Asizes.begin(), [](int64_t m, int64_t n) { return m * n; });
  std::inclusive_scan(Asizes.begin(), Asizes.end(), Aoffsets.begin() + 1);
  Aoffsets[0] = 0;
  Adata = std::vector<std::complex<double>>(Aoffsets.back(), std::complex<double>(0., 0.));
  std::transform(Aoffsets.begin(), Aoffsets.begin() + ARows[xlen], A.begin(), [&](const int64_t d) { return &Adata[d]; });
  
  for (int64_t i = 0; i < xlen; i++)
    elementsOnRow[i] = std::reduce(&Asizes[ARows[i]], &Asizes[ARows[i + 1]]);

  CM = std::vector<int64_t>();
  CN = std::vector<int64_t>();
  CRows = std::vector<int64_t>(nodes + 1);
  CCols = std::vector<int64_t>();
  CRows[0] = 0;

  for (int64_t y = 0; y < nodes; y++) {
    const int64_t* ycols = &ACols[0] + ARows[y + ylocal];
    const int64_t* ycols_end = &ACols[0] + ARows[y + ylocal + 1];
    std::set<std::pair<int64_t, int64_t>> fills_kx;
    for (int64_t yk = ARows[y + ylocal]; yk < ARows[y + ylocal + 1]; yk++) {
      int64_t k = comm.iLocal(ACols[yk]);
      for (int64_t kx = ARows[k]; kx < ARows[k + 1]; kx++)
        if (ycols_end == std::find(ycols, ycols_end, ACols[kx]))
          fills_kx.insert(std::make_pair(ACols[kx], N[kx]));
    }

    CRows[y + 1] = CRows[y] + fills_kx.size();
    CCols.resize(CRows[y + 1]);
    CM.resize(CRows[y + 1], Dims[y + ylocal]);
    CN.resize(CRows[y + 1]);
    std::transform(fills_kx.begin(), fills_kx.end(), &CCols[CRows[y]], [&](std::pair<int64_t, int64_t> kx) { return kx.first; });
    std::transform(fills_kx.begin(), fills_kx.end(), &CN[CRows[y]], [&](std::pair<int64_t, int64_t> kx) { return kx.second; });
  }

  CRankM = std::vector<int64_t>(CM.begin(), CM.end());
  CRankN = std::vector<int64_t>(CN.begin(), CN.end());
  std::vector<int64_t> Csizes(CRows[nodes]), Coffsets(CRows[nodes] + 1);
  std::transform(CM.begin(), CM.end(), CN.begin(), Csizes.begin(), [](int64_t m, int64_t n) { return m * n; });
  std::inclusive_scan(Csizes.begin(), Csizes.end(), Coffsets.begin() + 1);
  Coffsets[0] = 0;
  C = std::vector<std::complex<double>*>(CRows[nodes]);
  Cdata = std::vector<std::complex<double>>(Coffsets.back(), std::complex<double>(0., 0.));
  std::transform(Coffsets.begin(), Coffsets.begin() + CRows[nodes], C.begin(), [&](const int64_t d) { return &Cdata[d]; });
}

void UlvSolver::loadDataLeaf(const MatrixAccessor& eval, const Cell cells[], const double bodies[], const CellComm& comm) {
  int64_t xlen = comm.lenNeighbors();
  for (int64_t i = 0; i < xlen; i++) {
    int64_t y = comm.iGlobal(i);
    for (int64_t yx = ARows[i]; yx < ARows[i + 1]; yx++) {
      int64_t x = ACols[yx];
      int64_t m = cells[y].Body[1] - cells[y].Body[0];
      int64_t n = cells[x].Body[1] - cells[x].Body[0];
      const double* Ibodies = &bodies[3 * cells[y].Body[0]];
      const double* Jbodies = &bodies[3 * cells[x].Body[0]];
      gen_matrix(eval, m, n, Ibodies, Jbodies, const_cast<std::complex<double>*>(A[yx]), m);
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
  constexpr int64_t batch_size = 8;
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

  for (int64_t i = 0; i < nodes; i++) {
    int64_t m = basis.Dims[i + ibegin];
    std::fill(basis.R[i + ibegin], basis.R[i + ibegin] + m * m, std::complex<double>(0., 0.));

    std::vector<const std::complex<double>*> c(&C[CRows[i]], &C[CRows[i + 1]]);
    std::vector<int64_t> cn(&CN[CRows[i]], &CN[CRows[i + 1]]);
    std::vector<int64_t> cm(&CM[CRows[i]], &CM[CRows[i + 1]]);
    if (CRows[i] < CRows[i + 1])
      captureA(m, &cn[0], CRows[i + 1] - CRows[i], &c[0], &cm[0], basis.R[i + ibegin], m);

    for (int64_t ij = CRows[i]; ij < CRows[i + 1]; ij++) {
      std::vector<const std::complex<double>*> a, b;
      std::vector<int64_t> am, bm;

      for (int64_t ik = ARows[i + ibegin]; ik < ARows[i + ibegin + 1]; ik++) {
        int64_t k = comm.iLocal(ACols[ik]);
        int64_t kj = std::distance(&ACols[0], std::find(&ACols[ARows[k]], &ACols[ARows[k + 1]], CCols[ij]));
        if (k != i + ibegin && kj != ARows[k + 1]) {
          a.emplace_back(A[ik]);
          b.emplace_back(A[kj]);
          am.emplace_back(M[ik]);
          bm.emplace_back(M[kj]);
        }
      }

      int64_t n = CN[ij];
      int64_t lenk = a.size();
      captureAmulB(m, n, &bm[0], lenk, &a[0], &am[0], &b[0], &bm[0], basis.R[i + ibegin], m);
    }
  }
  
  basis.recompressR(epi, comm);
}
