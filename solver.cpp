
#include <solver.hpp>
#include <build_tree.hpp>
#include <comm.hpp>
#include <kernel.hpp>

#include <cblas.h>
#include <lapacke.h>
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

extern void zomatcopy(char, int64_t, int64_t, const std::complex<double>*, int64_t, std::complex<double>*, int64_t);

void compute_capture(int64_t M, const int64_t N[], int64_t lenA, const std::complex<double>* A[], const int64_t LDA[], std::complex<double> C[], int64_t LDC) {
  constexpr int64_t block_size = 1 << 11;
  if (M > 0) {
    int64_t K = std::max(M, block_size), B2 = K + M;
    std::vector<std::complex<double>> B(M * B2, 0.), TAU(M);
    std::complex<double> zero(0., 0.);

    lapack_complex_double* Bptr = reinterpret_cast<lapack_complex_double*>(&B[0]);
    lapack_complex_double* Cptr = reinterpret_cast<lapack_complex_double*>(C);
    lapack_complex_double* Tptr = reinterpret_cast<lapack_complex_double*>(&TAU[0]);
    lapack_complex_double* Zero = reinterpret_cast<lapack_complex_double*>(&zero);

    zomatcopy('N', M, M, C, LDC, &B[0], B2);
    int64_t loc = 0;
    for (int64_t i = 0; i < lenA; i++) {
      int64_t loc_i = 0;
      while(loc_i < N[i]) {
        int64_t len = std::min(N[i] - loc_i, K - loc);
        zomatcopy('T', M, len, A[i], LDA[i], &B[M + loc], B2);
        loc_i = loc_i + len;
        loc = loc + len;
        if (loc == K) {
          LAPACKE_zgeqrf(LAPACK_COL_MAJOR, M + K, M, Bptr, B2, Tptr);
          LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, *Zero, *Zero, &Bptr[1], B2);
          loc = 0;
        }
      }
    }

    if (loc > 0)
      LAPACKE_zgeqrf(LAPACK_COL_MAJOR, M + loc, M, Bptr, B2, Tptr);
    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'U', M, M, Bptr, B2, Cptr, LDC);
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, *Zero, *Zero, &Cptr[1], LDC);
  }
}

void UlvSolver::preCompressA2(ClusterBasis& basis, const CellComm& comm) {

}
