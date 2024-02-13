
#include <solver.hpp>
#include <build_tree.hpp>
#include <comm.hpp>

#include <algorithm>
#include <numeric>
#include <set>

BlockSparseMatrix::BlockSparseMatrix(const int64_t Dims[], const CSR& csr, const CellComm& comm) {
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

  std::vector<int64_t> Asizes(ARows[xlen]), Aoffsets(ARows[xlen] + 1);
  std::transform(M.begin(), M.end(), N.begin(), Asizes.begin(), [](int64_t m, int64_t n) { return m * n; });
  std::inclusive_scan(Asizes.begin(), Asizes.end(), Aoffsets.begin() + 1);
  Aoffsets[0] = 0;
  Adata = std::vector<std::complex<double>>(Aoffsets.back(), std::complex<double>(0., 0.));
  std::transform(Aoffsets.begin(), Aoffsets.begin() + ARows[xlen], A.begin(), [&](const int64_t d) { return &Adata[d]; });
  
  for (int64_t i = 0; i < xlen; i++)
    elementsOnRow[i] = std::reduce(&Asizes[ARows[i]], &Asizes[ARows[i + 1]]);

  FM = std::vector<int64_t>();
  FN = std::vector<int64_t>();
  FRows = std::vector<int64_t>(nodes + 1);
  FCols = std::vector<int64_t>();
  FRows[0] = 0;

  for (int64_t y = 0; y < nodes; y++) {
    const int64_t* ycols = &ACols[0] + ARows[y + ylocal];
    const int64_t* ycols_end = &ACols[0] + ARows[y + ylocal + 1];
    std::set<int64_t> fills_kx;
    for (int64_t yk = ARows[y + ylocal]; yk < ARows[y + ylocal + 1]; yk++) {
      int64_t k = comm.iLocal(ACols[yk]);
      for (int64_t kx = ARows[k]; kx < ARows[k + 1]; kx++)
        if (ycols_end == std::find(ycols, ycols_end, ACols[kx]))
          fills_kx.insert(kx);
    }

    FRows[y + 1] = FRows[y] + fills_kx.size();
    FCols.resize(FRows[y + 1]);
    FM.resize(FRows[y + 1], Dims[y + ylocal]);
    FN.resize(FRows[y + 1]);
    std::transform(fills_kx.begin(), fills_kx.end(), &FCols[FRows[y]], [&](int64_t kx) { return ACols[kx]; });
    std::transform(fills_kx.begin(), fills_kx.end(), &FN[FRows[y]], [&](int64_t kx) { return N[kx]; });
  }

  std::vector<int64_t> Fsizes(FRows[nodes]), Foffsets(FRows[nodes] + 1);
  std::transform(FM.begin(), FM.end(), FN.begin(), Fsizes.begin(), [](int64_t m, int64_t n) { return m * n; });
  std::inclusive_scan(Fsizes.begin(), Fsizes.end(), Foffsets.begin() + 1);
  Foffsets[0] = 0;
  F = std::vector<const std::complex<double>*>(FRows[nodes]);
  Fdata = std::vector<std::complex<double>>(Foffsets.back(), std::complex<double>(0., 0.));
  std::transform(Foffsets.begin(), Foffsets.begin() + FRows[nodes], F.begin(), [&](const int64_t d) { return &Fdata[d]; });
}
