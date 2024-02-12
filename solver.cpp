
#include <solver.hpp>
#include <build_tree.hpp>
#include <comm.hpp>

#include <algorithm>
#include <numeric>

BlockSparseMatrix::BlockSparseMatrix(const int64_t Dims[], const CSR& csr, const CellComm& comm) {
  int64_t xlen = comm.lenNeighbors();
  blocksOnRow = std::vector<int64_t>(xlen);
  elementsOnRow = std::vector<int64_t>(xlen);

  std::vector<int64_t> sumBlocks(xlen + 1);
  std::for_each(blocksOnRow.begin(), blocksOnRow.end(), 
    [&](int64_t& y) { int64_t i = comm.iGlobal(std::distance(&blocksOnRow[0], &y)); y = csr.RowIndex[i + 1] - csr.RowIndex[i]; });
  std::inclusive_scan(blocksOnRow.begin(), blocksOnRow.end(), sumBlocks.begin() + 1);
  sumBlocks[0] = 0;

  M = std::vector<int64_t>(sumBlocks[xlen]);
  N = std::vector<int64_t>(sumBlocks[xlen]);
  A = std::vector<const std::complex<double>*>(sumBlocks[xlen]);

  int64_t ybegin = comm.oGlobal();
  int64_t ylocal = comm.oLocal();
  int64_t nodes = comm.lenLocal();

  for (int64_t y = 0; y < nodes; y++) {
    int64_t m = Dims[y + ylocal];
    for (int64_t yx = csr.RowIndex[y + ybegin]; yx < csr.RowIndex[y + ybegin + 1]; yx++) {
      int64_t x = csr.ColIndex[yx];
      int64_t n = Dims[comm.iLocal(x)];
      int64_t k = sumBlocks[y + ylocal] + yx - csr.RowIndex[y + ybegin];
      M[k] = m;
      N[k] = n;
    }
  }

  comm.neighbor_bcast(&M[0], &blocksOnRow[0]);
  comm.neighbor_bcast(&N[0], &blocksOnRow[0]);
  comm.dup_bcast(&M[0], sumBlocks[xlen]);
  comm.dup_bcast(&N[0], sumBlocks[xlen]);

  std::vector<int64_t> Asizes(sumBlocks[xlen]), Aoffsets(sumBlocks[xlen] + 1);
  std::transform(M.begin(), M.end(), N.begin(), Asizes.begin(), [](int64_t m, int64_t n) { return m * n; });
  std::inclusive_scan(Asizes.begin(), Asizes.end(), Aoffsets.begin() + 1);
  Aoffsets[0] = 0;
  Adata = std::vector<std::complex<double>>(Aoffsets.back(), std::complex<double>(0., 0.));
  std::transform(Aoffsets.begin(), Aoffsets.begin() + sumBlocks[xlen], A.begin(), [&](const int64_t d) { return &Adata[d]; });
  
  for (int64_t i = 0; i < xlen; i++)
    elementsOnRow[i] = std::reduce(&Asizes[sumBlocks[i]], &Asizes[sumBlocks[i + 1]]);
}
