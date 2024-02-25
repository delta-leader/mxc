
#include <solver.hpp>
#include <basis.hpp>
#include <build_tree.hpp>
#include <comm.hpp>
#include <kernel.hpp>

#include <mkl.h>
#include <algorithm>
#include <numeric>
#include <map>

BlockSparseMatrix::BlockSparseMatrix(long long len, const std::pair<long long, long long> lil[], const std::pair<long long, long long> dim[], const CellComm& comm) {
  long long xlen = comm.lenNeighbors();
  long long nodes = comm.lenLocal();
  long long ibegin = comm.oLocal();
  long long ybegin = comm.oGlobal();

  LocalIndex = std::make_pair(ibegin, ibegin + nodes);
  blocksOnRow = std::vector<long long>(xlen);
  elementsOnRow = std::vector<long long>(xlen);
  DiagIndex = std::vector<long long>(xlen);
  const std::pair<long long, long long>* iter = lil;
  for (long long i = 0; i < nodes; i++) {
    long long cols = std::distance(iter, std::find_if_not(iter, &lil[len], [=](std::pair<long long, long long> l) { return l.first == i; }));
    long long diag = std::distance(iter, std::find_if(iter, &iter[cols], [=](std::pair<long long, long long> l) { return l.second == i + ybegin; }));
    blocksOnRow[i + ibegin] = cols;
    DiagIndex[i + ibegin] = diag;
    iter = &iter[cols];
  }

  const std::vector<long long> ones(xlen, 1);
  comm.neighbor_bcast(blocksOnRow.data(), ones.data());
  comm.neighbor_bcast(DiagIndex.data(), ones.data());

  RowIndex = std::vector<long long>(xlen + 1);
  RowIndex[0] = 0;
  std::inclusive_scan(blocksOnRow.begin(), blocksOnRow.end(), RowIndex.begin() + 1);
  std::transform(DiagIndex.begin(), DiagIndex.end(), RowIndex.begin(), DiagIndex.begin(), std::plus<long long>());

  ColIndex = std::vector<long long>(RowIndex[xlen]);
  ColIndexLocal = std::vector<long long>(RowIndex[xlen]);
  Dims = std::vector<std::pair<long long, long long>>(RowIndex[xlen]);
  DimsLr = std::vector<std::array<long long, 4>>(RowIndex[xlen]);

  std::copy(dim, &dim[len], &Dims[RowIndex[ibegin]]);
  std::transform(lil, &lil[len], &ColIndex[RowIndex[ibegin]], [](std::pair<long long, long long> l) { return l.second; });

  long long* DimsPtr = reinterpret_cast<long long*>(Dims.data());
  std::vector<long long> blocks2(xlen);
  std::transform(blocksOnRow.begin(), blocksOnRow.end(), blocks2.begin(), [](long long b) { return b + b; });
  comm.neighbor_bcast(DimsPtr, blocks2.data());
  comm.neighbor_bcast(ColIndex.data(), blocksOnRow.data());
  std::transform(ColIndex.begin(), ColIndex.end(), ColIndexLocal.begin(), [&](long long col) { return comm.iLocal(col); });

  std::vector<long long> DataSizes(RowIndex[xlen]);
  DataOffsets = std::vector<long long>(RowIndex[xlen] + 1);
  DataOffsets[0] = 0;

  std::transform(Dims.begin(), Dims.end(), DataSizes.begin(), [](std::pair<long long, long long> d) { return d.first * d.second; });
  std::inclusive_scan(DataSizes.begin(), DataSizes.end(), DataOffsets.begin() + 1);
  Data = std::vector<std::complex<double>>(DataOffsets.back(), std::complex<double>(0., 0.));

  for (long long i = 0; i < xlen; i++)
    elementsOnRow[i] = std::reduce(&DataSizes[RowIndex[i]], &DataSizes[RowIndex[i + 1]]);
}

const std::complex<double>* BlockSparseMatrix::operator[](long long i) const {
  return &Data[DataOffsets[i]];
}

std::complex<double>* BlockSparseMatrix::operator[](long long i) {
  return &Data[DataOffsets[i]];
}

long long BlockSparseMatrix::operator()(long long y, long long x) const {
  long long i = std::distance(&ColIndex[0], std::find(&ColIndex[RowIndex[y]], &ColIndex[RowIndex[y + 1]], x));
  return i < RowIndex[y + 1] ? i : -1;
}

long long BlockSparseMatrix::ijLower(long long i, long long* ij) const {
  const long long* cols = &ColIndexLocal[RowIndex[i]];
  const long long* cols_end = &ColIndexLocal[RowIndex[i + 1]];
  long long diag = DiagIndex[i];

  auto col_in_local = [=](long long col) { return LocalIndex.first <= col && col < LocalIndex.second; };
  long long local = std::distance(&ColIndexLocal[0], std::find_if(cols, cols_end, col_in_local));
  long long local_end = std::distance(&ColIndexLocal[0], std::find_if_not(&ColIndexLocal[local], cols_end, col_in_local));

  std::vector<long long> lis(RowIndex[i + 1] - RowIndex[i]);
  std::iota(lis.begin(), lis.end(), RowIndex[i]);

  long long* start = ij;
  if (i < LocalIndex.first)
    ij = std::copy_if(lis.begin(), lis.end(), ij, [&](long long k) { return 0 <= ColIndexLocal[k] && k < diag; });
  else if (LocalIndex.first <= i && i < LocalIndex.second)
    ij = std::copy_if(lis.begin(), lis.end(), ij, [&](long long k) { return 0 <= ColIndexLocal[k] && (k < diag || local_end <= k); });
  else
    ij = std::copy_if(lis.begin(), lis.end(), ij, [&](long long k) { return 0 <= ColIndexLocal[k] && k < diag && (k < local || local_end <= k); });

  return std::distance(start, ij);
}

long long BlockSparseMatrix::ijUpper(long long i, long long* ij) const {
  const long long* cols = &ColIndexLocal[RowIndex[i]];
  const long long* cols_end = &ColIndexLocal[RowIndex[i + 1]];
  long long diag = DiagIndex[i];

  auto col_in_local = [=](long long col) { return LocalIndex.first <= col && col < LocalIndex.second; };
  long long local = std::distance(&ColIndexLocal[0], std::find_if(cols, cols_end, col_in_local));
  long long local_end = std::distance(&ColIndexLocal[0], std::find_if_not(&ColIndexLocal[local], cols_end, col_in_local));

  std::vector<long long> lis(RowIndex[i + 1] - RowIndex[i]);
  std::iota(lis.begin(), lis.end(), RowIndex[i]);

  long long* start = ij;
  if (i < LocalIndex.first)
    ij = std::copy_if(lis.begin(), lis.end(), ij, [&](long long k) { return 0 <= ColIndexLocal[k] && diag < k; });
  else if (LocalIndex.first <= i && i < LocalIndex.second)
    ij = std::copy_if(lis.begin(), lis.end(), ij, [&](long long k) { return 0 <= ColIndexLocal[k] && diag < k && k < local_end; });
  else
    ij = std::copy_if(lis.begin(), lis.end(), ij, [&](long long k) { return 0 <= ColIndexLocal[k] && (diag < k || (local <= k && k < local_end)); });

  return std::distance(start, ij);
}

UlvSolver::UlvSolver(const long long Dims[], const CSR& Near, const CSR& Far, const CellComm& comm) {
  long long ibegin = comm.oLocal();
  long long ybegin = comm.oGlobal();
  long long nodes = comm.lenLocal();
  long long xlen = comm.lenNeighbors();

  long long lenA = Near.RowIndex[ybegin + nodes] - Near.RowIndex[ybegin];
  std::vector<std::pair<long long, long long>> lil(lenA), dims(lenA);

  for (long long i = 0; i < nodes; i++) {
    long long xbegin = Near.RowIndex[ybegin + i];
    long long xend = Near.RowIndex[ybegin + i + 1];
    long long lbegin = xbegin - Near.RowIndex[ybegin];
    std::transform(&Near.ColIndex[xbegin], &Near.ColIndex[xend], &lil[lbegin], 
      [=](long long col) { return std::make_pair(i, col); });
    std::transform(&Near.ColIndex[xbegin], &Near.ColIndex[xend], &dims[lbegin], 
      [&](long long col) { return std::make_pair(Dims[i + ibegin], Dims[comm.iLocal(col)]); });
  }

  A = BlockSparseMatrix(lenA, &lil[0], &dims[0], comm);
  std::map<std::pair<long long, long long>, std::pair<long long, long long>> fills;

  for (long long y = 0; y < nodes; y++) {
    long long* ycols = &A.ColIndex[0] + A.RowIndex[y + ibegin];
    long long* ycols_end = &A.ColIndex[0] + A.RowIndex[y + ibegin + 1];
    long long dy = std::distance(&A.ColIndex[0], std::find(ycols, ycols_end, ybegin + y));

    for (long long yk = dy + 1; yk < A.RowIndex[y + ibegin + 1]; yk++) {
      long long k = A.ColIndexLocal[yk];
      for (long long kx = A.RowIndex[k]; kx < A.RowIndex[k + 1]; kx++)
        if (ycols_end == std::find(ycols, ycols_end, A.ColIndex[kx]))
          fills.emplace(std::make_pair(y, A.ColIndex[kx]), std::make_pair(yk, kx));
    }

    for (long long yk = A.RowIndex[y + ibegin]; yk < dy; yk++) {
      long long k = A.ColIndexLocal[yk];
      for (long long kx = A.RowIndex[k]; kx < A.RowIndex[k + 1]; kx++)
        if (ycols_end == std::find(ycols, ycols_end, A.ColIndex[kx]))
          fills.emplace(std::make_pair(y, A.ColIndex[kx]), std::make_pair(yk, kx));
    }

    for (long long yx = Far.RowIndex[y + ybegin]; yx < Far.RowIndex[y + ybegin + 1]; yx++) {
      long long x = comm.iLocal(Far.ColIndex[yx]);
      long long dx = std::distance(&A.ColIndex[0], std::find(&A.ColIndex[A.RowIndex[x]], &A.ColIndex[A.RowIndex[x + 1]], Far.ColIndex[yx]));
      fills.emplace(std::make_pair(y, Far.ColIndex[yx]), std::make_pair(dy, dx));
    }
  }

  long long lenC = fills.size();
  std::vector<std::pair<long long, long long>> lilC(lenC), dimsC(lenC);
  std::transform(fills.begin(), fills.end(), lilC.begin(), 
    [&](std::pair<std::pair<long long, long long>, std::pair<long long, long long>> f) { return f.first; });
  std::transform(fills.begin(), fills.end(), dimsC.begin(), 
    [&](std::pair<std::pair<long long, long long>, std::pair<long long, long long>> f) { return std::make_pair(A.Dims[f.second.first].first, A.Dims[f.second.second].second); });

  C = BlockSparseMatrix(lenC, &lilC[0], &dimsC[0], comm);
  Ck = std::vector<long long>(C.RowIndex.back());
  std::transform(fills.begin(), fills.end(), &Ck[C.RowIndex[ibegin]], 
    [&](std::pair<std::pair<long long, long long>, std::pair<long long, long long>> f) { return A.ColIndex[f.second.first]; });
  comm.neighbor_bcast(Ck.data(), C.blocksOnRow.data());

  Apiv = std::vector<std::vector<long long>>(xlen);
  for (long long i = 0; i < xlen; i++) {
    Apiv[i] = std::vector<long long>(Dims[i]);
    std::iota(Apiv[i].begin(), Apiv[i].end(), 1);
  }
}

void UlvSolver::loadDataLeaf(const MatrixAccessor& eval, const Cell cells[], const double bodies[], const CellComm& comm) {
  long long xlen = comm.lenNeighbors();
  for (long long i = 0; i < xlen; i++) {
    long long y = comm.iGlobal(i);
    for (long long yx = A.RowIndex[i]; yx < A.RowIndex[i + 1]; yx++) {
      long long x = A.ColIndex[yx];
      long long m = cells[y].Body[1] - cells[y].Body[0];
      long long n = cells[x].Body[1] - cells[x].Body[0];
      const double* Ibodies = &bodies[3 * cells[y].Body[0]];
      const double* Jbodies = &bodies[3 * cells[x].Body[0]];
      gen_matrix(eval, m, n, Ibodies, Jbodies, A[yx]);
    }
  }
}

void UlvSolver::loadDataInterNode(const Cell cells[], const UlvSolver& prev_matrix, const CellComm& prev_comm, const CellComm& comm) {
  long long ibegin = comm.oLocal();
  long long ybegin = comm.oGlobal();
  long long nodes = comm.lenLocal();
  long long lowerY = prev_comm.oGlobal();
  long long lowerZ = lowerY + prev_comm.lenLocal();

  std::complex<double> one(1., 0.);
  for (long long i = 0; i < nodes; i++) {
    long long ci = std::max(lowerY, cells[i + ybegin].Child[0]);
    long long clen = std::min(lowerZ, cells[i + ybegin].Child[1]) - ci;
    long long cybegin = prev_comm.iLocal(ci);
    long long cyend = cybegin + clen;

    for (long long ij = A.RowIndex[i + ibegin]; ij < A.RowIndex[i + ibegin + 1]; ij++) {
      long long j = A.ColIndex[ij];
      long long cxbegin = cells[j].Child[0];
      long long cxend = cells[j].Child[1];
      long long AM = A.Dims[ij].first;
      std::complex<double>* Aptr = A[ij];

      for (long long cy = cybegin; cy < cyend && 0 < AM; cy++)
        for (long long cx = cxbegin; cx < cxend; cx++) {
          long long lowA = prev_matrix.A(cy, cx);
          long long lowC = prev_matrix.C(cy, cx);
          if (0 <= lowA) {
            std::array<long long, 4> ADIM = prev_matrix.A.DimsLr[lowA];
            long long LDA = prev_matrix.A.Dims[lowA].first;
            long long offset = ADIM[2] + AM * ADIM[3];
            MKL_Zomatcopy('C', 'N', ADIM[0], ADIM[1], one, prev_matrix.A[lowA], LDA, &Aptr[offset], AM);
          }
          else if (0 <= lowC) {
            std::array<long long, 4> CDIM = prev_matrix.C.DimsLr[lowC];
            long long offset = CDIM[2] + AM * CDIM[3];
            MKL_Zomatcopy('C', 'N', CDIM[0], CDIM[1], one, prev_matrix.C[lowC], CDIM[0], &Aptr[offset], AM);
          }
        }
    }

    for (long long ij = C.RowIndex[i + ibegin]; ij < C.RowIndex[i + ibegin + 1]; ij++) {
      long long j = C.ColIndex[ij];
      long long cxbegin = cells[j].Child[0];
      long long cxend = cells[j].Child[1];
      long long CM = C.Dims[ij].first;
      std::complex<double>* Cptr = C[ij];

      for (long long cy = cybegin; cy < cyend && 0 < CM; cy++)
        for (long long cx = cxbegin; cx < cxend; cx++) {
          long long lowA = prev_matrix.A(cy, cx);
          long long lowC = prev_matrix.C(cy, cx);
          if (0 <= lowA) {
            std::array<long long, 4> ADIM = prev_matrix.A.DimsLr[lowA];
            long long LDA = prev_matrix.A.Dims[lowA].first;
            long long offset = ADIM[2] + CM * ADIM[3];
            MKL_Zomatcopy('C', 'N', ADIM[0], ADIM[1], one, prev_matrix.A[lowA], LDA, &Cptr[offset], CM);
          }
          else if (0 <= lowC) {
            std::array<long long, 4> CDIM = prev_matrix.C.DimsLr[lowC];
            long long offset = CDIM[2] + CM * CDIM[3];
            MKL_Zomatcopy('C', 'N', CDIM[0], CDIM[1], one, prev_matrix.C[lowC], CDIM[0], &Cptr[offset], CM);
          }
        }
    }
  }

  comm.level_merge(A.Data.data(), A.DataOffsets.back());
  comm.level_merge(C.Data.data(), C.DataOffsets.back());
  comm.neighbor_bcast(A.Data.data(), A.elementsOnRow.data());
  comm.neighbor_bcast(C.Data.data(), C.elementsOnRow.data());
}

void captureA(long long M, const long long N[], long long lenA, const std::complex<double>* A[], std::complex<double> C[]) {
  constexpr long long block_size = 1 << 11;
  if (0 < M && 0 < lenA) {
    long long K = std::max(M, block_size), B2 = K + M;
    std::vector<std::complex<double>> B(M * B2, 0.), TAU(M);
    std::complex<double> zero(0., 0.), one(1., 0.);

    MKL_Zomatcopy('C', 'N', M, M, one, C, M, &B[0], B2);
    long long loc = 0;
    for (long long i = 0; i < lenA; i++) {
      long long loc_i = 0;
      while(loc_i < N[i]) {
        long long len = std::min(N[i] - loc_i, K - loc);
        MKL_Zomatcopy('C', 'T', M, len, one, &(A[i])[loc_i * M], M, &B[M + loc], B2);
        loc_i = loc_i + len;
        loc = loc + len;
        if (loc == K) {
          LAPACKE_zgeqrf(LAPACK_COL_MAJOR, B2, M, &B[0], B2, &TAU[0]);
          LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, zero, zero, &B[1], B2);
          loc = 0;
        }
      }
    }

    if (0 < loc)
      LAPACKE_zgeqrf(LAPACK_COL_MAJOR, M + loc, M, &B[0], B2, &TAU[0]);
    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'U', M, M, &B[0], B2, C, M);
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, zero, zero, &C[1], M);
  }
}

void captureAmulB(long long M, long long N, const long long K[], long long lenAB, const std::complex<double>* A[], const std::complex<double>* B[], std::complex<double> C[]) {
  constexpr long long batch_size = 4;
  if (0 < M && 0 < lenAB) {
    long long B2 = std::max(M, batch_size * N) + M;
    std::vector<std::complex<double>> Y(M * B2, 0.), TAU(M);
    std::complex<double> zero(0., 0.), one(1., 0.);

    MKL_Zomatcopy('C', 'N', M, M, one, C, M, &Y[0], B2);
    long long rem = lenAB % batch_size;
    if (0 < rem) {
      for (long long b = 0; b < rem; b++)
        cblas_zgemm(CblasColMajor, CblasTrans, CblasTrans, N, M, K[b], &one, B[b], K[b], A[b], M, &zero, &Y[M + b * N], B2);

      LAPACKE_zgeqrf(LAPACK_COL_MAJOR, M + rem * N, M, &Y[0], B2, &TAU[0]);
      LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, zero, zero, &Y[1], B2);
    }

    for (long long i = rem; i < lenAB; i += batch_size) {
      for (long long b = 0; b < batch_size; b++)
        cblas_zgemm(CblasColMajor, CblasTrans, CblasTrans, N, M, K[i + b], &one, B[i + b], K[i + b], A[i + b], M, &zero, &Y[M + b * N], B2);

      LAPACKE_zgeqrf(LAPACK_COL_MAJOR, M + batch_size * N, M, &Y[0], B2, &TAU[0]);
      LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, zero, zero, &Y[1], B2);
    }

    LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'U', M, M, &Y[0], B2, C, M);
    LAPACKE_zlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, zero, zero, &C[1], M);
  }
}

void mulQhAQc(long long M, long long N, std::complex<double>* A, long long K1, const std::complex<double>* Ql, long long K2, const std::complex<double>* Qr) {
  if (0 < K1 && 0 < K2) {
    std::vector<std::complex<double>> B(M * K2);
    std::complex<double> zero(0., 0.), one(1., 0.);
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasTrans, K2, M, N, &one, Qr, N, A, M, &zero, &B[0], K2);
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasTrans, K1, K2, M, &one, Ql, M, &B[0], K2, &zero, A, K1);
    std::fill(&A[K1 * K2], &A[M * N], zero);
  }
}

void UlvSolver::preCompressA2(double epi, ClusterBasis& basis, const CellComm& comm) {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long ybegin = comm.oGlobal();
  long long xlen = comm.lenNeighbors();

  long long lenC = C.RowIndex[ibegin + nodes] - C.RowIndex[ibegin];
  std::vector<const std::complex<double>*> Cptr(lenC);
  std::transform(&C.DataOffsets[C.RowIndex[ibegin]], &C.DataOffsets[C.RowIndex[ibegin + nodes]], 
    Cptr.begin(), [&](long long offset) { return &C.Data[offset]; });

  for (long long i = 0; i < nodes; i++) {
    long long m = basis.Dims[i + ibegin];
    std::fill(basis.R[i + ibegin], basis.R[i + ibegin] + m * m, std::complex<double>(0., 0.));

    long long offsetCi = C.RowIndex[i + ibegin];
    long long lenCi = C.RowIndex[i + ibegin + 1] - offsetCi;
    std::vector<long long> CN(lenCi);
    std::transform(&C.Dims[offsetCi], &C.Dims[offsetCi + lenCi], CN.begin(), [](const std::pair<long long, long long>& d) { return d.second; });
    if (0 < lenCi)
      captureA(m, CN.data(), lenCi, &Cptr[offsetCi - C.RowIndex[ibegin]], basis.R[i + ibegin]);
    for (long long ij = 0; ij < lenCi; ij++) {
      std::vector<const std::complex<double>*> Aptr, Bptr;
      std::vector<long long> K;

      for (long long ik = A.RowIndex[i + ibegin]; ik < A.RowIndex[i + ibegin + 1]; ik++) {
        long long k = A.ColIndexLocal[ik];
        long long kj = std::distance(&A.ColIndex[0], std::find(&A.ColIndex[A.RowIndex[k]], &A.ColIndex[A.RowIndex[k + 1]], C.ColIndex[offsetCi + ij]));
        if (k != i + ibegin && kj != A.RowIndex[k + 1]) {
          Aptr.emplace_back(A[ik]);
          Bptr.emplace_back(A[kj]);
          K.emplace_back(A.Dims[kj].first);
        }
      }
      captureAmulB(m, CN[ij], &K[0], Aptr.size(), &Aptr[0], &Bptr[0], basis.R[i + ibegin]);
    }
  }
  
  basis.recompressR(epi, comm);
  std::complex<double> one(1., 0.);
  for (long long i = 0; i < xlen; i++) {
    for (long long ij = C.RowIndex[i]; ij < C.RowIndex[i + 1]; ij++) {
      long long j = C.ColIndexLocal[ij];
      long long CM = C.Dims[ij].first;
      long long CN = C.Dims[ij].second;

      std::array<long long, 4> d{ 0, 0, 0, 0 };
      if (ybegin <= Ck[ij] && Ck[ij] < ybegin + nodes) {
        d = std::array<long long, 4>{ 
          basis.DimsLr[i], basis.DimsLr[j], basis.copyOffset(i), basis.copyOffset(j) };
        mulQhAQc(CM, CN, C[ij], d[0], basis.Q[i], d[1], basis.Q[j]);
      }
      else
        std::fill(C[ij], C[ij] + CM * CN, std::complex<double>(0., 0.));
      C.DimsLr[ij] = d;
    }

    if (ibegin <= i && i < ibegin + nodes)
      for (long long ij = basis.CRows[i - ibegin]; ij < basis.CRows[i - ibegin + 1]; ij++) {
        long long j = basis.CColsLocal[ij];
        long long mn = basis.DimsLr[i] * basis.DimsLr[j];
        long long cloc = C(i, basis.CCols[ij]);
        cblas_zaxpy(mn, &one, basis.C[ij], 1, C[cloc], 1);
      }
  }

  long long* DimsLrPtr = reinterpret_cast<long long*>(C.DimsLr.data());
  std::vector<long long> blocks4(xlen);
  std::transform(C.blocksOnRow.begin(), C.blocksOnRow.end(), blocks4.begin(), [](long long b) { return b * 4; });
  comm.neighbor_reduce(C.Data.data(), C.elementsOnRow.data());
  comm.neighbor_reduce(DimsLrPtr, blocks4.data());
}

std::array<std::complex<double>*, 4> inline matrixSplits(long long M, long long rankM, long long rankN, std::complex<double>* A) {
  return std::array<std::complex<double>*, 4>{ &A[rankM + M * rankN], &A[M * rankN], &A[rankM], A };
}

std::array<const std::complex<double>*, 4> inline matrixSplits(long long M, long long rankM, long long rankN, const std::complex<double>* A) {
  return std::array<const std::complex<double>*, 4>{ &A[rankM + M * rankN], &A[M * rankN], &A[rankM], A };
}

void UlvSolver::factorizeA(const ClusterBasis& basis, const CellComm& comm) {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();

  for (long long i = 0; i < nodes; i++)
    for (long long ij = A.RowIndex[i + ibegin]; ij < A.RowIndex[i + ibegin + 1]; ij++) {
      long long j = A.ColIndexLocal[ij];
      long long AM = A.Dims[ij].first;
      long long AN = A.Dims[ij].second;
      mulQhAQc(AM, AN, A[ij], AM, basis.Q[i + ibegin], AN, basis.Q[j]);
      A.DimsLr[ij] = std::array<long long, 4>{ 
        basis.DimsLr[i + ibegin], basis.DimsLr[j], basis.copyOffset(i + ibegin), basis.copyOffset(j) };
    }

  long long* DimsLrPtr = reinterpret_cast<long long*>(A.DimsLr.data());
  std::vector<long long> blocks4(A.blocksOnRow.size());
  std::transform(A.blocksOnRow.begin(), A.blocksOnRow.end(), blocks4.begin(), [](long long b) { return b * 4; });
  comm.neighbor_bcast(A.Data.data(), A.elementsOnRow.data());
  comm.neighbor_bcast(DimsLrPtr, blocks4.data());

  std::complex<double> one(1., 0.), minus_one(-1., 0.);
  for (long long i = ibegin; i < (ibegin + nodes); i++) {
    long long ii = A.DiagIndex[i];
    long long AM = A.Dims[ii].first;
    long long lenS = A.DimsLr[ii][0];
    long long lenR = AM - lenS;
    std::array<std::complex<double>*, 4> splitsD = matrixSplits(AM, lenS, lenS, A[ii]);

    if (0 < lenR) {
      LAPACKE_zgetrf(LAPACK_COL_MAJOR, lenR, lenR, splitsD[0], AM, Apiv[i].data());
      if (0 < lenS) {
        LAPACKE_zlaswp(LAPACK_COL_MAJOR, lenS, splitsD[2], AM, 1, lenR, Apiv[i].data(), 1);
        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, lenR, lenS, &one, splitsD[0], AM, splitsD[2], AM);
        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, lenS, lenR, &one, splitsD[0], AM, splitsD[1], AM);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, lenS, lenS, lenR, &minus_one, splitsD[1], AM, splitsD[2], AM, &one, splitsD[3], AM);
      }
    }
  }
}

void UlvSolver::forwardSubstitute(long long nrhs, long long lenY, std::complex<double> X[], std::complex<double> Y[], const ClusterBasis& basis, const CellComm& comm) const {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long xlen = comm.lenNeighbors();

  std::vector<long long> Xoffsets(xlen + 1);
  std::inclusive_scan(basis.Dims.begin(), basis.Dims.end(), Xoffsets.begin() + 1);
  Xoffsets[0] = 0;
  long long lenX = Xoffsets.back();
  comm.level_merge(X, lenX * nrhs);

  std::vector<long long> Yoffsets(nodes + 1);
  std::inclusive_scan(basis.DimsLr.begin() + ibegin, basis.DimsLr.begin() + (ibegin + nodes), Yoffsets.begin() + 1);
  Yoffsets[0] = 0;

  std::vector<std::complex<double>> Z(X, &X[lenX * nrhs]);
  std::complex<double> one(1., 0.), zero(0., 0.), minus_one(-1., 0.);
  for (long long i = ibegin; i < (ibegin + nodes); i++) {
    long long M = basis.Dims[i], N = basis.DimsLr[i], K = M - N;
    long long offsetIn = Xoffsets[i];
    long long offsetX = Xoffsets[i] * nrhs;
    long long offsety = Yoffsets[i - ibegin];
    const std::complex<double>* Qr = &(basis.Q[i])[M * N];
    if (0 < K)
      cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, K, nrhs, M, &one, Qr, M, &Z[offsetIn], lenX, &zero, &X[offsetX], M);
    if (0 < N)
      cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, N, nrhs, M, &one, basis.Q[i], M, &Z[offsetIn], lenX, &zero, &Y[offsety], lenY);
  }

  std::vector<long long> Xsizes(xlen);
  std::transform(basis.Dims.begin(), basis.Dims.end(), Xsizes.begin(), [=](long long d) { return d * nrhs; });
  comm.neighbor_bcast(X, Xsizes.data());

  std::vector<long long> Iiters(xlen);
  std::iota(Iiters.begin(), Iiters.begin() + ibegin, 0);
  std::iota(Iiters.begin() + ibegin, Iiters.begin() + (xlen - nodes), ibegin + nodes);
  std::iota(Iiters.begin() + (xlen - nodes), Iiters.end(), ibegin);

  for (std::vector<long long>::iterator iter = Iiters.begin(); iter != Iiters.end(); iter++) {
    long long i = *iter;
    long long Mi = basis.Dims[i], Ni = basis.DimsLr[i], Ki = Mi - Ni;
    long long offseti = Xoffsets[i] * nrhs;
    long long Ad = A.DiagIndex[i];

    if (0 < Ki) {
      std::vector<long long> ijLis(A.RowIndex[i + 1] - A.RowIndex[i]);
      std::vector<long long>::iterator LisEnd = ijLis.begin() + A.ijLower(i, &ijLis[0]);
      
      for (std::vector<long long>::iterator ij = ijLis.begin(); ij != LisEnd; ij++) {
        long long j = A.ColIndexLocal[*ij];
        long long Mj = basis.Dims[j], Nj = basis.DimsLr[j], Kj = Mj - Nj;
        long long offsetj = Xoffsets[j] * nrhs;
        std::array<const std::complex<double>*, 4> splitsij = matrixSplits(Mi, Ni, Nj, A[*ij]);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Ki, nrhs, Kj, &minus_one, splitsij[0], Mi, &X[offsetj], Mj, &one, &X[offseti], Mi);
      }

      std::array<const std::complex<double>*, 4> splitsii = matrixSplits(Mi, Ni, Ni, A[Ad]);
      LAPACKE_zlaswp(LAPACK_COL_MAJOR, nrhs, &X[offseti], Mi, 1, Ki, Apiv[i].data(), 1);
      cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, Ki, nrhs, &one, splitsii[0], Mi, &X[offseti], Mi);
    }
  }

  for (long long i = ibegin; i < (ibegin + nodes); i++) {
    long long Mi = basis.Dims[i], Ni = basis.DimsLr[i];
    long long offsety = Yoffsets[i - ibegin];

    if (0 < Ni)
      for (long long ij = A.RowIndex[i]; ij < A.RowIndex[i + 1]; ij++) {
        long long j = A.ColIndexLocal[ij];
        long long Mj = basis.Dims[j], Nj = basis.DimsLr[j], Kj = Mj - Nj;
        long long offsetj = Xoffsets[j] * nrhs;
        std::array<const std::complex<double>*, 4> splitsij = matrixSplits(Mi, Ni, Nj, A[ij]);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Ni, nrhs, Kj, &minus_one, splitsij[1], Mi, &X[offsetj], Mj, &one, &Y[offsety], lenY);
      }
  }
}

void UlvSolver::backwardSubstitute(long long nrhs, long long lenY, const std::complex<double> Y[], std::complex<double> X[], const ClusterBasis& basis, const CellComm& comm) const {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long xlen = comm.lenNeighbors();

  std::vector<long long> Xoffsets(xlen + 1);
  std::inclusive_scan(basis.Dims.begin(), basis.Dims.end(), Xoffsets.begin() + 1);
  Xoffsets[0] = 0;
  long long lenX = Xoffsets.back();

  std::vector<long long> Yoffsets(nodes + 1);
  std::inclusive_scan(basis.DimsLr.begin() + ibegin, basis.DimsLr.begin() + (ibegin + nodes), Yoffsets.begin() + 1);
  Yoffsets[0] = 0;

  std::complex<double> one(1., 0.), minus_one(-1., 0.), zero(0., 0.);
  std::vector<std::complex<double>> Z(lenX * nrhs, zero);
  for (long long i = ibegin; i < (ibegin + nodes); i++) {
    long long M = basis.Dims[i], N = basis.DimsLr[i];
    long long offsetZ = Xoffsets[i] * nrhs;
    long long offsetY = Yoffsets[i - ibegin];
    if (0 < N)
      MKL_Zomatcopy('C', 'N', N, nrhs, one, &Y[offsetY], lenY, &Z[offsetZ], M);
  }

  std::vector<long long> Zsizes(xlen);
  std::transform(basis.Dims.begin(), basis.Dims.end(), Zsizes.begin(), [=](long long d) { return d * nrhs; });
  comm.neighbor_bcast(Z.data(), Zsizes.data());
  
  for (long long i = ibegin + nodes - 1; i >= ibegin; i--) {
    long long Mi = basis.Dims[i], Ni = basis.DimsLr[i], Ki = Mi - Ni;
    long long offseti = Xoffsets[i] * nrhs;
    long long Ad = A.DiagIndex[i];

    if (0 < Ki) {
      std::vector<long long> ijLis(A.RowIndex[i + 1] - A.RowIndex[i]);
      std::vector<long long>::iterator LisEnd = ijLis.begin() + A.ijUpper(i, &ijLis[0]);

      for (std::vector<long long>::iterator ij = ijLis.begin(); ij != LisEnd; ij++) {
        long long j = A.ColIndexLocal[*ij];
        long long Mj = basis.Dims[j], Nj = basis.DimsLr[j], Kj = Mj - Nj;
        long long offsetj = Xoffsets[j] * nrhs;
        std::array<const std::complex<double>*, 4> splitsij = matrixSplits(Mi, Ni, Nj, A[*ij]);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Ki, nrhs, Kj, &minus_one, splitsij[0], Mi, &X[offsetj], Mj, &one, &X[offseti], Mi);
      }

      for (long long ij = A.RowIndex[i]; ij < A.RowIndex[i + 1]; ij++) {
        long long j = A.ColIndexLocal[ij];
        long long Mj = basis.Dims[j], Nj = basis.DimsLr[j], Kj = Mj - Nj;
        long long offsetj = Xoffsets[j] * nrhs;
        std::array<const std::complex<double>*, 4> splitsij = matrixSplits(Mi, Ni, Nj, A[ij]);
        if (0 < Nj)
          cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Ki, nrhs, Nj, &minus_one, splitsij[2], Mi, &Z[offsetj], Mj, &one, &X[offseti], Mi);
      }

      std::array<const std::complex<double>*, 4> splitsii = matrixSplits(Mi, Ni, Ni, A[Ad]);
      cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, Ki, nrhs, &one, splitsii[0], Mi, &X[offseti], Mi);
    }
  }

  std::fill(Z.begin(), Z.end(), zero);
  for (long long i = ibegin; i < ibegin + nodes; i++) {
    long long Mi = basis.Dims[i], Ni = basis.DimsLr[i], Ki = Mi - Ni;
    long long offseti = Xoffsets[i] * nrhs;
    long long offsety = Yoffsets[i - ibegin];
    const std::complex<double>* Qr = &(basis.Q[i])[Mi * Ni];
    if (0 < Ni)
      cblas_zgemm(CblasColMajor, CblasTrans, CblasConjTrans, nrhs, Mi, Ni, &one, &Y[offsety], lenY, basis.Q[i], Mi, &one, &Z[offseti], nrhs);
    if (0 < Ki)
      cblas_zgemm(CblasColMajor, CblasTrans, CblasConjTrans, nrhs, Mi, Ki, &one, &X[offseti], Mi, Qr, Mi, &one, &Z[offseti], nrhs);
  }

  for (long long i = ibegin; i < ibegin + nodes; i++) {
    long long M = basis.Dims[i];
    long long offsetZ = Xoffsets[i] * nrhs;
    long long offsetOut = Xoffsets[i];
    if (0 < M)
      MKL_Zomatcopy('C', 'T', nrhs, M, one, &Z[offsetZ], nrhs, &X[offsetOut], lenX);
  }
}

void SolveULV(long long nrhs, std::complex<double> X[], const UlvSolver matrix[], const ClusterBasis basis[], const CellComm comm[], long long levels) {
  std::vector<long long> Ysizes(levels + 1);
  std::vector<long long> Yoffset(levels + 1);
  std::vector<std::vector<std::complex<double>>> Y(levels + 1);

  for (long long l = 0; l <= levels; l++) {
    long long ibegin = comm[l].oLocal();
    Ysizes[l] = std::reduce(basis[l].Dims.begin(), basis[l].Dims.end());
    Yoffset[l] = std::reduce(basis[l].Dims.begin(), basis[l].Dims.begin() + ibegin, basis[l].childWriteOffset());
    Y[l] = std::vector<std::complex<double>>(nrhs * Ysizes[l], std::complex<double>(0., 0.));
  }

  long long ibegin = comm[levels].oLocal();
  long long nodes = comm[levels].lenLocal();
  long long offset = std::reduce(basis[levels].Dims.begin(), basis[levels].Dims.begin() + ibegin);
  long long lenX = std::reduce(basis[levels].Dims.begin() + ibegin, basis[levels].Dims.begin() + (ibegin + nodes));
  long long lenY = std::reduce(basis[levels].Dims.begin(), basis[levels].Dims.end());
  MKL_Zomatcopy('C', 'N', lenX, nrhs, std::complex<double>(1., 0.), X, lenX, &(Y[levels])[offset], lenY);

  for (long long l = levels; l > 0; l--)
    matrix[l].forwardSubstitute(nrhs, Ysizes[l - 1], Y[l].data(), &(Y[l - 1])[Yoffset[l - 1]], basis[l], comm[l]);

  matrix[0].forwardSubstitute(nrhs, Ysizes[0], Y[0].data(), Y[0].data(), basis[0], comm[0]);
  matrix[0].backwardSubstitute(nrhs, Ysizes[0], Y[0].data(), Y[0].data(), basis[0], comm[0]);

  for (long long l = 1; l <= levels; l++)
    matrix[l].backwardSubstitute(nrhs, Ysizes[l - 1], &(Y[l - 1])[Yoffset[l - 1]], Y[l].data(), basis[l], comm[l]);
  MKL_Zomatcopy('C', 'N', lenX, nrhs, std::complex<double>(1., 0.), &(Y[levels])[offset], lenY, X, lenX);
}
