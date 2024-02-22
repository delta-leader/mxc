
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
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();

  blocksOnRow = std::vector<long long>(xlen);
  elementsOnRow = std::vector<long long>(xlen);
  const std::pair<long long, long long>* iter = lil;
  for (long long i = 0; i < nodes; i++) {
    long long cols = std::distance(iter, std::find_if_not(iter, &lil[len], [=](std::pair<long long, long long> l) { return l.first == i; }));
    blocksOnRow[i + ibegin] = cols;
    iter = &iter[cols];
  }

  const std::vector<long long> ones(xlen, 1);
  comm.neighbor_bcast(blocksOnRow.data(), ones.data());
  comm.dup_bcast(blocksOnRow.data(), xlen);

  RowIndex = std::vector<long long>(xlen + 1);
  RowIndex[0] = 0;
  std::inclusive_scan(blocksOnRow.begin(), blocksOnRow.end(), RowIndex.begin() + 1);

  ColIndex = std::vector<long long>(RowIndex[xlen]);
  Dims = std::vector<std::pair<long long, long long>>(RowIndex[xlen]);
  DimsLr = std::vector<std::array<long long, 4>>(RowIndex[xlen]);

  std::copy(dim, &dim[len], &Dims[RowIndex[ibegin]]);
  std::transform(lil, &lil[len], &ColIndex[RowIndex[ibegin]], [](std::pair<long long, long long> l) { return l.second; });

  long long* DimsPtr = reinterpret_cast<long long*>(Dims.data());
  std::vector<long long> blocks2(xlen);
  std::transform(blocksOnRow.begin(), blocksOnRow.end(), blocks2.begin(), [](long long b) { return b + b; });
  comm.neighbor_bcast(DimsPtr, blocks2.data());
  comm.neighbor_bcast(ColIndex.data(), blocksOnRow.data());
  comm.dup_bcast(DimsPtr, RowIndex[xlen] * 2);
  comm.dup_bcast(ColIndex.data(), RowIndex[xlen]);

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
      long long k = comm.iLocal(A.ColIndex[yk]);
      for (long long kx = A.RowIndex[k]; kx < A.RowIndex[k + 1]; kx++)
        if (ycols_end == std::find(ycols, ycols_end, A.ColIndex[kx]))
          fills.emplace(std::make_pair(y, A.ColIndex[kx]), std::make_pair(yk, kx));
    }

    for (long long yk = A.RowIndex[y + ibegin]; yk < dy; yk++) {
      long long k = comm.iLocal(A.ColIndex[yk]);
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
  comm.dup_bcast(Ck.data(), C.RowIndex.back());

  Apiv = std::vector<std::vector<long long>>(xlen);
  for (long long i = 0; i < xlen; i++)
    Apiv[i] = std::vector<long long>(Dims[i], 0);
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

      for (long long cy = cybegin; cy < cyend; cy++)
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

      for (long long cy = cybegin; cy < cyend; cy++)
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
  comm.neighbor_bcast(A.Data.data(), A.elementsOnRow.data());
  comm.dup_bcast(A.Data.data(), A.DataOffsets.back());
  comm.level_merge(C.Data.data(), C.DataOffsets.back());
  comm.dup_bcast(C.Data.data(), C.DataOffsets.back());
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

void mulQhAQ(long long M, long long N, std::complex<double>* A, long long K1, const std::complex<double>* Ql, long long K2, const std::complex<double>* Qr) {
  if (0 < M && 0 < N) {
    std::vector<std::complex<double>> B(M * K2);
    std::complex<double> zero(0., 0.), one(1., 0.);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, K2, N, &one, A, M, Qr, N, &zero, &B[0], M);
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, K1, K2, M, &one, Ql, M, &B[0], M, &zero, A, K1);
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
        long long k = comm.iLocal(A.ColIndex[ik]);
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
      long long j = comm.iLocal(C.ColIndex[ij]);
      long long CM = C.Dims[ij].first;
      long long CN = C.Dims[ij].second;

      std::array<long long, 4> d{ 0, 0, 0, 0 };
      if (ybegin <= Ck[ij] && Ck[ij] < ybegin + nodes) {
        d = std::array<long long, 4>{ 
          basis.DimsLr[i], basis.DimsLr[j], basis.copyOffset(i), basis.copyOffset(j) };
        mulQhAQ(CM, CN, C[ij], d[0], basis.Q[i], d[1], basis.Q[j]);
      }
      else
        std::fill(C[ij], C[ij] + CM * CN, std::complex<double>(0., 0.));
      C.DimsLr[ij] = d;
    }

    if (ibegin <= i && i < ibegin + nodes)
      for (long long ij = basis.CRows[i - ibegin]; ij < basis.CRows[i - ibegin + 1]; ij++) {
        long long j = comm.iLocal(basis.CCols[ij]);
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
  comm.dup_bcast(C.Data.data(), C.DataOffsets.back());
  comm.dup_bcast(DimsLrPtr, C.RowIndex.back() * 4);
}

void UlvSolver::factorizeA(const ClusterBasis& basis, const CellComm& comm) {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();

  for (long long i = 0; i < nodes; i++)
    for (long long ij = A.RowIndex[i + ibegin]; ij < A.RowIndex[i + ibegin + 1]; ij++) {
      long long j = comm.iLocal(A.ColIndex[ij]);
      long long AM = A.Dims[ij].first;
      long long AN = A.Dims[ij].second;
      mulQhAQ(AM, AN, A[ij], AM, basis.Q[i + ibegin], AN, basis.Q[j]);
      A.DimsLr[ij] = std::array<long long, 4>{ 
        basis.DimsLr[i + ibegin], basis.DimsLr[j], basis.copyOffset(i + ibegin), basis.copyOffset(j) };
    }

  long long* DimsLrPtr = reinterpret_cast<long long*>(A.DimsLr.data());
  std::vector<long long> blocks4(A.blocksOnRow.size());
  std::transform(A.blocksOnRow.begin(), A.blocksOnRow.end(), blocks4.begin(), [](long long b) { return b * 4; });
  comm.neighbor_bcast(A.Data.data(), A.elementsOnRow.data());
  comm.neighbor_bcast(DimsLrPtr, blocks4.data());
  comm.dup_bcast(A.Data.data(), A.DataOffsets.back());
  comm.dup_bcast(DimsLrPtr, A.RowIndex.back() * 4);
}
