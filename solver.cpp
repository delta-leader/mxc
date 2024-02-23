
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

  RowIndex = std::vector<long long>(xlen + 1);
  RowIndex[0] = 0;
  std::inclusive_scan(blocksOnRow.begin(), blocksOnRow.end(), RowIndex.begin() + 1);

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

  Ad = std::vector<long long>(xlen);
  ALocalCol = std::vector<std::pair<long long, long long>>(xlen);
  ALocalElements = std::vector<std::vector<long long>>(xlen);
  Apiv = std::vector<std::vector<long long>>(xlen);

  for (long long i = 0; i < xlen; i++) {
    long long* begin = &A.ColIndexLocal[A.RowIndex[i]];
    long long* end = &A.ColIndexLocal[A.RowIndex[i + 1]];
    Ad[i] = std::distance(&A.ColIndexLocal[0], std::find(begin, end, i));

    ALocalCol[i].first = std::distance(&A.ColIndexLocal[0], std::find_if(begin, end, 
      [=](long long col) { return ibegin <= col && col < ibegin + nodes; }));
    ALocalCol[i].second = std::distance(&A.ColIndexLocal[0], std::find_if_not(&A.ColIndexLocal[ALocalCol[i].first], end, 
      [=](long long col) { return ibegin <= col && col < ibegin + nodes; }));

    ALocalElements[i] = std::vector<long long>(std::distance(begin, end));
    Apiv[i] = std::vector<long long>(Dims[i]);
    std::iota(ALocalElements[i].begin(), ALocalElements[i].end(), A.RowIndex[i]);
    std::remove_if(ALocalElements[i].begin(), ALocalElements[i].end(), [&](long long ij) { return A.ColIndexLocal[ij] < 0; });
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
        mulQhAQ(CM, CN, C[ij], d[0], basis.Q[i], d[1], basis.Q[j]);
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

std::array<std::complex<double>*, 4> matrixSplits(long long M, long long rankM, long long rankN, std::complex<double>* A) {
  return std::array<std::complex<double>*, 4>{ &A[rankM + M * rankN], &A[M * rankN], &A[rankM], A };
}

std::array<const std::complex<double>*, 4> matrixSplits(long long M, long long rankM, long long rankN, const std::complex<double>* A) {
  return std::array<const std::complex<double>*, 4>{ &A[rankM + M * rankN], &A[M * rankN], &A[rankM], A };
}

void UlvSolver::factorizeA(const ClusterBasis& basis, const CellComm& comm) {
  long long ibegin = comm.oLocal();
  long long ybegin = comm.oGlobal();
  long long nodes = comm.lenLocal();

  for (long long i = 0; i < nodes; i++)
    for (long long ij = A.RowIndex[i + ibegin]; ij < A.RowIndex[i + ibegin + 1]; ij++) {
      long long j = A.ColIndexLocal[ij];
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

  std::complex<double> one(1., 0.), minus_one(-1., 0.);
  for (long long i = 0; i < nodes; i++) {
    long long ii = A(i + ibegin, i + ybegin);
    long long AM = A.Dims[ii].first;
    long long lenS = A.DimsLr[ii][0];
    long long lenR = AM - lenS;
    std::array<std::complex<double>*, 4> splitsD = matrixSplits(AM, lenS, lenS, A[ii]);

    if (0 < lenR) {
      LAPACKE_zgetrf(LAPACK_COL_MAJOR, lenR, lenR, splitsD[0], AM, Apiv[i + ibegin].data());
      if (0 < lenS) {
        LAPACKE_zlaswp(LAPACK_COL_MAJOR, lenS, splitsD[2], AM, 1, lenR, Apiv[i + ibegin].data(), 1);
        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, lenR, lenS, &one, splitsD[0], AM, splitsD[2], AM);
        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, lenS, lenR, &one, splitsD[0], AM, splitsD[1], AM);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, lenS, lenS, lenR, &minus_one, splitsD[1], AM, splitsD[2], AM, &one, splitsD[3], AM);
      }
    }
  }
}

void UlvSolver::forwardSubstitute(long long nrhs, std::complex<double> X[], std::complex<double> Y[], const ClusterBasis& basis, const CellComm& comm) const {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long xlen = comm.lenNeighbors();

  std::vector<long long> Xoffsets(xlen + 1);
  std::inclusive_scan(basis.Dims.begin(), basis.Dims.end(), Xoffsets.begin() + 1);
  Xoffsets[0] = 0;
  long long lenX = Xoffsets.back();
  comm.level_merge(X, lenX * nrhs);

  std::vector<std::complex<double>> Z(lenX * nrhs);
  std::complex<double> one(1., 0.), zero(0., 0.), minus_one(-1., 0.);
  for (long long i = 0; i < nodes; i++) {
    long long M = basis.Dims[i + ibegin], N = basis.DimsLr[i + ibegin], K = M - N;
    long long offsetIn = Xoffsets[i + ibegin];
    long long offsetX = Xoffsets[i + ibegin] * nrhs;
    const std::complex<double>* Qr = &(basis.Q[i + ibegin])[M * N];

    if (0 < M) {
      MKL_Zomatcopy('C', 'N', M, nrhs, one, &X[offsetIn], lenX, &Z[offsetX], M);
      cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, K, nrhs, M, &one, Qr, M, &Z[offsetX], M, &zero, &X[offsetX], M);
    }
  }

  std::vector<long long> Xsizes(xlen);
  std::transform(basis.Dims.begin(), basis.Dims.end(), Xsizes.begin(), [=](long long d) { return d * nrhs; });
  comm.neighbor_bcast(X, Xsizes.data());

  for (long long i = 0; i < ibegin; i++) {
    long long Mi = basis.Dims[i], Ni = basis.DimsLr[i], Ki = Mi - Ni;
    long long offseti = Xoffsets[i] * nrhs;

    if (0 < Ki) {
      std::vector<long long> ijLis(A.RowIndex[i + 1] - A.RowIndex[i]);
      std::vector<long long>::iterator LisEnd = std::copy_if(ALocalElements[i].begin(), ALocalElements[i].end(), ijLis.begin(), 
        [&](long long ij) { return ij < Ad[i]; });
      
      for (std::vector<long long>::iterator ij = ijLis.begin(); ij != LisEnd; ij++) {
        long long j = A.ColIndexLocal[*ij];
        long long Mj = basis.Dims[j], Nj = basis.DimsLr[j], Kj = Mj - Nj;
        long long offsetj = Xoffsets[j] * nrhs;
        std::array<const std::complex<double>*, 4> splitsij = matrixSplits(Mi, Ni, Nj, A[*ij]);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Ki, nrhs, Kj, &minus_one, splitsij[0], Mi, &X[offsetj], Mj, &one, &X[offseti], Mi);
      }

      std::array<const std::complex<double>*, 4> splitsii = matrixSplits(Mi, Ni, Ni, A[Ad[i]]);
      LAPACKE_zlaswp(LAPACK_COL_MAJOR, nrhs, &X[offseti], Mi, 1, Ki, Apiv[i].data(), 1);
      cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, Ki, nrhs, &one, splitsii[0], Mi, &X[offseti], Mi);
    }
  }

  for (long long i = ibegin + nodes; i < xlen; i++) {
    long long Mi = basis.Dims[i], Ni = basis.DimsLr[i], Ki = Mi - Ni;
    long long offseti = Xoffsets[i] * nrhs;

    if (0 < Ki) {
      std::vector<long long> ijLis(A.RowIndex[i + 1] - A.RowIndex[i]);
      std::vector<long long>::iterator LisEnd = std::copy_if(ALocalElements[i].begin(), ALocalElements[i].end(), ijLis.begin(), 
        [&](long long ij) { return ij < Ad[i] && (ij < ALocalCol[i].first || ALocalCol[i].second <= ij); });

      for (std::vector<long long>::iterator ij = ijLis.begin(); ij != LisEnd; ij++) {
        long long j = A.ColIndexLocal[*ij];
        long long Mj = basis.Dims[j], Nj = basis.DimsLr[j], Kj = Mj - Nj;
        long long offsetj = Xoffsets[j] * nrhs;
        std::array<const std::complex<double>*, 4> splitsij = matrixSplits(Mi, Ni, Nj, A[*ij]);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Ki, nrhs, Kj, &minus_one, splitsij[0], Mi, &X[offsetj], Mj, &one, &X[offseti], Mi);
      }

      std::array<const std::complex<double>*, 4> splitsii = matrixSplits(Mi, Ni, Ni, A[Ad[i]]);
      LAPACKE_zlaswp(LAPACK_COL_MAJOR, nrhs, &X[offseti], Mi, 1, Ki, Apiv[i].data(), 1);
      cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, Ki, nrhs, &one, splitsii[0], Mi, &X[offseti], Mi);
    }
  }

  std::vector<long long> Yoffsets(xlen + 1);
  std::inclusive_scan(basis.DimsLr.begin(), basis.DimsLr.end(), Yoffsets.begin() + 1);
  Yoffsets[0] = 0;
  long long lenY = Yoffsets.back();

  for (long long i = ibegin; i < (ibegin + nodes); i++) {
    long long Mi = basis.Dims[i], Ni = basis.DimsLr[i], Ki = Mi - Ni;
    long long offseti = Xoffsets[i] * nrhs;
    long long offsety = Yoffsets[i];
    if (0 < Mi)
      cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, Ni, nrhs, Mi, &one, basis.Q[i], Mi, &Z[offseti], Mi, &zero, &Y[offsety], lenY);

    for (long long ij = A.RowIndex[i]; ij < A.RowIndex[i + 1] && 0 < Mi; ij++) {
      long long j = A.ColIndexLocal[ij];
      long long Mj = basis.Dims[j], Nj = basis.DimsLr[j], Kj = Mj - Nj;
      long long offsetj = Xoffsets[j] * nrhs;
      std::array<const std::complex<double>*, 4> splitsij = matrixSplits(Mi, Ni, Nj, A[ij]);
      if (j < Ad[i] || ALocalCol[i].second <= j)
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Ki, nrhs, Kj, &minus_one, splitsij[0], Mi, &X[offsetj], Mj, &one, &X[offseti], Mi);
      cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Ni, nrhs, Kj, &minus_one, splitsij[1], Mi, &X[offsetj], Mj, &one, &Y[offsety], lenY);
    }

    if (0 < Mi) {
      std::array<const std::complex<double>*, 4> splitsii = matrixSplits(Mi, Ni, Ni, A[Ad[i]]);
      LAPACKE_zlaswp(LAPACK_COL_MAJOR, nrhs, &X[offseti], Mi, 1, Ki, Apiv[i].data(), 1);
      cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, Ki, nrhs, &one, splitsii[0], Mi, &X[offseti], Mi);
    }
  }
}

void UlvSolver::backwardSubstitute(long long nrhs, const std::complex<double> Y[], std::complex<double> X[], const ClusterBasis& basis, const CellComm& comm) const {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long xlen = comm.lenNeighbors();

  std::vector<long long> Xoffsets(xlen + 1);
  std::inclusive_scan(basis.Dims.begin(), basis.Dims.end(), Xoffsets.begin() + 1);
  Xoffsets[0] = 0;
  long long lenX = Xoffsets.back();

  std::vector<std::complex<double>> Z(X, &X[lenX * nrhs]);
  std::vector<long long> Yoffsets(xlen + 1);
  std::inclusive_scan(basis.DimsLr.begin(), basis.DimsLr.end(), Yoffsets.begin() + 1);
  Yoffsets[0] = 0;
  long long lenY = Yoffsets.back();
  
  std::complex<double> one(1., 0.), minus_one(-1., 0.), zero(0., 0.);
  for (long long i = ibegin; i < (ibegin + nodes); i++) {
    long long Mi = basis.Dims[i], Ni = basis.DimsLr[i], Ki = Mi - Ni;
    long long offseti = Xoffsets[i] * nrhs;
    
    for (long long ij = A.RowIndex[i]; ij < A.RowIndex[i + 1] && 0 < Mi; ij++) {
      long long j = A.ColIndexLocal[ij];
      long long Mj = basis.Dims[j], Nj = basis.DimsLr[j], Kj = Mj - Nj;
      long long offsetj = Xoffsets[j] * nrhs;
      long long offsety = Yoffsets[j];
      std::array<const std::complex<double>*, 4> splitsij = matrixSplits(Mi, Ni, Nj, A[ij]);
      if (Ad[i] < j && j < ALocalCol[i].second)
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Ki, nrhs, Kj, &minus_one, splitsij[0], Mi, &X[offsetj], Mj, &one, &X[offseti], Mi);
      cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Ki, nrhs, Nj, &minus_one, splitsij[2], Mi, &Y[offsety], lenY, &one, &X[offseti], Mi);
    }

    if (0 < Mi) {
      std::array<const std::complex<double>*, 4> splitsii = matrixSplits(Mi, Ni, Ni, A[Ad[i]]);
      cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, Ki, nrhs, &one, splitsii[0], Mi, &X[offseti], Mi);

      long long offsety = Yoffsets[i];
      const std::complex<double>* Qr = &(basis.Q[i])[Mi * Ni];
      cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Mi, nrhs, Ni, &one, basis.Q[i], Mi, &Y[offsety], lenY, &zero, &Z[offseti], Mi);
      cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Mi, nrhs, Ki, &one, Qr, Mi, &X[offseti], Mi, &one, &Z[offseti], Mi);
    }
  }

  std::vector<long long> Zsizes(xlen);
  std::transform(basis.Dims.begin(), basis.Dims.end(), Zsizes.begin(), [=](long long d) { return d * nrhs; });
  comm.neighbor_bcast(Z.data(), Zsizes.data());

  for (long long i = 0; i < nodes; i++) {
    long long M = basis.Dims[i + ibegin], N = basis.DimsLr[i + ibegin], K = M - N;
    long long offsetX = Xoffsets[i + ibegin] * nrhs;
    long long offsetOut = Xoffsets[i + ibegin];
    if (0 < M)
      MKL_Zomatcopy('C', 'N', K, nrhs, one, &Z[offsetX], M, &X[offsetOut], lenX);
  }
}
