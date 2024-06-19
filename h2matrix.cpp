#include <h2matrix.hpp>
#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <kernel.hpp>

#include <mkl.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/LU>
#include <algorithm>
#include <numeric>
#include <cmath>

WellSeparatedApproximation::WellSeparatedApproximation(const MatrixAccessor& eval, double epi, long long rank, long long lbegin, long long len, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& upper) :
  lbegin(lbegin), lend(lbegin + len), M(len) {
  std::vector<std::vector<double>> Fbodies(len);
  for (long long i = upper.lbegin; i < upper.lend; i++)
    for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; c++)
      if (lbegin <= c && c < lend)
        M[c - lbegin] = std::vector<double>(upper.M[i - upper.lbegin].begin(), upper.M[i - upper.lbegin].end());

  for (long long y = lbegin; y < lend; y++) {
    for (long long yx = Far.RowIndex[y]; yx < Far.RowIndex[y + 1]; yx++) {
      long long x = Far.ColIndex[yx];
      long long m = cells[y].Body[1] - cells[y].Body[0];
      long long n = cells[x].Body[1] - cells[x].Body[0];
      const double* Xbodies = &bodies[3 * cells[x].Body[0]];
      const double* Ybodies = &bodies[3 * cells[y].Body[0]];

      long long k = std::min(rank, std::min(m, n));
      std::vector<long long> ipiv(k);
      std::vector<std::complex<double>> U(n * k);
      long long iters = interpolative_decomp_aca(epi, eval, n, m, k, Xbodies, Ybodies, &ipiv[0], &U[0]);
      std::vector<double> Fbodies(3 * iters);
      for (long long i = 0; i < iters; i++)
        std::copy(&Xbodies[3 * ipiv[i]], &Xbodies[3 * (ipiv[i] + 1)], &Fbodies[3 * i]);
      M[y - lbegin].insert(M[y - lbegin].end(), Fbodies.begin(), Fbodies.end());
    }
  }
}

long long WellSeparatedApproximation::fbodies_size_at_i(long long i) const {
  return 0 <= i && i < (long long)M.size() ? M[i].size() / 3 : 0;
}

const double* WellSeparatedApproximation::fbodies_at_i(long long i) const {
  return 0 <= i && i < (long long)M.size() ? M[i].data() : nullptr;
}

long long compute_basis(const MatrixAccessor& eval, double epi, long long M, long long N, double Xbodies[], const double Fbodies[], std::complex<double> a[], std::complex<double> c[]) {
  long long K = std::min(M, N), rank = 0;
  if (0 < K) {
    Eigen::MatrixXcd RX = Eigen::MatrixXcd::Zero(K, M);

    if (K < N) {
      Eigen::MatrixXcd XF(N, M);
      gen_matrix(eval, N, M, Fbodies, Xbodies, XF.data());
      Eigen::HouseholderQR<Eigen::MatrixXcd> qr(XF);
      RX = qr.matrixQR().topRows(K).triangularView<Eigen::Upper>();
    }
    else
      gen_matrix(eval, N, M, Fbodies, Xbodies, RX.data());
    
    Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> rrqr(RX);
    rank = std::min(K, (long long)std::floor(epi));
    if (epi < 1.) {
      rrqr.setThreshold(epi);
      rank = rrqr.rank();
    }

    Eigen::Map<Eigen::MatrixXcd> A(a, M, M), C(c, M, M);
    if (0 < rank && rank < M) {
      C.topRows(rank) = rrqr.matrixR().topRows(rank);
      C.topLeftCorner(rank, rank).triangularView<Eigen::Upper>().solveInPlace(C.topRightCorner(rank, M - rank));
      C.topLeftCorner(rank, rank) = Eigen::MatrixXcd::Identity(rank, rank);

      Eigen::Map<Eigen::MatrixXd> body(Xbodies, 3, M);
      body = body * rrqr.colsPermutation();

      Eigen::HouseholderQR<Eigen::MatrixXcd> qr = (A.triangularView<Eigen::Upper>() * (rrqr.colsPermutation() * C.topRows(rank).transpose())).householderQr();
      A = qr.householderQ();
      C = Eigen::MatrixXcd::Zero(M, M);
      C.topLeftCorner(rank, rank) = qr.matrixQR().topRows(rank).triangularView<Eigen::Upper>();
    }
    else {
      A = Eigen::MatrixXcd::Identity(M, M);
      C = Eigen::MatrixXcd::Identity(M, M);
    }
  }
  return rank;
}

long long lookupIJ(const std::vector<long long>& RowIndex, const std::vector<long long>& ColIndex, long long i, long long j) {
  if (i < 0 || RowIndex.size() <= (1ull + i))
    return -1;
  long long k = std::distance(ColIndex.begin(), std::find(ColIndex.begin() + RowIndex[i], ColIndex.begin() + RowIndex[i + 1], j));
  return (k < RowIndex[i + 1]) ? k : -1;
}

H2Matrix::H2Matrix(const MatrixAccessor& eval, double epi, const Cell cells[], const CSR& Near, const CSR& Far, const double bodies[], const WellSeparatedApproximation& wsa, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm, bool use_near_bodies) {
  long long xlen = comm.lenNeighbors();
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long ybegin = comm.oGlobal();

  long long ychild = cells[ybegin].Child[0];
  long long localChildIndex = lowerComm.iLocal(ychild);

  std::vector<long long> localChildOffsets(nodes + 1);
  localChildOffsets[0] = 0;
  std::transform(&cells[ybegin], &cells[ybegin + nodes], localChildOffsets.begin() + 1, [=](const Cell& c) { return c.Child[1] - ychild; });
  Dims = std::vector<long long>(xlen, 0);
  DimsLr = std::vector<long long>(xlen, 0);
  UpperStride = std::vector<long long>(nodes, 0);

  ARows = std::vector<long long>(Near.RowIndex.begin() + ybegin, Near.RowIndex.begin() + ybegin + nodes + 1);
  ACols = std::vector<long long>(Near.ColIndex.begin() + ARows[0], Near.ColIndex.begin() + ARows[nodes]);
  long long offset = ARows[0];
  std::for_each(ARows.begin(), ARows.end(), [=](long long& i) { i = i - offset; });
  std::for_each(ACols.begin(), ACols.end(), [&](long long& col) { col = comm.iLocal(col); });
  NA = std::vector<const std::complex<double>*>(ARows[nodes], nullptr);

  CRows = std::vector<long long>(Far.RowIndex.begin() + ybegin, Far.RowIndex.begin() + ybegin + nodes + 1);
  CCols = std::vector<long long>(Far.ColIndex.begin() + CRows[0], Far.ColIndex.begin() + CRows[nodes]);
  offset = CRows[0];
  std::for_each(CRows.begin(), CRows.end(), [=](long long& i) { i = i - offset; });
  std::for_each(CCols.begin(), CCols.end(), [&](long long& col) { col = comm.iLocal(col); });
  C = std::vector<const std::complex<double>*>(CRows[nodes], nullptr);

  if (localChildOffsets.back() == 0)
    std::transform(&cells[ybegin], &cells[ybegin + nodes], &Dims[ibegin], [](const Cell& c) { return c.Body[1] - c.Body[0]; });
  else {
    std::vector<long long>::const_iterator iter = lowerA.DimsLr.begin() + localChildIndex;
    std::transform(localChildOffsets.begin(), localChildOffsets.begin() + nodes, localChildOffsets.begin() + 1, &Dims[ibegin],
      [&](long long start, long long end) { return std::reduce(iter + start, iter + end); });
  }
  comm.neighbor_bcast(Dims.data());

  std::vector<long long> Qsizes(xlen, 0);
  std::transform(Dims.begin(), Dims.end(), Qsizes.begin(), [](const long long d) { return d * d; });
  Q = MatrixDataContainer<std::complex<double>>(xlen, Qsizes.data());
  R = MatrixDataContainer<std::complex<double>>(xlen, Qsizes.data());

  std::vector<long long> Ssizes(xlen);
  std::transform(Dims.begin(), Dims.end(), Ssizes.begin(), [](const long long d) { return 3 * d; });
  S = MatrixDataContainer<double>(xlen, Ssizes.data());

  std::vector<long long> Asizes(ARows[nodes]);
  for (long long i = 0; i < nodes; i++)
    std::transform(ACols.begin() + ARows[i], ACols.begin() + ARows[i + 1], Asizes.begin() + ARows[i],
      [&](long long col) { return Dims[i + ibegin] * Dims[col]; });
  A = MatrixDataContainer<std::complex<double>>(ARows[nodes], Asizes.data());

  if (std::reduce(Dims.begin(), Dims.end())) {
    for (long long i = 0; i < nodes; i++) {
      long long M = Dims[i + ibegin];
      long long ci = i + ybegin;
      long long childi = localChildIndex + localChildOffsets[i];
      long long cend = localChildIndex + localChildOffsets[i + 1];
      Eigen::Map<Eigen::MatrixXcd> Qi(Q[i + ibegin], M, M);

      if (cend <= childi) {
        std::copy(&bodies[3 * cells[ci].Body[0]], &bodies[3 * cells[ci].Body[1]], S[i + ibegin]);
        Qi = Eigen::MatrixXcd::Identity(M, M);
      }
      for (long long j = childi; j < cend; j++) {
        long long offset = std::reduce(&lowerA.DimsLr[childi], &lowerA.DimsLr[j]);
        long long len = lowerA.DimsLr[j];
        std::copy(lowerA.S[j], lowerA.S[j] + (len * 3), &(S[i + ibegin])[offset * 3]);

        Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> Rj(lowerA.R[j], len, len, Eigen::Stride<Eigen::Dynamic, 1>(lowerA.Dims[j], 1));
        Qi.block(offset, offset, len, len) = Rj;
      }
    }
    comm.neighbor_bcast(S);
    comm.neighbor_bcast(Q);

    long long pbegin = lowerComm.oLocal();
    long long pend = pbegin + lowerComm.lenLocal();

    for (long long i = 0; i < nodes; i++) {
      long long childi = localChildIndex + localChildOffsets[i];
      long long cendi = localChildIndex + localChildOffsets[i + 1];
      long long M = Dims[i + ibegin];

      for (long long y = childi; y < cendi; y++)
        if (pbegin <= y && y < pend)
          lowerA.UpperStride[y - pbegin] = M;

      for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
        long long j = ACols[ij];
        long long N = Dims[j];

        long long j_global = Near.ColIndex[ij + Near.RowIndex[ybegin]];
        long long childj = lowerComm.iLocal(cells[j_global].Child[0]);
        long long cendj = (0 <= childj) ? (childj + cells[j_global].Child[1] - cells[j_global].Child[0]) : -1;

        if (0 < M && 0 < N) {
          Eigen::Map<Eigen::MatrixXcd> Aij(A[ij], M, N);
          Eigen::Map<const Eigen::MatrixXcd> Qi(Q[i + ibegin], M, M), Qj(Q[j], N, N);
          gen_matrix(eval, M, N, S[i + ibegin], S[j], Aij.data());
          Aij = Qi.triangularView<Eigen::Upper>() * Aij * Qj.transpose().triangularView<Eigen::Lower>();
          
          for (long long y = childi; y < cendi; y++) {
            long long offset_y = std::reduce(&lowerA.DimsLr[childi], &lowerA.DimsLr[y]);
            for (long long x = childj; x < cendj; x++) {
              long long offset_x = std::reduce(&lowerA.DimsLr[childj], &lowerA.DimsLr[x]);
              long long lowN = lookupIJ(lowerA.ARows, lowerA.ACols, y - pbegin, x);
              long long lowC = lookupIJ(lowerA.CRows, lowerA.CCols, y - pbegin, x);
              std::complex<double>* data = A[ij] + offset_y + offset_x * M;
              if (0 <= lowN)
                lowerA.NA[lowN] = data;
              if (0 <= lowC)
                lowerA.C[lowC] = data;
            }
          }
        }
      }
    }

    for (long long i = 0; i < nodes; i++) {
      long long fsize = wsa.fbodies_size_at_i(i);
      const double* fbodies = wsa.fbodies_at_i(i);
      std::vector<double> cbodies;

      if (use_near_bodies) {
        std::vector<long long>::iterator neighbors = ACols.begin() + ARows[i];
        std::vector<long long>::iterator neighbors_end = ACols.begin() + ARows[i + 1];
        long long csize = std::transform_reduce(neighbors, neighbors_end, -Dims[i + ibegin], std::plus<long long>(), [&](long long col) { return Dims[col]; });

        cbodies = std::vector<double>(3 * (fsize + csize));
        long long loc = 0;
        for (long long n = 0; n < (ARows[i + 1] - ARows[i]); n++) {
          long long col = neighbors[n];
          long long len = 3 * Dims[col];
          if (col != (i + ibegin)) {
            std::copy(S[col], S[col] + len, cbodies.begin() + loc);
            loc += len;
          }
        }
        std::copy(fbodies, &fbodies[3 * fsize], cbodies.begin() + loc);

        fsize += csize;
        fbodies = cbodies.data();
      }
      
      long long rank = compute_basis(eval, epi, Dims[i + ibegin], fsize, S[i + ibegin], fbodies, Q[i + ibegin], R[i + ibegin]);
      DimsLr[i + ibegin] = rank;
    }

    comm.neighbor_bcast(DimsLr.data());
    comm.neighbor_bcast(S);
    comm.neighbor_bcast(Q);
    comm.neighbor_bcast(R);
  }
}

/*void H2Matrix::factorize(const ColCommMPI& comm) {
  comm.level_merge(Adata.data(), Adata.size());

  long long lbegin = comm.oLocal();
  long long llen = comm.lenLocal();
  long long xlen = comm.lenNeighbors();

  for (long long i = 0; i < llen; i++) {
    long long diag = lookupIJ(ARows, ACols, i, i + lbegin);

  }
}*/

