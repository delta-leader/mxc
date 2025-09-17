#include <h2matrix.hpp>
#include <h-matrix.hpp>
#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <kernel.hpp>

#include <numeric>
#include <algorithm>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include<iostream>

long long compute_basis(const MatrixAccessor& eval, double epi, long long M, long long N, double Xbodies[], const double Fbodies[], std::complex<double> a[], std::complex<double> c[], bool orth) {
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

    Eigen::ColPivHouseholderQR<Eigen::Ref<Eigen::MatrixXcd>> rrqr(RX);
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

      if (orth) {
        RX = A.triangularView<Eigen::Upper>() * (rrqr.colsPermutation() * C.topRows(rank).transpose());
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXcd>> qr(RX);
        A = qr.householderQ();
        C.setZero();
        C.topLeftCorner(rank, rank) = qr.matrixQR().topRows(rank).triangularView<Eigen::Upper>();
      }
      else {
        A.setZero();
        A.leftCols(rank) = rrqr.colsPermutation() * C.topRows(rank).transpose();
        C.setZero();
        C.topLeftCorner(rank, rank) = Eigen::MatrixXcd::Identity(rank, rank);
      }
    }
    else {
      C = A.triangularView<Eigen::Upper>();
      A = Eigen::MatrixXcd::Identity(M, M);
    }
  }
  return rank;
}

long long compute_basis(const Eigen::MatrixXcd& mat, double epi, long long s[], std::complex<double> q[], std::complex<double> r[], bool orth) {
  long long M = mat.rows();
  long long N = mat.cols();
  long long K = std::min(M, N);
  long long rank = 0;

  if (0 < K) {
    Eigen::MatrixXcd RX = Eigen::MatrixXcd::Zero(K, N);

    if (K < M) {
      Eigen::HouseholderQR<Eigen::MatrixXcd> qr(mat);
      RX = qr.matrixQR().topRows(K).triangularView<Eigen::Upper>();
    } else {
      RX = mat;
    }
    
    // fixed a bug where we directly used mat here
    Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> rrqr(RX);
    rank = std::min(K, (long long)std::floor(epi));
    if (epi < 1.) {
      rrqr.setThreshold(epi);
      rank = rrqr.rank();
    }
    
    Eigen::Map<Eigen::MatrixXcd> Q(q, N, N), R(r, N, N);
    if (0 < rank && rank < N) {
      R.topRows(rank) = rrqr.matrixR().topRows(rank);
      R.topLeftCorner(rank, rank).triangularView<Eigen::Upper>().solveInPlace(R.topRightCorner(rank, N - rank));
      R.topLeftCorner(rank, rank) = Eigen::MatrixXcd::Identity(rank, rank);
      
      Eigen::Map<Eigen::RowVector<long long, Eigen::Dynamic>> indices(s, N);
      indices = indices * rrqr.colsPermutation();

      if (orth) {
        Eigen::MatrixXcd  RX = Q.triangularView<Eigen::Upper>() * (rrqr.colsPermutation() * R.topRows(rank).transpose());
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXcd>> qr(RX);
        Q = qr.householderQ();
        R.setZero();
        R.topLeftCorner(rank, rank) = qr.matrixQR().topRows(rank).triangularView<Eigen::Upper>();
      }
      else {
        Q.setZero();
        Q.leftCols(rank) = rrqr.colsPermutation() * R.topRows(rank).transpose();
        R.setZero();
        R.topLeftCorner(rank, rank) = Eigen::MatrixXcd::Identity(rank, rank);
      }
    }
    else {
      R = Q.triangularView<Eigen::Upper>();
      Q = Eigen::MatrixXcd::Identity(N, N);
    }
  }
  return rank;
}

long long compute_basis(const Eigen::MatrixXcd& mat, double epi, long long s[], long long s_local[], std::complex<double> q[], std::complex<double> r[], bool orth) {
  long long M = mat.rows();
  long long N = mat.cols();
  long long K = std::min(M, N);
  long long rank = 0;

  if (0 < K) {
    Eigen::MatrixXcd RX = Eigen::MatrixXcd::Zero(K, N);

    if (K < M) {
      Eigen::HouseholderQR<Eigen::MatrixXcd> qr(mat);
      RX = qr.matrixQR().topRows(K).triangularView<Eigen::Upper>();
    } else {
      RX = mat;
    }
    
    // fixed a bug where we directly used mat here
    Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> rrqr(RX);
    rank = std::min(K, (long long)std::floor(epi));
    if (epi < 1.) {
      rrqr.setThreshold(epi);
      rank = rrqr.rank();
    }
    
    Eigen::Map<Eigen::MatrixXcd> Q(q, N, N), R(r, N, N);
    if (0 < rank && rank < N) {
      R.topRows(rank) = rrqr.matrixR().topRows(rank);
      R.topLeftCorner(rank, rank).triangularView<Eigen::Upper>().solveInPlace(R.topRightCorner(rank, N - rank));
      R.topLeftCorner(rank, rank) = Eigen::MatrixXcd::Identity(rank, rank);
      
      Eigen::Map<Eigen::RowVector<long long, Eigen::Dynamic>> indices(s, N);
      indices = indices * rrqr.colsPermutation();
      Eigen::Map<Eigen::RowVector<long long, Eigen::Dynamic>> indices_local(s_local, N);
      indices_local = indices_local * rrqr.colsPermutation();

      if (orth) {
        Eigen::MatrixXcd  RX = Q.triangularView<Eigen::Upper>() * (rrqr.colsPermutation() * R.topRows(rank).transpose());
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXcd>> qr(RX);
        Q = qr.householderQ();
        R.setZero();
        R.topLeftCorner(rank, rank) = qr.matrixQR().topRows(rank).triangularView<Eigen::Upper>();
      }
      else {
        Q.setZero();
        Q.leftCols(rank) = rrqr.colsPermutation() * R.topRows(rank).transpose();
        R.setZero();
        R.topLeftCorner(rank, rank) = Eigen::MatrixXcd::Identity(rank, rank);
      }
    }
    else {
      R = Q.triangularView<Eigen::Upper>();
      Q = Eigen::MatrixXcd::Identity(N, N);
    }
  }
  return rank;
}

long long compute_basis_rid(const Eigen::MatrixXcd& mat, double epi, long long s[], std::complex<double> q[], std::complex<double> r[], bool orth, long long oversampling=5) {
  long long M = mat.rows();
  long long N = mat.cols();
  long long K = std::min(M, N);
  long long rank = 0;
  // we always compress the rows (lower triangular part)
  if (0 < K) {
    Eigen::MatrixXcd RX = Eigen::MatrixXcd::Zero(K, N);

    rank = std::min(K, (long long)std::floor(epi));
    Eigen::MatrixXcd RN = Eigen::MatrixXcd::Random(mat.rows(), rank + oversampling);
    Eigen::MatrixXcd Y = (mat.transpose() * RN).transpose();

    //if (K < M) {
    //  Eigen::HouseholderQR<Eigen::MatrixXcd> qr(Y);
    //  RX = qr.matrixQR().topRows(K).triangularView<Eigen::Upper>();
    //} else {
    //  RX = Y;
    //}


    Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> rrqr(Y);
    //rank = std::min(K, (long long)std::floor(epi));
    //rank = std::min(K, (long long)std::floor(epi));
    //std::cout<<"Used rank: "<<rank<<std::endl;
    //rrqr.setThreshold(1e-2);
    //long long drank = rrqr.rank();
    //std::cout<<"Determined rank: "<<drank<<std::endl;
    if (epi < 1.) {
      rrqr.setThreshold(epi);
      rank = rrqr.rank();
    }

    Eigen::Map<Eigen::MatrixXcd> Q(q, N, N), R(r, N, N);
    if (0 < rank && rank < N) {
      R.topRows(rank) = rrqr.matrixR().topRows(rank);
      R.topLeftCorner(rank, rank).triangularView<Eigen::Upper>().solveInPlace(R.topRightCorner(rank, N - rank));
      R.topLeftCorner(rank, rank) = Eigen::MatrixXcd::Identity(rank, rank);
      
      Eigen::Map<Eigen::RowVector<long long, Eigen::Dynamic>> indices(s, N);
      indices = indices * rrqr.colsPermutation();
      if (orth) {
        Eigen::MatrixXcd  RX = Q.triangularView<Eigen::Upper>() * (rrqr.colsPermutation() * R.topRows(rank).transpose());
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXcd>> qr(RX);
        Q = qr.householderQ();
        R.setZero();
        R.topLeftCorner(rank, rank) = qr.matrixQR().topRows(rank).triangularView<Eigen::Upper>();
      }
      else {
        Q.setZero();
        Q.leftCols(rank) = rrqr.colsPermutation() * R.topRows(rank).transpose();
        R.setZero();
        R.topLeftCorner(rank, rank) = Eigen::MatrixXcd::Identity(rank, rank);
      }
    }
    else {
      R = Q.triangularView<Eigen::Upper>();
      Q = Eigen::MatrixXcd::Identity(N, N);
    }
  }
  return rank;
}

inline long long lookupIJ(const std::vector<long long>& RowIndex, const std::vector<long long>& ColIndex, long long i, long long j) {
  if (i < 0 || RowIndex.size() <= (1ull + i))
    return -1;
  long long k = std::distance(ColIndex.begin(), std::find(ColIndex.begin() + RowIndex[i], ColIndex.begin() + RowIndex[i + 1], j));
  return (k < RowIndex[i + 1]) ? k : -1;
}

void H2Matrix::construct(const MatrixAccessor& eval, double epi, const Cell cells[], const CSR& Near, const double bodies[], const Hmatrix& wsa, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm) {
  long long xlen = comm.lenNeighbors();
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long ybegin = comm.oGlobal();

  Dims.resize(xlen, 0);
  DimsLr.resize(xlen, 0);
  UpperStride.resize(nodes, 0);

  ARows.insert(ARows.begin(), comm.ARowOffsets.begin(), comm.ARowOffsets.end());
  ACols.insert(ACols.begin(), comm.AColumns.begin(), comm.AColumns.end());
  CRows.insert(CRows.begin(), comm.CRowOffsets.begin(), comm.CRowOffsets.end());
  CCols.insert(CCols.begin(), comm.CColumns.begin(), comm.CColumns.end());
  NA.resize(ARows[nodes], -1);

  long long localChildLen = cells[ybegin + nodes - 1].Child[1] - cells[ybegin].Child[0];
  std::vector<long long> localChildOffsets(nodes + 1, -1);
  
  if (0 < localChildLen) {
    long long lowerBegin = lowerComm.oLocal() + comm.LowerX;
    long long localChildIndex = lowerBegin - cells[ybegin].Child[0];
    std::transform(&cells[ybegin], &cells[ybegin + nodes], localChildOffsets.begin() + 1, [=](const Cell& c) { return localChildIndex + c.Child[1]; });
    localChildOffsets[0] = lowerBegin;

    std::vector<long long> ranks_offsets(localChildLen + 1);
    std::inclusive_scan(lowerA.DimsLr.begin() + localChildOffsets[0], lowerA.DimsLr.begin() + localChildOffsets[nodes], ranks_offsets.begin() + 1);
    ranks_offsets[0] = 0;

    std::transform(localChildOffsets.begin(), localChildOffsets.begin() + nodes, localChildOffsets.begin() + 1, &Dims[ibegin],
      [&](long long start, long long end) { return ranks_offsets[end - lowerBegin] - ranks_offsets[start - lowerBegin]; });

    lenX = ranks_offsets.back();
    LowerZ = std::reduce(lowerA.DimsLr.begin(), lowerA.DimsLr.begin() + lowerBegin, 0ll);
  }
  else {
    std::transform(&cells[ybegin], &cells[ybegin + nodes], &Dims[ibegin], [](const Cell& c) { return c.Body[1] - c.Body[0]; });
    lenX = std::reduce(&Dims[ibegin], &Dims[ibegin + nodes]);
    LowerZ = 0;
  }

  std::vector<long long> neighbor_ones(xlen, 1ll);
  comm.dataSizesToNeighborOffsets(neighbor_ones.data());
  comm.neighbor_bcast(Dims.data(), neighbor_ones.data());
  X.alloc(xlen, Dims.data());
  Y.alloc(xlen, Dims.data());

  std::vector<long long> Qsizes(xlen, 0);
  std::transform(Dims.begin(), Dims.end(), Qsizes.begin(), [](const long long d) { return d * d; });
  Q.alloc(xlen, Qsizes.data());
  R.alloc(xlen, Qsizes.data());

  std::vector<long long> Ssizes(xlen);
  std::transform(Dims.begin(), Dims.end(), Ssizes.begin(), [](const long long d) { return 3 * d; });
  S.alloc(xlen, Ssizes.data());

  std::vector<long long> Asizes(ARows[nodes]);
  for (long long i = 0; i < nodes; i++)
    std::transform(ACols.begin() + ARows[i], ACols.begin() + ARows[i + 1], Asizes.begin() + ARows[i],
      [&](long long col) { return Dims[i + ibegin] * Dims[col]; });
  A.alloc(ARows[nodes], Asizes.data());

  typedef Eigen::Stride<Eigen::Dynamic, 1> Stride_t;
  typedef Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Stride_t> Matrix_t; 

  if (std::reduce(Dims.begin(), Dims.end())) {
    long long pbegin = lowerComm.oLocal();
    long long pend = pbegin + lowerComm.lenLocal();

    for (long long i = 0; i < nodes; i++) {
      long long M = Dims[i + ibegin];
      long long childi = localChildOffsets[i];
      long long cendi = localChildOffsets[i + 1];
      Eigen::Map<Eigen::MatrixXcd> Qi(Q[i + ibegin], M, M);

      for (long long y = childi; y < cendi; y++) { // Intermediate levels
        long long offset_y = std::reduce(&lowerA.DimsLr[childi], &lowerA.DimsLr[y]);
        long long ny = lowerA.DimsLr[y];
        std::copy(lowerA.S[y], lowerA.S[y] + (ny * 3), &(S[i + ibegin])[offset_y * 3]);

        Matrix_t Ry(lowerA.R[y], ny, ny, Stride_t(lowerA.Dims[y], 1));
        Qi.block(offset_y, offset_y, ny, ny) = Ry;

        if (pbegin <= y && y < pend && 0 < M) {
          long long py = y - pbegin;
          lowerA.UpperStride[py] = M;

          for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
            long long j_global = Near.ColIndex[ij + Near.RowIndex[ybegin]];
            long long childj = lowerComm.iLocal(cells[j_global].Child[0]);
            long long cendj = (0 <= childj) ? (childj + cells[j_global].Child[1] - cells[j_global].Child[0]) : -1;

            for (long long x = childj; x < cendj; x++) {
              long long offset_x = std::reduce(&lowerA.DimsLr[childj], &lowerA.DimsLr[x]);
              long long nx = lowerA.DimsLr[x];
              long long lowN = lookupIJ(lowerA.ARows, lowerA.ACols, py, x);
              long long lowC = lookupIJ(lowerA.CRows, lowerA.CCols, py, x);
              std::complex<double>* dp = A[ij] + offset_y + offset_x * M;
              if (0 <= lowN)
                lowerA.NA[lowN] = std::distance(A[0], dp);
              else if (0 <= lowC)
                Matrix_t(dp, ny, nx, Stride_t(M, 1)) = Eigen::Map<Eigen::MatrixXcd>(lowerA.C[lowC], ny, nx);
            }
          }
        }
      }

      if (cendi <= childi) { // Leaf level
        long long ci = i + ybegin;
        std::copy(&bodies[3 * cells[ci].Body[0]], &bodies[3 * cells[ci].Body[1]], S[i + ibegin]);
        Qi = Eigen::MatrixXcd::Identity(M, M);

        for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
          long long N = Dims[ACols[ij]];
          long long cj = Near.ColIndex[ij + Near.RowIndex[ybegin]];
          gen_matrix(eval, M, N, &bodies[3 * cells[ci].Body[0]], &bodies[3 * cells[cj].Body[0]], A[ij]);
        }
      }
    }

    comm.dataSizesToNeighborOffsets(Ssizes.data());
    comm.neighbor_bcast(S[0], Ssizes.data());

    for (long long i = 0; i < nodes; i++) {
      long long fsize = wsa.fbodies_size_at_i(i);
      const double* fbodies = wsa.fbodies_at_i(i);
      long long rank = compute_basis(eval, epi, Dims[i + ibegin], fsize, S[i + ibegin], fbodies, Q[i + ibegin], R[i + ibegin], 1. <= epi);
      DimsLr[i + ibegin] = rank;
    }

    comm.dataSizesToNeighborOffsets(Qsizes.data());
    comm.neighbor_bcast(DimsLr.data(), neighbor_ones.data());
    comm.neighbor_bcast(S[0], Ssizes.data());
    comm.neighbor_bcast(Q[0], Qsizes.data());
    comm.neighbor_bcast(R[0], Qsizes.data());
  }

  if (std::reduce(DimsLr.begin(), DimsLr.end())) {
    std::vector<long long> Csizes(CRows[nodes]);
    for (long long i = 0; i < nodes; i++)
      std::transform(CCols.begin() + CRows[i], CCols.begin() + CRows[i + 1], Csizes.begin() + CRows[i],
        [&](long long col) { return DimsLr[i + ibegin] * DimsLr[col]; });
    C.alloc(CRows[nodes], Csizes.data());

    std::vector<long long> Usizes(nodes);
    std::transform(&Dims[ibegin], &Dims[ibegin + nodes], &DimsLr[ibegin], Usizes.begin(), std::multiplies<long long>());
    U.alloc(nodes, Usizes.data());
    Z.alloc(xlen, DimsLr.data());
    W.alloc(xlen, DimsLr.data());

    for (long long i = 0; i < nodes; i++) {
      long long y = i + ibegin;
      long long M = DimsLr[y];
      Matrix_t Ry(R[y], M, M, Stride_t(Dims[y], 1));
      Eigen::Map<Eigen::MatrixXcd>(U[i], Dims[y], M) = Eigen::Map<Eigen::MatrixXcd>(Q[y], Dims[y], M);

      for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
        long long x = CCols[ij];
        long long N = DimsLr[CCols[ij]];
        Matrix_t Rx(R[x], N, N, Stride_t(Dims[x], 1));

        Eigen::Map<Eigen::MatrixXcd> Cyx(C[ij], M, N);
        if (1. <= epi) {
          Eigen::MatrixXcd Ayx(M, N);
          gen_matrix(eval, M, N, S[y], S[x], Ayx.data());
          Cyx.noalias() = Ry.triangularView<Eigen::Upper>() * Ayx * Rx.transpose().triangularView<Eigen::Lower>();
        }
        else
          gen_matrix(eval, M, N, S[y], S[x], Cyx.data());
      }
    }
  }

  NbXoffsets.insert(NbXoffsets.begin(), Dims.begin(), Dims.end());
  NbXoffsets.erase(NbXoffsets.begin() + comm.dataSizesToNeighborOffsets(NbXoffsets.data()), NbXoffsets.end());
  NbZoffsets.insert(NbZoffsets.begin(), DimsLr.begin(), DimsLr.end());
  NbZoffsets.erase(NbZoffsets.begin() + comm.dataSizesToNeighborOffsets(NbZoffsets.data()), NbZoffsets.end());
}


void H2Matrix::construct(const MatrixGenerator& matgen, double epi, const Cell cells[], const CSR& Near, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm, const double omega, const double scale) {
  // number of cells on this level (this process and neighbors)
  long long xlen = comm.lenNeighbors();
  // index of the first cell for this process on this level
  long long ibegin = comm.oLocal();
  // number of cells for this process on this level
  long long nodes = comm.lenLocal();
  // index of the first cell for this process in the global cell array
  long long ybegin = comm.oGlobal();
  std::cout<<"START"<<std::endl;

  // dimensions for each cell on this level
  Dims.resize(xlen, 0);
  // LR dimensions for each cell on this level
  DimsLr.resize(xlen, 0);
  // stide for what?
  UpperStride.resize(nodes, 0);

  // get the Nearfield indices CSR
  ARows.insert(ARows.begin(), comm.ARowOffsets.begin(), comm.ARowOffsets.end());
  ACols.insert(ACols.begin(), comm.AColumns.begin(), comm.AColumns.end());
  // get the far-field indices CSR
  CRows.insert(CRows.begin(), comm.CRowOffsets.begin(), comm.CRowOffsets.end());
  CCols.insert(CCols.begin(), comm.CColumns.begin(), comm.CColumns.end());
  // ?
  NA.resize(ARows[nodes], -1);

  // get the number of local children
  long long localChildLen = cells[ybegin + nodes - 1].Child[1] - cells[ybegin].Child[0];
  std::vector<long long> localChildOffsets(nodes + 1, -1);
  
  if (0 < localChildLen) {
    long long lowerBegin = lowerComm.oLocal() + comm.LowerX;
    long long localChildIndex = lowerBegin - cells[ybegin].Child[0];
    std::transform(&cells[ybegin], &cells[ybegin + nodes], localChildOffsets.begin() + 1, [=](const Cell& c) { return localChildIndex + c.Child[1]; });
    localChildOffsets[0] = lowerBegin;

    std::vector<long long> ranks_offsets(localChildLen + 1);
    std::inclusive_scan(lowerA.DimsLr.begin() + localChildOffsets[0], lowerA.DimsLr.begin() + localChildOffsets[nodes], ranks_offsets.begin() + 1);
    ranks_offsets[0] = 0;

    std::transform(localChildOffsets.begin(), localChildOffsets.begin() + nodes, localChildOffsets.begin() + 1, &Dims[ibegin],
      [&](long long start, long long end) { return ranks_offsets[end - lowerBegin] - ranks_offsets[start - lowerBegin]; });

    lenX = ranks_offsets.back();
    LowerZ = std::reduce(lowerA.DimsLr.begin(), lowerA.DimsLr.begin() + lowerBegin, 0ll);
    n_mat = std::reduce(&Dims[ibegin], &Dims[ibegin + nodes], 0ll);
    // Mat on the intermediate levels stores only the far field matrices
    std::vector<long long> row_offsets(&Dims[ibegin], &Dims[ibegin + nodes]);
    Cols.resize(nodes, 0);
    std::cout<<"Mat sizes: ";
    for (size_t i = 0; i < nodes; ++i) {
      Cols[i] = n_mat - row_offsets[i];
      row_offsets[i] *= Cols[i];
      std::cout<<row_offsets[i]<<", ";
    }
    std::cout<<std::endl;
    // todo the above is only correct for the next level after the leaf level?
    Mat.alloc(nodes, row_offsets.data());
  }
  else {
    // only for leaf level
    // Dims stores the number of particels for each cell (multiplied by 3)
    std::transform(&cells[ybegin], &cells[ybegin + nodes], &Dims[ibegin], [](const Cell& c) { return (c.Body[1] - c.Body[0]) * 3; });
    // total number of particles on this process
    lenX = std::reduce(&Dims[ibegin], &Dims[ibegin + nodes]);
    LowerZ = 0;
    n_mat = matgen.get_num_total() * 3;
    std::vector<long long> row_offsets(&Dims[ibegin], &Dims[ibegin + nodes]);
    for (auto& offset : row_offsets)
      offset *= n_mat;
    Mat.alloc(nodes, row_offsets.data());
    // I do this in a loop now, but I could also just create one big matrix
    // in one go if I reduce the dimensions first
    // this would however, change the data layout and I would need to account for that
    for (long long i = 0; i < nodes; ++i) {
      matgen.gen_matrix_sorted(Mat[i], cells[ybegin + i].Body[0], Dims[ibegin + i] / 3, omega, scale);
    }
    Cols.resize(nodes, n_mat);
     // we should also calculate the scale distributed, but lets keep that for later
  }

  std::vector<long long> neighbor_ones(xlen, 1ll);
  comm.dataSizesToNeighborOffsets(neighbor_ones.data());
  comm.neighbor_bcast(Dims.data(), neighbor_ones.data());
  X.alloc(xlen, Dims.data());
  Y.alloc(xlen, Dims.data());
  // S stores the indices
  std::vector<long long> Ssizes(Dims);
  S_ind.alloc(xlen, Ssizes.data());
  S_ind_orig.alloc(xlen, Ssizes.data());
  dim_offsets.resize(Dims.size());
  std::exclusive_scan(Dims.begin(), Dims.end(), dim_offsets.begin(), 0);
  std::cout<<"Dim offsets ";
  for (size_t i = 0; i < dim_offsets.size(); ++i)
    std::cout<<dim_offsets[i]<<", ";
  std::cout<<std::endl;
  std::cout<<"Lowest "<<lowest<<" "<<lowerA.lowest<<std::endl;

  std::vector<long long> Qsizes(xlen, 0);
  // Qs and Rs are square matrices
  std::transform(Dims.begin(), Dims.end(), Qsizes.begin(), [](const long long d) { return d * d; });
  Q.alloc(xlen, Qsizes.data());
  R.alloc(xlen, Qsizes.data());

  // near field blocks (i.e. dense matrices on the leaf level), not necessarily square
  std::vector<long long> Asizes(ARows[nodes]);
  for (long long i = 0; i < nodes; i++)
    std::transform(ACols.begin() + ARows[i], ACols.begin() + ARows[i + 1], Asizes.begin() + ARows[i],
      [&](long long col) { return Dims[i + ibegin] * Dims[col]; });
  A.alloc(ARows[nodes], Asizes.data());

  typedef Eigen::Stride<Eigen::Dynamic, 1> Stride_t;
  typedef Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Stride_t> Matrix_t; 

  // skip blocks that contain no points (because they are split further)
  if (std::reduce(Dims.begin(), Dims.end())) {
    // index of the first cell for this process on this level
    long long pbegin = lowerComm.oLocal();
    // number of cells for this process/level
    long long pend = pbegin + lowerComm.lenLocal();

    // loop over all nodes
    for (long long i = 0; i < nodes; i++) {
      //std::cout<<"Node "<<i<<std::endl;
      // number of rows in that cell
      long long M = Dims[i + ibegin];
      //std::cout<<"Rows "<<M<<std::endl;
      long long childi = localChildOffsets[i];
      long long cendi = localChildOffsets[i + 1];
      // get the corresponding Q matrix (as a reference)
      Eigen::Map<Eigen::MatrixXcd> Qi(Q[i + ibegin], M, M);
      // initialize the local indices
      //long long offset_ind = std::reduce(&Dims[ibegin], &Dims[ibegin+i]);
      //std::iota(S_ind_local[i + ibegin], S_ind_local[i + ibegin + 1], offset_ind);

      // for all children (i.e. only on the intermediate levels)
      for (long long y = childi; y < cendi; y++) {
        long long offset_y = std::reduce(&lowerA.DimsLr[childi], &lowerA.DimsLr[y]);
        long long ny = lowerA.DimsLr[y];
        // S_ind already has been broadcast on the lower level, so this is fine
        std::copy(lowerA.S_ind[y], lowerA.S_ind[y] + ny, &(S_ind[i + ibegin])[offset_y]);
        std::copy(lowerA.S_ind_orig[y], lowerA.S_ind_orig[y] + ny, &(S_ind_orig[i + ibegin])[offset_y]);

        Matrix_t Ry(lowerA.R[y], ny, ny, Stride_t(lowerA.Dims[y], 1));
        Qi.block(offset_y, offset_y, ny, ny) = Ry;

        if (pbegin <= y && y < pend && 0 < M) {
          long long py = y - pbegin;
          lowerA.UpperStride[py] = M;

          for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
            long long j_global = Near.ColIndex[ij + Near.RowIndex[ybegin]];
            long long childj = lowerComm.iLocal(cells[j_global].Child[0]);
            long long cendj = (0 <= childj) ? (childj + cells[j_global].Child[1] - cells[j_global].Child[0]) : -1;

            for (long long x = childj; x < cendj; x++) {
              long long offset_x = std::reduce(&lowerA.DimsLr[childj], &lowerA.DimsLr[x]);
              long long nx = lowerA.DimsLr[x];
              long long lowN = lookupIJ(lowerA.ARows, lowerA.ACols, py, x);
              long long lowC = lookupIJ(lowerA.CRows, lowerA.CCols, py, x);
              std::complex<double>* dp = A[ij] + offset_y + offset_x * M;
              if (0 <= lowN)
                lowerA.NA[lowN] = std::distance(A[0], dp);
              else if (0 <= lowC)
                Matrix_t(dp, ny, nx, Stride_t(M, 1)) = Eigen::Map<Eigen::MatrixXcd>(lowerA.C[lowC], ny, nx);
            }
          }
        }
      }
       
      // leaf level (i.e. no children)
      if (cendi <= childi) {
        long long ci = i + ybegin;
        // numbering the indices locally will not work for the far field
        std::iota(S_ind[i + ibegin], S_ind[i + ibegin + 1], cells[ci].Body[0] * 3);
        std::iota(S_ind_orig[i + ibegin], S_ind_orig[i + ibegin + 1], cells[ci].Body[0] * 3);
        Qi = Eigen::MatrixXcd::Identity(M, M);

        long long far_cols = n_mat;
        Eigen::Map<Eigen::MatrixXcd> Mat_i(Mat[i], M, n_mat);
        //std::cout<<Mat_i.rows()<<" "<<Mat_i.cols()<<std::endl;
        // generate the near field aka dense matrices in A
        for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
          //std::cout<<"J "<<ij<<std::endl;
          long long N = Dims[ACols[ij]];
          far_cols -= N;
          long long cj = Near.ColIndex[ij + Near.RowIndex[ybegin]];
          //memcpy(A[ij], M[i] + cells[cj].Body[0] * 3, sizeof(std::complex<double>) * M *N);
          Eigen::Map<Eigen::MatrixXcd> A_ij(A[ij], M, N);
          // we could optimize this, as we don't necessarily need to make a copy here
          A_ij = Mat_i.block(0, cells[cj].Body[0] * 3, M, N);
        }
        if (1. <= epi) {
          // build an HSS basis
          far_cols = n_mat - M;
          Eigen::MatrixXcd far(M, far_cols);
          long long diag = Near.ColIndex[ARows[i] + Near.RowIndex[ybegin]];
          long long left = cells[ci].Body[0] * 3;
          long long right = cells[ci].Body[1] * 3;
          //std::cout<<"Left Cols "<<0<<" "<<cells[ci].Body[0] * 3<<" | "<<M<<std::endl;
          far.leftCols(left) = Mat_i.leftCols(left);
          far.rightCols(n_mat - right) = Mat_i.rightCols(n_mat - right);
          long long rank = compute_basis(far.transpose(), epi, S_ind[i + ibegin], S_ind_orig[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
          //std::cout<<"Rank "<<rank<<std::endl;
          /*for (int c = 0; c < far.cols(); ++c) {
            double col_norm = far.col(c).norm();
            long long count = 0;
            for (int r = 0; r < far.rows(); ++r)
              if (std::abs(far(r,c)) >= threshold * col_norm)
                count++;
            std::cout<<"Col "<< c <<": " << count<<", Density: "<< ((double)count)/(far.rows())<<std::endl;
          }*/
          DimsLr[i + ibegin] = rank;
        } else {
          // build an H2 basis
          // generate the far field only if it exists
          if (far_cols > 0) {
            // not tested after transpose
            Eigen::MatrixXcd far(M, far_cols);
            long long current_near = Near.ColIndex[ARows[i] + Near.RowIndex[ybegin]];
            //std::cout<<"Current Near "<<current_near<<std::endl;
            //std::cout<<ARows[i]<<" "<<ARows[i+1]<<std::endl;
            long long current_cols = cells[current_near].Body[0] * 3;
            //std::cout<<"Top Rows "<<0<<" "<<cells[ci].Body[0] * 3<<" | "<<current_rows<<" "<<M<<std::endl;
            far.leftCols(current_cols) = Mat_i.block(cells[ci].Body[0] * 3, 0, M, current_cols);
            //far.topRows(current_rows) = mat.block(0, cells[ci].Body[0] * 3, current_rows, M);
            for (long long ij = ARows[i]; ij < ARows[i + 1] - 1; ij++) {
              current_near = Near.ColIndex[ij + Near.RowIndex[ybegin]];
              //std::cout<<"Current Near "<<current_near<<std::endl;
              long long next_near = Near.ColIndex[ij + 1 + Near.RowIndex[ybegin]];
              //std::cout<<"Next Near "<<next_near<<std::endl;
              long long add_cols = cells[next_near].Body[0] * 3 - cells[current_near].Body[1] * 3;
              //std::cout<<"Middle Rows "<<current_rows<<" "<<add_rows<<std::endl;
              //std::cout<<cells[current_near].Body[1] * 3<<" "<<cells[ci].Body[0] * 3<<" | "<<add_rows<<" "<<M<<std::endl;
              far.middleCols(current_cols, add_cols) = Mat_i.block(cells[current_near].Body[0] * 3, cells[ci].Body[1] * 3, M, add_cols);
              //far.middleRows(current_rows, add_rows) = mat.block(cells[current_near].Body[1] * 3, cells[ci].Body[0] * 3, add_rows, M);
              current_cols += add_cols;
            }
            current_near = Near.ColIndex[ARows[i + 1] - 1 + Near.RowIndex[ybegin]];
            long long add_cols = n_mat - cells[current_near].Body[1] * 3;
            //long long add_rows = mat.rows() - cells[current_near].Body[1] * 3;
            //std::cout<<"Bottom Rows "<<cells[current_near].Body[1] * 3<<" "<<cells[ci].Body[0] * 3<<" | "<<add_rows<<" "<<M<<std::endl;
            //std::cout<<"Current Near "<<current_near<<std::endl;
            far.rightCols(add_cols) = Mat_i.block(cells[current_near].Body[0] * 3, cells[ci].Body[1] * 3, M, add_cols);
            //far.bottomRows(add_rows) = mat.block(cells[current_near].Body[1] * 3, cells[ci].Body[0] * 3, add_rows, M);
            long long rank = compute_basis(far.transpose(), epi, S_ind[i + ibegin], S_ind_orig[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
            //std::cout<<"Rank "<<rank<<std::endl;
            /*for (int c = 0; c < far.cols(); ++c) {
              double col_norm = far.col(c).norm();
              long long count = 0;
              for (int r = 0; r < far.rows(); ++r)
                if (std::abs(far(r,c)) >= threshold * col_norm)
                  count++;
              std::cout<<"Col "<< c <<": " << count<<", Density: "<< ((double)count)/(far.rows())<<std::endl;
            }*/
            DimsLr[i + ibegin] = rank;
          }
        }
      }
    }
    std::cout<<"finished loop"<<std::endl;
    // Note that this call will change the actual contents of Ssizes
    // so I am not sure what they contain afterwards
    comm.dataSizesToNeighborOffsets(Ssizes.data());
    comm.neighbor_bcast(S_ind[0], Ssizes.data());
    comm.neighbor_bcast(S_ind_orig[0], Ssizes.data());
    std::cout<<"Finished bcast"<<std::endl;
    //std::cout<<"Ssizes: ";
    //for (auto& val : Ssizes)
    //  std::cout<<val<<", ";
    //std::cout<<std::endl;

    for (long long i = 0; i < nodes; i++) {
      std::cout<<"Node "<<i<<std::endl;
      // Generate far field for the upper levels
      if (localChildOffsets[i+1] <= localChildOffsets[i]) {
        continue;
      }

      long long M = Dims[i + ibegin];
      long long childi = localChildOffsets[i];
      long long cendi = localChildOffsets[i + 1];
      // get the corresponding M matrix (as a reference)
      Eigen::Map<Eigen::MatrixXcd> Mat_i(Mat[i + ibegin], M, n_mat - M);
      std::cout<<"Mat size: "<<Mat_i.rows() * Mat_i.cols()<<" "<<M<<" x "<<Mat_i.cols()<<std::endl;

       std::vector<long long> far_field(Dims);
      if (1. <= epi) {
        // HSS basis
        far_field[i] = 0;
      } else {
        // H2 basis
        for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
          far_field[ACols[ij]] = 0;
        }
      }
      auto far_cols = std::reduce(far_field.begin(), far_field.end());
      // only compute the far field if it exists
      if (far_cols) {
        std::vector<long long> FS_ind(far_cols);
        long long start = 0;
        long long corr_start;
        for (long long ij = 0; ij < nodes; ij++) {
          if (far_field[ij]) {
            //std::cout<<"Far field "<<ij<<std::endl;
            std::copy(S_ind[ij + ibegin], S_ind[ij + ibegin] + far_field[ij], &FS_ind[start]);
            start += far_field[ij];
          } else {
            corr_start = start;
          }
        }
        // todo there will be gaps in the far field where ij is 0
        // we need to correct the indices in that case
        // we don't need to correct if the previous level was the leaf level though
        std::cout<<"Far cols before: "<<far_cols<<" "<<far_cols * M<<": ";
        for (long long k = 0; k < far_cols; k++) {
          std::cout<<FS_ind[k]<<", ";
        }
        std::cout<<std::endl;
        if (!lowerA.lowest) {
          std::cout<<"Triggered correction from "<<start<<std::endl;
          for (long long k = corr_start; k < far_cols; k++) {
          // is this correct?
            FS_ind[k] -= M;
          }
        }


        std::cout<<"Far cols after: "<<far_cols<<" "<<far_cols * M<<": ";
        for (long long k = 0; k < far_cols; k++) {
          std::cout<<FS_ind[k]<<", ";
        }
        std::cout<<std::endl;
        // now we have the indices for the far field columns
        // create the far field in Mat_i from each child
        long long offset = 0;
        for (long long y = childi; y < cendi; y++) {
          std::cout<<"Child "<<y<<std::endl;
          Matrix_t F(&Mat_i(offset, 0), lowerA.DimsLr[y], far_cols, Stride_t(M, 1));
          std::cout<<"F allocated "<<F.rows()<<" x "<<F.cols()<<std::endl;
          //Eigen::Map<Eigen::MatrixXcd> F(Mat_i, lowerA.DimsLr[y], far_cols);
          // it seems like i need to store the dimensions of the far field somewhere
          // Last dimension of L is too long on the second level, 786 vs 704
          Eigen::Map<Eigen::MatrixXcd> L(lowerA.Mat[y], lowerA.Dims[y], lowerA.Cols[i]);
          std::cout<<"L generated "<<L.rows()<<" x "<<L.cols()<<std::endl;
          //todo : S_indices are global, but we need them locally
          std::cout<<"Indices ";
          for (int k = 0; k < lowerA.DimsLr[y]; ++k)
            std::cout<<*(S_ind[i+ibegin] +k +offset)<<" ";
          std::cout<<std::endl;
          // We should have already copied the indices of interest from the lower level to S_ind
          // todo: the conversion to local indexing works only for the first intermediate level,
          // because on the second level we can have a range of 120 but only 60 rows
          gen_matrix(L, lowerA.DimsLr[y], far_cols, lowerA.S_ind[y], FS_ind.data(), lowerA.dim_offsets[y], F);
          std::cout<<"L assigned to F"<<std::endl;
          offset += lowerA.DimsLr[y];
        }
        // reset S_ind to local indices
        std::iota(S_ind[i + ibegin], S_ind[i + ibegin + 1], dim_offsets[i]);
        long long rank = compute_basis(Mat_i.transpose(), epi, S_ind[i + ibegin], S_ind_orig[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
        std::cout<<"Rank "<<rank<<std::endl;
        DimsLr[i + ibegin] = rank;
      }
    }

    comm.dataSizesToNeighborOffsets(Qsizes.data());
    comm.neighbor_bcast(DimsLr.data(), neighbor_ones.data());
    // we need to communicate S again because the order has changed in the above
    // compute_basis() call
    comm.neighbor_bcast(S_ind[0], Ssizes.data());
    comm.neighbor_bcast(S_ind_orig[0], Ssizes.data());
    comm.neighbor_bcast(Q[0], Qsizes.data());
    comm.neighbor_bcast(R[0], Qsizes.data());
  }

  if (std::reduce(DimsLr.begin(), DimsLr.end())) {
    std::vector<long long> Csizes(CRows[nodes]);
    for (long long i = 0; i < nodes; i++)
      std::transform(CCols.begin() + CRows[i], CCols.begin() + CRows[i + 1], Csizes.begin() + CRows[i],
        [&](long long col) { return DimsLr[i + ibegin] * DimsLr[col]; });
    C.alloc(CRows[nodes], Csizes.data());

    std::vector<long long> Usizes(nodes);
    std::transform(&Dims[ibegin], &Dims[ibegin + nodes], &DimsLr[ibegin], Usizes.begin(), std::multiplies<long long>());
    U.alloc(nodes, Usizes.data());
    Z.alloc(xlen, DimsLr.data());
    W.alloc(xlen, DimsLr.data());

    //std::cout<<"Part2"<<std::endl;
    for (long long i = 0; i < nodes; i++) {
      //std::cout<<"Node "<<i<<std::endl;
      long long y = i + ibegin;
      long long M = DimsLr[y];
      //std::cout<<"M "<<M<<std::endl;
      Matrix_t Ry(R[y], M, M, Stride_t(Dims[y], 1));
      Eigen::Map<Eigen::MatrixXcd>(U[i], Dims[y], M) = Eigen::Map<Eigen::MatrixXcd>(Q[y], Dims[y], M);

      for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
        long long x = CCols[ij];
        //std::cout<<"x "<<x<<std::endl;
        long long N = DimsLr[CCols[ij]];
        //std::cout<<"N "<<N<<std::endl;
        Matrix_t Rx(R[x], N, N, Stride_t(Dims[x], 1));

        Eigen::Map<Eigen::MatrixXcd> Cyx(C[ij], M, N);
        if (1. <= epi) {
          Eigen::MatrixXcd Ayx(M, N);
          // todo create this matrix and the one below
          //gen_matrix(eval, M, N, S[y], S[x], Ayx.data());
          //gen_matrix(Mat_i, M, N, S_ind[y], S_ind[x], Ayx);
          Cyx.noalias() = Ry.triangularView<Eigen::Upper>() * Ayx * Rx.transpose().triangularView<Eigen::Lower>();
        }
        else
          ;
          //gen_matrix(Mat_i, M, N, S_ind[y], S_ind[x], Cyx);
          //gen_matrix(eval, M, N, S[y], S[x], Cyx.data());
      }
    }
  }

  std::cout <<"S_ind_orig: "<<S_ind.size()<<std::endl;
  for (long long i = 0; i < S_ind.size(); ++i)
    std::cout<<*(S_ind_orig[0] + i)<<", ";
  std::cout<<std::endl;
  std::cout <<"S_ind: "<<S_ind.size()<<std::endl;
  for (long long i = 0; i < S_ind.size(); ++i)
    std::cout<<*(S_ind[0] + i)<<", ";
  std::cout<<std::endl;
  NbXoffsets.insert(NbXoffsets.begin(), Dims.begin(), Dims.end());
  NbXoffsets.erase(NbXoffsets.begin() + comm.dataSizesToNeighborOffsets(NbXoffsets.data()), NbXoffsets.end());
  NbZoffsets.insert(NbZoffsets.begin(), DimsLr.begin(), DimsLr.end());
  NbZoffsets.erase(NbZoffsets.begin() + comm.dataSizesToNeighborOffsets(NbZoffsets.data()), NbZoffsets.end());
}

void H2Matrix::construct_proto(const MatrixGenerator& matgen, double epi, const Cell cells[], const CSR& Near, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm, const double omega, const double scale) {
  // number of cells on this level (this process and neighbors)
  long long xlen = comm.lenNeighbors();
  // index of the first cell for this process on this level
  long long ibegin = comm.oLocal();
  // number of cells for this process on this level
  long long nodes = comm.lenLocal();
  // index of the first cell for this process in the global cell array
  long long ybegin = comm.oGlobal();
  std::cout<<"START"<<std::endl;

  // dimensions for each cell on this level
  Dims.resize(xlen, 0);
  // LR dimensions for each cell on this level
  DimsLr.resize(xlen, 0);
  // stide for what?
  UpperStride.resize(nodes, 0);

  // get the Nearfield indices CSR
  ARows.insert(ARows.begin(), comm.ARowOffsets.begin(), comm.ARowOffsets.end());
  ACols.insert(ACols.begin(), comm.AColumns.begin(), comm.AColumns.end());
  // get the far-field indices CSR
  CRows.insert(CRows.begin(), comm.CRowOffsets.begin(), comm.CRowOffsets.end());
  CCols.insert(CCols.begin(), comm.CColumns.begin(), comm.CColumns.end());
  // ?
  NA.resize(ARows[nodes], -1);

  // get the number of local children
  long long localChildLen = cells[ybegin + nodes - 1].Child[1] - cells[ybegin].Child[0];
  std::vector<long long> localChildOffsets(nodes + 1, -1);
  
  if (0 < localChildLen) {
    long long lowerBegin = lowerComm.oLocal() + comm.LowerX;
    long long localChildIndex = lowerBegin - cells[ybegin].Child[0];
    std::transform(&cells[ybegin], &cells[ybegin + nodes], localChildOffsets.begin() + 1, [=](const Cell& c) { return localChildIndex + c.Child[1]; });
    localChildOffsets[0] = lowerBegin;

    std::vector<long long> ranks_offsets(localChildLen + 1);
    std::inclusive_scan(lowerA.DimsLr.begin() + localChildOffsets[0], lowerA.DimsLr.begin() + localChildOffsets[nodes], ranks_offsets.begin() + 1);
    ranks_offsets[0] = 0;

    std::transform(localChildOffsets.begin(), localChildOffsets.begin() + nodes, localChildOffsets.begin() + 1, &Dims[ibegin],
      [&](long long start, long long end) { return ranks_offsets[end - lowerBegin] - ranks_offsets[start - lowerBegin]; });

    lenX = ranks_offsets.back();
    LowerZ = std::reduce(lowerA.DimsLr.begin(), lowerA.DimsLr.begin() + lowerBegin, 0ll);
    n_mat = std::reduce(&Dims[ibegin], &Dims[ibegin + nodes], 0ll);
    // Mat on the intermediate levels stores only the far field matrices
    std::vector<long long> row_offsets(&Dims[ibegin], &Dims[ibegin + nodes]);
    Cols.resize(nodes, 0);
    std::cout<<"Mat sizes: ";
    for (size_t i = 0; i < nodes; ++i) {
      Cols[i] = n_mat - row_offsets[i];
      row_offsets[i] *= Cols[i];
      std::cout<<row_offsets[i]<<", ";
    }
    std::cout<<std::endl;
    // todo the above is only correct for the next level after the leaf level?
    Mat.alloc(nodes, row_offsets.data());
  }
  else {
    // only for leaf level
    // Dims stores the number of particels for each cell (multiplied by 3)
    std::transform(&cells[ybegin], &cells[ybegin + nodes], &Dims[ibegin], [](const Cell& c) { return (c.Body[1] - c.Body[0]) * 3; });
    // total number of particles on this process
    lenX = std::reduce(&Dims[ibegin], &Dims[ibegin + nodes]);
    LowerZ = 0;
    n_mat = matgen.get_num_total() * 3;
    std::vector<long long> row_offsets(&Dims[ibegin], &Dims[ibegin + nodes]);
    for (auto& offset : row_offsets)
      offset *= n_mat;
    Mat.alloc(nodes, row_offsets.data());
    // I do this in a loop now, but I could also just create one big matrix
    // in one go if I reduce the dimensions first
    // this would however, change the data layout and I would need to account for that
    for (long long i = 0; i < nodes; ++i) {
      matgen.gen_matrix_sorted(Mat[i], cells[ybegin + i].Body[0], Dims[ibegin + i] / 3, omega, scale);
    }
    Cols.resize(nodes, n_mat);
     // we should also calculate the scale distributed, but lets keep that for later
  }

  std::vector<long long> neighbor_ones(xlen, 1ll);
  comm.dataSizesToNeighborOffsets(neighbor_ones.data());
  comm.neighbor_bcast(Dims.data(), neighbor_ones.data());
  X.alloc(xlen, Dims.data());
  Y.alloc(xlen, Dims.data());
  // S stores the indices
  std::vector<long long> Ssizes(Dims);
  S_ind.alloc(xlen, Ssizes.data());
  S_ind_orig.alloc(xlen, Ssizes.data());
  dim_offsets.resize(Dims.size());
  std::exclusive_scan(Dims.begin(), Dims.end(), dim_offsets.begin(), 0);
  std::cout<<"Dim offsets ";
  for (size_t i = 0; i < dim_offsets.size(); ++i)
    std::cout<<dim_offsets[i]<<", ";
  std::cout<<std::endl;
  std::cout<<"Lowest "<<lowest<<" "<<lowerA.lowest<<std::endl;

  std::vector<long long> Qsizes(xlen, 0);
  // Qs and Rs are square matrices
  std::transform(Dims.begin(), Dims.end(), Qsizes.begin(), [](const long long d) { return d * d; });
  Q.alloc(xlen, Qsizes.data());
  R.alloc(xlen, Qsizes.data());

  // near field blocks (i.e. dense matrices on the leaf level), not necessarily square
  std::vector<long long> Asizes(ARows[nodes]);
  for (long long i = 0; i < nodes; i++)
    std::transform(ACols.begin() + ARows[i], ACols.begin() + ARows[i + 1], Asizes.begin() + ARows[i],
      [&](long long col) { return Dims[i + ibegin] * Dims[col]; });
  A.alloc(ARows[nodes], Asizes.data());

  typedef Eigen::Stride<Eigen::Dynamic, 1> Stride_t;
  typedef Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Stride_t> Matrix_t; 

  // skip blocks that contain no points (because they are split further)
  if (std::reduce(Dims.begin(), Dims.end())) {
    // index of the first cell for this process on this level
    long long pbegin = lowerComm.oLocal();
    // number of cells for this process/level
    long long pend = pbegin + lowerComm.lenLocal();

    // loop over all nodes
    for (long long i = 0; i < nodes; i++) {
      //std::cout<<"Node "<<i<<std::endl;
      // number of rows in that cell
      long long M = Dims[i + ibegin];
      //std::cout<<"Rows "<<M<<std::endl;
      long long childi = localChildOffsets[i];
      long long cendi = localChildOffsets[i + 1];
      // get the corresponding Q matrix (as a reference)
      Eigen::Map<Eigen::MatrixXcd> Qi(Q[i + ibegin], M, M);
      // initialize the local indices
      //long long offset_ind = std::reduce(&Dims[ibegin], &Dims[ibegin+i]);
      //std::iota(S_ind_local[i + ibegin], S_ind_local[i + ibegin + 1], offset_ind);

      // for all children (i.e. only on the intermediate levels)
      for (long long y = childi; y < cendi; y++) {
        long long offset_y = std::reduce(&lowerA.DimsLr[childi], &lowerA.DimsLr[y]);
        long long ny = lowerA.DimsLr[y];
        // S_ind already has been broadcast on the lower level, so this is fine
        std::copy(lowerA.S_ind[y], lowerA.S_ind[y] + ny, &(S_ind[i + ibegin])[offset_y]);
        std::copy(lowerA.S_ind_orig[y], lowerA.S_ind_orig[y] + ny, &(S_ind_orig[i + ibegin])[offset_y]);

        Matrix_t Ry(lowerA.R[y], ny, ny, Stride_t(lowerA.Dims[y], 1));
        Qi.block(offset_y, offset_y, ny, ny) = Ry;

        if (pbegin <= y && y < pend && 0 < M) {
          long long py = y - pbegin;
          lowerA.UpperStride[py] = M;

          for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
            long long j_global = Near.ColIndex[ij + Near.RowIndex[ybegin]];
            long long childj = lowerComm.iLocal(cells[j_global].Child[0]);
            long long cendj = (0 <= childj) ? (childj + cells[j_global].Child[1] - cells[j_global].Child[0]) : -1;

            for (long long x = childj; x < cendj; x++) {
              long long offset_x = std::reduce(&lowerA.DimsLr[childj], &lowerA.DimsLr[x]);
              long long nx = lowerA.DimsLr[x];
              long long lowN = lookupIJ(lowerA.ARows, lowerA.ACols, py, x);
              long long lowC = lookupIJ(lowerA.CRows, lowerA.CCols, py, x);
              std::complex<double>* dp = A[ij] + offset_y + offset_x * M;
              if (0 <= lowN)
                lowerA.NA[lowN] = std::distance(A[0], dp);
              else if (0 <= lowC)
                Matrix_t(dp, ny, nx, Stride_t(M, 1)) = Eigen::Map<Eigen::MatrixXcd>(lowerA.C[lowC], ny, nx);
            }
          }
        }
      }
       
      // leaf level (i.e. no children)
      if (cendi <= childi) {
        long long ci = i + ybegin;
        // numbering the indices locally will not work for the far field
        std::iota(S_ind[i + ibegin], S_ind[i + ibegin + 1], cells[ci].Body[0] * 3);
        std::iota(S_ind_orig[i + ibegin], S_ind_orig[i + ibegin + 1], cells[ci].Body[0] * 3);
        Qi = Eigen::MatrixXcd::Identity(M, M);

        long long far_cols = n_mat;
        Eigen::Map<Eigen::MatrixXcd> Mat_i(Mat[i], M, n_mat);
        //std::cout<<Mat_i.rows()<<" "<<Mat_i.cols()<<std::endl;
        // generate the near field aka dense matrices in A
        for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
          //std::cout<<"J "<<ij<<std::endl;
          long long N = Dims[ACols[ij]];
          far_cols -= N;
          long long cj = Near.ColIndex[ij + Near.RowIndex[ybegin]];
          //memcpy(A[ij], M[i] + cells[cj].Body[0] * 3, sizeof(std::complex<double>) * M *N);
          Eigen::Map<Eigen::MatrixXcd> A_ij(A[ij], M, N);
          // we could optimize this, as we don't necessarily need to make a copy here
          A_ij = Mat_i.block(0, cells[cj].Body[0] * 3, M, N);
        }
        if (1. <= epi) {
          // build an HSS basis
          far_cols = n_mat - M;
          Eigen::MatrixXcd far(M, far_cols);
          long long diag = Near.ColIndex[ARows[i] + Near.RowIndex[ybegin]];
          long long left = cells[ci].Body[0] * 3;
          long long right = cells[ci].Body[1] * 3;
          //std::cout<<"Left Cols "<<0<<" "<<cells[ci].Body[0] * 3<<" | "<<M<<std::endl;
          far.leftCols(left) = Mat_i.leftCols(left);
          far.rightCols(n_mat - right) = Mat_i.rightCols(n_mat - right);
          long long rank = compute_basis(far.transpose(), epi, S_ind[i + ibegin], S_ind_orig[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
          //std::cout<<"Rank "<<rank<<std::endl;
          /*for (int c = 0; c < far.cols(); ++c) {
            double col_norm = far.col(c).norm();
            long long count = 0;
            for (int r = 0; r < far.rows(); ++r)
              if (std::abs(far(r,c)) >= threshold * col_norm)
                count++;
            std::cout<<"Col "<< c <<": " << count<<", Density: "<< ((double)count)/(far.rows())<<std::endl;
          }*/
          DimsLr[i + ibegin] = rank;
        } else {
          // build an H2 basis
          // generate the far field only if it exists
          if (far_cols > 0) {
            // not tested after transpose
            Eigen::MatrixXcd far(M, far_cols);
            long long current_near = Near.ColIndex[ARows[i] + Near.RowIndex[ybegin]];
            //std::cout<<"Current Near "<<current_near<<std::endl;
            //std::cout<<ARows[i]<<" "<<ARows[i+1]<<std::endl;
            long long current_cols = cells[current_near].Body[0] * 3;
            //std::cout<<"Top Rows "<<0<<" "<<cells[ci].Body[0] * 3<<" | "<<current_rows<<" "<<M<<std::endl;
            far.leftCols(current_cols) = Mat_i.block(cells[ci].Body[0] * 3, 0, M, current_cols);
            //far.topRows(current_rows) = mat.block(0, cells[ci].Body[0] * 3, current_rows, M);
            for (long long ij = ARows[i]; ij < ARows[i + 1] - 1; ij++) {
              current_near = Near.ColIndex[ij + Near.RowIndex[ybegin]];
              //std::cout<<"Current Near "<<current_near<<std::endl;
              long long next_near = Near.ColIndex[ij + 1 + Near.RowIndex[ybegin]];
              //std::cout<<"Next Near "<<next_near<<std::endl;
              long long add_cols = cells[next_near].Body[0] * 3 - cells[current_near].Body[1] * 3;
              //std::cout<<"Middle Rows "<<current_rows<<" "<<add_rows<<std::endl;
              //std::cout<<cells[current_near].Body[1] * 3<<" "<<cells[ci].Body[0] * 3<<" | "<<add_rows<<" "<<M<<std::endl;
              far.middleCols(current_cols, add_cols) = Mat_i.block(cells[current_near].Body[0] * 3, cells[ci].Body[1] * 3, M, add_cols);
              //far.middleRows(current_rows, add_rows) = mat.block(cells[current_near].Body[1] * 3, cells[ci].Body[0] * 3, add_rows, M);
              current_cols += add_cols;
            }
            current_near = Near.ColIndex[ARows[i + 1] - 1 + Near.RowIndex[ybegin]];
            long long add_cols = n_mat - cells[current_near].Body[1] * 3;
            //long long add_rows = mat.rows() - cells[current_near].Body[1] * 3;
            //std::cout<<"Bottom Rows "<<cells[current_near].Body[1] * 3<<" "<<cells[ci].Body[0] * 3<<" | "<<add_rows<<" "<<M<<std::endl;
            //std::cout<<"Current Near "<<current_near<<std::endl;
            far.rightCols(add_cols) = Mat_i.block(cells[current_near].Body[0] * 3, cells[ci].Body[1] * 3, M, add_cols);
            //far.bottomRows(add_rows) = mat.block(cells[current_near].Body[1] * 3, cells[ci].Body[0] * 3, add_rows, M);
            long long rank = compute_basis(far.transpose(), epi, S_ind[i + ibegin], S_ind_orig[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
            //std::cout<<"Rank "<<rank<<std::endl;
            /*for (int c = 0; c < far.cols(); ++c) {
              double col_norm = far.col(c).norm();
              long long count = 0;
              for (int r = 0; r < far.rows(); ++r)
                if (std::abs(far(r,c)) >= threshold * col_norm)
                  count++;
              std::cout<<"Col "<< c <<": " << count<<", Density: "<< ((double)count)/(far.rows())<<std::endl;
            }*/
            DimsLr[i + ibegin] = rank;
          }
        }
      }
    }
    std::cout<<"finished loop"<<std::endl;
    // Note that this call will change the actual contents of Ssizes
    // so I am not sure what they contain afterwards
    comm.dataSizesToNeighborOffsets(Ssizes.data());
    comm.neighbor_bcast(S_ind[0], Ssizes.data());
    comm.neighbor_bcast(S_ind_orig[0], Ssizes.data());
    std::cout<<"Finished bcast"<<std::endl;
    //std::cout<<"Ssizes: ";
    //for (auto& val : Ssizes)
    //  std::cout<<val<<", ";
    //std::cout<<std::endl;

    for (long long i = 0; i < nodes; i++) {
      std::cout<<"Node "<<i<<std::endl;
      // Generate far field for the upper levels
      if (localChildOffsets[i+1] <= localChildOffsets[i]) {
        continue;
      }

      long long M = Dims[i + ibegin];
      long long childi = localChildOffsets[i];
      long long cendi = localChildOffsets[i + 1];
      // get the corresponding M matrix (as a reference)
      Eigen::Map<Eigen::MatrixXcd> Mat_i(Mat[i + ibegin], M, n_mat - M);
      std::cout<<"Mat size: "<<Mat_i.rows() * Mat_i.cols()<<" "<<M<<" x "<<Mat_i.cols()<<std::endl;

       std::vector<long long> far_field(Dims);
      if (1. <= epi) {
        // HSS basis
        far_field[i] = 0;
      } else {
        // H2 basis
        for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
          far_field[ACols[ij]] = 0;
        }
      }
      auto far_cols = std::reduce(far_field.begin(), far_field.end());
      // only compute the far field if it exists
      if (far_cols) {
        std::vector<long long> FS_ind(far_cols);
        long long start = 0;
        long long corr_start;
        for (long long ij = 0; ij < nodes; ij++) {
          if (far_field[ij]) {
            //std::cout<<"Far field "<<ij<<std::endl;
            std::copy(S_ind[ij + ibegin], S_ind[ij + ibegin] + far_field[ij], &FS_ind[start]);
            start += far_field[ij];
          } else {
            corr_start = start;
          }
        }
        // todo there will be gaps in the far field where ij is 0
        // we need to correct the indices in that case
        // we don't need to correct if the previous level was the leaf level though
        std::cout<<"Far cols before: "<<far_cols<<" "<<far_cols * M<<": ";
        for (long long k = 0; k < far_cols; k++) {
          std::cout<<FS_ind[k]<<", ";
        }
        std::cout<<std::endl;
        if (!lowerA.lowest) {
          std::cout<<"Triggered correction from "<<start<<std::endl;
          for (long long k = corr_start; k < far_cols; k++) {
          // is this correct?
            FS_ind[k] -= M;
          }
        }


        std::cout<<"Far cols after: "<<far_cols<<" "<<far_cols * M<<": ";
        for (long long k = 0; k < far_cols; k++) {
          std::cout<<FS_ind[k]<<", ";
        }
        std::cout<<std::endl;
        // now we have the indices for the far field columns
        // create the far field in Mat_i from each child
        long long offset = 0;
        for (long long y = childi; y < cendi; y++) {
          std::cout<<"Child "<<y<<std::endl;
          Matrix_t F(&Mat_i(offset, 0), lowerA.DimsLr[y], far_cols, Stride_t(M, 1));
          std::cout<<"F allocated "<<F.rows()<<" x "<<F.cols()<<std::endl;
          //Eigen::Map<Eigen::MatrixXcd> F(Mat_i, lowerA.DimsLr[y], far_cols);
          // it seems like i need to store the dimensions of the far field somewhere
          // Last dimension of L is too long on the second level, 786 vs 704
          Eigen::Map<Eigen::MatrixXcd> L(lowerA.Mat[y], lowerA.Dims[y], lowerA.Cols[i]);
          std::cout<<"L generated "<<L.rows()<<" x "<<L.cols()<<std::endl;
          //todo : S_indices are global, but we need them locally
          std::cout<<"Indices ";
          for (int k = 0; k < lowerA.DimsLr[y]; ++k)
            std::cout<<*(S_ind[i+ibegin] +k +offset)<<" ";
          std::cout<<std::endl;
          // We should have already copied the indices of interest from the lower level to S_ind
          // todo: the conversion to local indexing works only for the first intermediate level,
          // because on the second level we can have a range of 120 but only 60 rows
          gen_matrix(L, lowerA.DimsLr[y], far_cols, lowerA.S_ind[y], FS_ind.data(), lowerA.dim_offsets[y], F);
          std::cout<<"L assigned to F"<<std::endl;
          offset += lowerA.DimsLr[y];
        }
        // reset S_ind to local indices
        std::iota(S_ind[i + ibegin], S_ind[i + ibegin + 1], dim_offsets[i]);
        long long rank = compute_basis(Mat_i.transpose(), epi, S_ind[i + ibegin], S_ind_orig[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
        std::cout<<"Rank "<<rank<<std::endl;
        DimsLr[i + ibegin] = rank;
      }
    }

    comm.dataSizesToNeighborOffsets(Qsizes.data());
    comm.neighbor_bcast(DimsLr.data(), neighbor_ones.data());
    // we need to communicate S again because the order has changed in the above
    // compute_basis() call
    comm.neighbor_bcast(S_ind[0], Ssizes.data());
    comm.neighbor_bcast(S_ind_orig[0], Ssizes.data());
    comm.neighbor_bcast(Q[0], Qsizes.data());
    comm.neighbor_bcast(R[0], Qsizes.data());
  }

  if (std::reduce(DimsLr.begin(), DimsLr.end())) {
    std::vector<long long> Csizes(CRows[nodes]);
    for (long long i = 0; i < nodes; i++)
      std::transform(CCols.begin() + CRows[i], CCols.begin() + CRows[i + 1], Csizes.begin() + CRows[i],
        [&](long long col) { return DimsLr[i + ibegin] * DimsLr[col]; });
    C.alloc(CRows[nodes], Csizes.data());

    std::vector<long long> Usizes(nodes);
    std::transform(&Dims[ibegin], &Dims[ibegin + nodes], &DimsLr[ibegin], Usizes.begin(), std::multiplies<long long>());
    U.alloc(nodes, Usizes.data());
    Z.alloc(xlen, DimsLr.data());
    W.alloc(xlen, DimsLr.data());

    //std::cout<<"Part2"<<std::endl;
    for (long long i = 0; i < nodes; i++) {
      //std::cout<<"Node "<<i<<std::endl;
      long long y = i + ibegin;
      long long M = DimsLr[y];
      //std::cout<<"M "<<M<<std::endl;
      Matrix_t Ry(R[y], M, M, Stride_t(Dims[y], 1));
      Eigen::Map<Eigen::MatrixXcd>(U[i], Dims[y], M) = Eigen::Map<Eigen::MatrixXcd>(Q[y], Dims[y], M);

      for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
        long long x = CCols[ij];
        //std::cout<<"x "<<x<<std::endl;
        long long N = DimsLr[CCols[ij]];
        //std::cout<<"N "<<N<<std::endl;
        Matrix_t Rx(R[x], N, N, Stride_t(Dims[x], 1));

        Eigen::Map<Eigen::MatrixXcd> Cyx(C[ij], M, N);
        if (1. <= epi) {
          Eigen::MatrixXcd Ayx(M, N);
          // todo create this matrix and the one below
          //gen_matrix(eval, M, N, S[y], S[x], Ayx.data());
          //gen_matrix(Mat_i, M, N, S_ind[y], S_ind[x], Ayx);
          Cyx.noalias() = Ry.triangularView<Eigen::Upper>() * Ayx * Rx.transpose().triangularView<Eigen::Lower>();
        }
        else
          ;
          //gen_matrix(Mat_i, M, N, S_ind[y], S_ind[x], Cyx);
          //gen_matrix(eval, M, N, S[y], S[x], Cyx.data());
      }
    }
  }

  std::cout <<"S_ind_orig: "<<S_ind.size()<<std::endl;
  for (long long i = 0; i < S_ind.size(); ++i)
    std::cout<<*(S_ind_orig[0] + i)<<", ";
  std::cout<<std::endl;
  std::cout <<"S_ind: "<<S_ind.size()<<std::endl;
  for (long long i = 0; i < S_ind.size(); ++i)
    std::cout<<*(S_ind[0] + i)<<", ";
  std::cout<<std::endl;
  NbXoffsets.insert(NbXoffsets.begin(), Dims.begin(), Dims.end());
  NbXoffsets.erase(NbXoffsets.begin() + comm.dataSizesToNeighborOffsets(NbXoffsets.data()), NbXoffsets.end());
  NbZoffsets.insert(NbZoffsets.begin(), DimsLr.begin(), DimsLr.end());
  NbZoffsets.erase(NbZoffsets.begin() + comm.dataSizesToNeighborOffsets(NbZoffsets.data()), NbZoffsets.end());
}

void H2Matrix::construct(const Eigen::Ref<const Eigen::MatrixXcd> &mat, double epi, const Cell cells[], const CSR& Near, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm) {
  // number of cells on this level
  long long xlen = comm.lenNeighbors();
  // index of the first cell for this porcess on this level
  long long ibegin = comm.oLocal();
  // number of cells for this process on this level
  long long nodes = comm.lenLocal();
  // index of the first cell for this process in the global cell array
  long long ybegin = comm.oGlobal();

  // dimensions for each cell on this level
  Dims.resize(xlen, 0);
  // LR dimensions for each cell on this level
  DimsLr.resize(xlen, 0);
  // stide for what?
  UpperStride.resize(nodes, 0);

  // get the Nearfield indices CSR
  ARows.insert(ARows.begin(), comm.ARowOffsets.begin(), comm.ARowOffsets.end());
  ACols.insert(ACols.begin(), comm.AColumns.begin(), comm.AColumns.end());
  // get the far-field indices CSR
  CRows.insert(CRows.begin(), comm.CRowOffsets.begin(), comm.CRowOffsets.end());
  CCols.insert(CCols.begin(), comm.CColumns.begin(), comm.CColumns.end());
  // ?
  NA.resize(ARows[nodes], -1);

  // get the number of local children
  long long localChildLen = cells[ybegin + nodes - 1].Child[1] - cells[ybegin].Child[0];
  std::vector<long long> localChildOffsets(nodes + 1, -1);
  
  if (0 < localChildLen) {
    long long lowerBegin = lowerComm.oLocal() + comm.LowerX;
    long long localChildIndex = lowerBegin - cells[ybegin].Child[0];
    std::transform(&cells[ybegin], &cells[ybegin + nodes], localChildOffsets.begin() + 1, [=](const Cell& c) { return localChildIndex + c.Child[1]; });
    localChildOffsets[0] = lowerBegin;

    std::vector<long long> ranks_offsets(localChildLen + 1);
    std::inclusive_scan(lowerA.DimsLr.begin() + localChildOffsets[0], lowerA.DimsLr.begin() + localChildOffsets[nodes], ranks_offsets.begin() + 1);
    ranks_offsets[0] = 0;

    std::transform(localChildOffsets.begin(), localChildOffsets.begin() + nodes, localChildOffsets.begin() + 1, &Dims[ibegin],
      [&](long long start, long long end) { return ranks_offsets[end - lowerBegin] - ranks_offsets[start - lowerBegin]; });

    lenX = ranks_offsets.back();
    LowerZ = std::reduce(lowerA.DimsLr.begin(), lowerA.DimsLr.begin() + lowerBegin, 0ll);
  }
  else {
    // only for leaf level
    // Dims stores the number of particels for each cell (multiplied by 3)
    std::transform(&cells[ybegin], &cells[ybegin + nodes], &Dims[ibegin], [](const Cell& c) { return (c.Body[1] - c.Body[0]) * 3; });
    // total number of particles on this process
    lenX = std::reduce(&Dims[ibegin], &Dims[ibegin + nodes]);
    LowerZ = 0;
    //std::cout<<"Rows on this node: "<<lenX<<std::endl;
  }

  std::vector<long long> neighbor_ones(xlen, 1ll);
  comm.dataSizesToNeighborOffsets(neighbor_ones.data());
  comm.neighbor_bcast(Dims.data(), neighbor_ones.data());
  X.alloc(xlen, Dims.data());
  Y.alloc(xlen, Dims.data());

  std::vector<long long> Qsizes(xlen, 0);
  // Qs and Rs are square matrices
  std::transform(Dims.begin(), Dims.end(), Qsizes.begin(), [](const long long d) { return d * d; });
  Q.alloc(xlen, Qsizes.data());
  R.alloc(xlen, Qsizes.data());

  // S stores the indices
  std::vector<long long> Ssizes(xlen);
  std::transform(Dims.begin(), Dims.end(), Ssizes.begin(), [](const long long d) { return d; });
  S_ind.alloc(xlen, Ssizes.data());

  // near field blocks (i.e. dense matrices on the leaf level), not necessarily square
  std::vector<long long> Asizes(ARows[nodes]);
  for (long long i = 0; i < nodes; i++)
    std::transform(ACols.begin() + ARows[i], ACols.begin() + ARows[i + 1], Asizes.begin() + ARows[i],
      [&](long long col) { return Dims[i + ibegin] * Dims[col]; });
  A.alloc(ARows[nodes], Asizes.data());

  typedef Eigen::Stride<Eigen::Dynamic, 1> Stride_t;
  typedef Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Stride_t> Matrix_t; 

  // skip blocks that contain no points (because they are split further)
  if (std::reduce(Dims.begin(), Dims.end())) {
    // index of the first cell for this process on this level
    long long pbegin = lowerComm.oLocal();
    // number of cells for this process/level
    long long pend = pbegin + lowerComm.lenLocal();

    // loop over all nodes
    for (long long i = 0; i < nodes; i++) {
      //std::cout<<"Node "<<i<<std::endl;
      // number of rows in that cell
      long long M = Dims[i + ibegin];
      //std::cout<<"Rows "<<M<<std::endl;
      long long childi = localChildOffsets[i];
      long long cendi = localChildOffsets[i + 1];
      // get the corresponding Q matrix (as a reference)
      Eigen::Map<Eigen::MatrixXcd> Qi(Q[i + ibegin], M, M);

      // for all children (i.e. only on the intermediate levels)
      for (long long y = childi; y < cendi; y++) {
        long long offset_y = std::reduce(&lowerA.DimsLr[childi], &lowerA.DimsLr[y]);
        long long ny = lowerA.DimsLr[y];
        std::copy(lowerA.S_ind[y], lowerA.S_ind[y] + (ny), &(S_ind[i + ibegin])[offset_y]);

        // todo check the lr dimensions
        Matrix_t Ry(lowerA.R[y], ny, ny, Stride_t(lowerA.Dims[y], 1));
        Qi.block(offset_y, offset_y, ny, ny) = Ry;

        if (pbegin <= y && y < pend && 0 < M) {
          long long py = y - pbegin;
          lowerA.UpperStride[py] = M;

          for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
            long long j_global = Near.ColIndex[ij + Near.RowIndex[ybegin]];
            long long childj = lowerComm.iLocal(cells[j_global].Child[0]);
            long long cendj = (0 <= childj) ? (childj + cells[j_global].Child[1] - cells[j_global].Child[0]) : -1;

            for (long long x = childj; x < cendj; x++) {
              long long offset_x = std::reduce(&lowerA.DimsLr[childj], &lowerA.DimsLr[x]);
              long long nx = lowerA.DimsLr[x];
              long long lowN = lookupIJ(lowerA.ARows, lowerA.ACols, py, x);
              long long lowC = lookupIJ(lowerA.CRows, lowerA.CCols, py, x);
              std::complex<double>* dp = A[ij] + offset_y + offset_x * M;
              if (0 <= lowN)
                lowerA.NA[lowN] = std::distance(A[0], dp);
              else if (0 <= lowC)
                Matrix_t(dp, ny, nx, Stride_t(M, 1)) = Eigen::Map<Eigen::MatrixXcd>(lowerA.C[lowC], ny, nx);
            }
          }
        }
      }
       
      // leaf level (i.e. no children)
      if (cendi <= childi) {
        long long ci = i + ybegin;
        std::iota(S_ind[i + ibegin], S_ind[i + ibegin + 1], cells[ci].Body[0] * 3);
        Qi = Eigen::MatrixXcd::Identity(M, M);

        long long far_cols = mat.cols();
        // generate the near field aka dense matrices in A
        for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
          //std::cout<<"J "<<ij<<std::endl;
          long long N = Dims[ACols[ij]];
          far_cols -= N;
          long long cj = Near.ColIndex[ij + Near.RowIndex[ybegin]];
          Eigen::Map<Eigen::MatrixXcd> A_ij(A[ij], M, N);
          A_ij = mat.block(cells[ci].Body[0] * 3, cells[cj].Body[0] * 3, M, N);
        }
        //double threshold = 1e-2;
        if (1. <= epi) {
          // build an HSS basis
          far_cols = mat.cols() - M;
          Eigen::MatrixXcd far(M, far_cols);
          long long diag = Near.ColIndex[ARows[i] + Near.RowIndex[ybegin]];
          long long left = cells[ci].Body[0] * 3;
          long long right = cells[ci].Body[1] * 3;
          //std::cout<<"Left Cols "<<0<<" "<<cells[ci].Body[0] * 3<<" | "<<M<<std::endl;
          far.leftCols(left) = mat.block(left, 0, M, left);
          far.rightCols(mat.cols() - right) = mat.block(left, right, M, mat.cols() - right);
          long long rank = compute_basis(far.transpose(), epi, S_ind[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
          //std::cout<<"Rank "<<rank<<std::endl;
          /*for (int c = 0; c < far.cols(); ++c) {
            double col_norm = far.col(c).norm();
            long long count = 0;
            for (int r = 0; r < far.rows(); ++r)
              if (std::abs(far(r,c)) >= threshold * col_norm)
                count++;
            std::cout<<"Col "<< c <<": " << count<<", Density: "<< ((double)count)/(far.rows())<<std::endl;
          }*/
          DimsLr[i + ibegin] = rank;
        } else {
          // build an H2 basis
          // generate the far field only if it exists
          if (far_cols > 0) {
            // not tested after transpose
            Eigen::MatrixXcd far(M, far_cols);
            long long current_near = Near.ColIndex[ARows[i] + Near.RowIndex[ybegin]];
            //std::cout<<"Current Near "<<current_near<<std::endl;
            //std::cout<<ARows[i]<<" "<<ARows[i+1]<<std::endl;
            long long current_cols = cells[current_near].Body[0] * 3;
            //std::cout<<"Top Rows "<<0<<" "<<cells[ci].Body[0] * 3<<" | "<<current_rows<<" "<<M<<std::endl;
            far.leftCols(current_cols) = mat.block(cells[ci].Body[0] * 3, 0, M, current_cols);
            //far.topRows(current_rows) = mat.block(0, cells[ci].Body[0] * 3, current_rows, M);
            for (long long ij = ARows[i]; ij < ARows[i + 1] - 1; ij++) {
              current_near = Near.ColIndex[ij + Near.RowIndex[ybegin]];
              //std::cout<<"Current Near "<<current_near<<std::endl;
              long long next_near = Near.ColIndex[ij + 1 + Near.RowIndex[ybegin]];
              //std::cout<<"Next Near "<<next_near<<std::endl;
              long long add_cols = cells[next_near].Body[0] * 3 - cells[current_near].Body[1] * 3;
              //std::cout<<"Middle Rows "<<current_rows<<" "<<add_rows<<std::endl;
              //std::cout<<cells[current_near].Body[1] * 3<<" "<<cells[ci].Body[0] * 3<<" | "<<add_rows<<" "<<M<<std::endl;
              far.middleCols(current_cols, add_cols) = mat.block(cells[current_near].Body[0] * 3, cells[ci].Body[1] * 3, M, add_cols);
              //far.middleRows(current_rows, add_rows) = mat.block(cells[current_near].Body[1] * 3, cells[ci].Body[0] * 3, add_rows, M);
              current_cols += add_cols;
            }
            current_near = Near.ColIndex[ARows[i + 1] - 1 + Near.RowIndex[ybegin]];
            long long add_cols = mat.cols() - cells[current_near].Body[1] * 3;
            //long long add_rows = mat.rows() - cells[current_near].Body[1] * 3;
            //std::cout<<"Bottom Rows "<<cells[current_near].Body[1] * 3<<" "<<cells[ci].Body[0] * 3<<" | "<<add_rows<<" "<<M<<std::endl;
            //std::cout<<"Current Near "<<current_near<<std::endl;
            far.rightCols(add_cols) = mat.block(cells[current_near].Body[0] * 3, cells[ci].Body[1] * 3, M, add_cols);
            //far.bottomRows(add_rows) = mat.block(cells[current_near].Body[1] * 3, cells[ci].Body[0] * 3, add_rows, M);
            long long rank = compute_basis(far.transpose(), epi, S_ind[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
            //std::cout<<"Rank "<<rank<<std::endl;
            /*for (int c = 0; c < far.cols(); ++c) {
              double col_norm = far.col(c).norm();
              long long count = 0;
              for (int r = 0; r < far.rows(); ++r)
                if (std::abs(far(r,c)) >= threshold * col_norm)
                  count++;
              std::cout<<"Col "<< c <<": " << count<<", Density: "<< ((double)count)/(far.rows())<<std::endl;
            }*/
            DimsLr[i + ibegin] = rank;
          }
        }
      }
    }

    //comm.dataSizesToNeighborOffsets(Ssizes.data());
    //comm.neighbor_bcast(S[0], Ssizes.data());

    for (long long i = 0; i < nodes; i++) {
      // Generate far field for the upper levels
      if (localChildOffsets[i+1] <= localChildOffsets[i]) {
        continue;
      }

      long long M = Dims[i + ibegin];
      std::vector<long long> far_field(Dims);
      if (1. <= epi) {
        // HSS basis
        far_field[i] = 0;
      } else {
        // H2 basis
        for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
          far_field[ACols[ij]] = 0;
        }
      }
      auto far_cols = std::reduce(far_field.begin(), far_field.end());
      // only compute the far field if it exists
      if (far_cols) {
        std::vector<long long> FS_ind(far_cols);
        long long start = 0;
        for (long long ij = 0; ij < nodes; ij++) {
          if (far_field[ij]) {
            std::copy(S_ind[ij + ibegin], S_ind[ij + ibegin] + far_field[ij], &FS_ind[start]);
            start += far_field[ij];
          }
        }
        Eigen::MatrixXcd F(M, far_cols);
        gen_matrix(mat, M, far_cols, S_ind[i + ibegin], FS_ind.data(), F);
        long long rank = compute_basis(F.transpose(), epi, S_ind[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
        //std::cout<<"Rank "<<rank<<std::endl;
        DimsLr[i + ibegin] = rank;
      }
    }

    comm.dataSizesToNeighborOffsets(Qsizes.data());
    comm.neighbor_bcast(DimsLr.data(), neighbor_ones.data());
    comm.neighbor_bcast(S_ind[0], Ssizes.data());
    comm.neighbor_bcast(Q[0], Qsizes.data());
    comm.neighbor_bcast(R[0], Qsizes.data());
  }

  if (std::reduce(DimsLr.begin(), DimsLr.end())) {
    std::vector<long long> Csizes(CRows[nodes]);
    for (long long i = 0; i < nodes; i++)
      std::transform(CCols.begin() + CRows[i], CCols.begin() + CRows[i + 1], Csizes.begin() + CRows[i],
        [&](long long col) { return DimsLr[i + ibegin] * DimsLr[col]; });
    C.alloc(CRows[nodes], Csizes.data());

    std::vector<long long> Usizes(nodes);
    std::transform(&Dims[ibegin], &Dims[ibegin + nodes], &DimsLr[ibegin], Usizes.begin(), std::multiplies<long long>());
    U.alloc(nodes, Usizes.data());
    Z.alloc(xlen, DimsLr.data());
    W.alloc(xlen, DimsLr.data());

    //std::cout<<"Part2"<<std::endl;
    for (long long i = 0; i < nodes; i++) {
      //std::cout<<"Node "<<i<<std::endl;
      long long y = i + ibegin;
      long long M = DimsLr[y];
      //std::cout<<"M "<<M<<std::endl;
      Matrix_t Ry(R[y], M, M, Stride_t(Dims[y], 1));
      Eigen::Map<Eigen::MatrixXcd>(U[i], Dims[y], M) = Eigen::Map<Eigen::MatrixXcd>(Q[y], Dims[y], M);

      for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
        long long x = CCols[ij];
        //std::cout<<"x "<<x<<std::endl;
        long long N = DimsLr[CCols[ij]];
        //std::cout<<"N "<<N<<std::endl;
        Matrix_t Rx(R[x], N, N, Stride_t(Dims[x], 1));

        Eigen::Map<Eigen::MatrixXcd> Cyx(C[ij], M, N);
        if (1. <= epi) {
          Eigen::MatrixXcd Ayx(M, N);
          //gen_matrix(eval, M, N, S[y], S[x], Ayx.data());
          gen_matrix(mat, M, N, S_ind[y], S_ind[x], Ayx);
          Cyx.noalias() = Ry.triangularView<Eigen::Upper>() * Ayx * Rx.transpose().triangularView<Eigen::Lower>();
        }
        else
          gen_matrix(mat, M, N, S_ind[y], S_ind[x], Cyx);
          //gen_matrix(eval, M, N, S[y], S[x], Cyx.data());
      }
    }
  }

  std::cout <<"S_ind: "<<S_ind.size()<<std::endl;
  for (long long i = 0; i < S_ind.size(); ++i)
    std::cout<<*(S_ind[0] + i)<<", ";
  std::cout<<std::endl;
  NbXoffsets.insert(NbXoffsets.begin(), Dims.begin(), Dims.end());
  NbXoffsets.erase(NbXoffsets.begin() + comm.dataSizesToNeighborOffsets(NbXoffsets.data()), NbXoffsets.end());
  NbZoffsets.insert(NbZoffsets.begin(), DimsLr.begin(), DimsLr.end());
  NbZoffsets.erase(NbZoffsets.begin() + comm.dataSizesToNeighborOffsets(NbZoffsets.data()), NbZoffsets.end());
  //*/
}

void H2Matrix::construct_sparse(const Eigen::Ref<const Eigen::MatrixXcd> &mat, double epi, const Cell cells[], const CSR& Near, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm) {
  // number of cells on this level
  long long xlen = comm.lenNeighbors();
  // index of the first cell for this porcess on this level
  long long ibegin = comm.oLocal();
  // number of cells for this process on this level
  long long nodes = comm.lenLocal();
  // index of the first cell for this process in the global cell array
  long long ybegin = comm.oGlobal();

  // dimensions for each cell on this level
  Dims.resize(xlen, 0);
  // LR dimensions for each cell on this level
  DimsLr.resize(xlen, 0);
  // stide for what?
  UpperStride.resize(nodes, 0);

  // get the Nearfield indices CSR
  ARows.insert(ARows.begin(), comm.ARowOffsets.begin(), comm.ARowOffsets.end());
  ACols.insert(ACols.begin(), comm.AColumns.begin(), comm.AColumns.end());
  // get the far-field indices CSR
  CRows.insert(CRows.begin(), comm.CRowOffsets.begin(), comm.CRowOffsets.end());
  CCols.insert(CCols.begin(), comm.CColumns.begin(), comm.CColumns.end());
  // ?
  NA.resize(ARows[nodes], -1);

  // get the number of local children
  long long localChildLen = cells[ybegin + nodes - 1].Child[1] - cells[ybegin].Child[0];
  std::vector<long long> localChildOffsets(nodes + 1, -1);
  
  if (0 < localChildLen) {
    long long lowerBegin = lowerComm.oLocal() + comm.LowerX;
    long long localChildIndex = lowerBegin - cells[ybegin].Child[0];
    std::transform(&cells[ybegin], &cells[ybegin + nodes], localChildOffsets.begin() + 1, [=](const Cell& c) { return localChildIndex + c.Child[1]; });
    localChildOffsets[0] = lowerBegin;

    std::vector<long long> ranks_offsets(localChildLen + 1);
    std::inclusive_scan(lowerA.DimsLr.begin() + localChildOffsets[0], lowerA.DimsLr.begin() + localChildOffsets[nodes], ranks_offsets.begin() + 1);
    ranks_offsets[0] = 0;

    std::transform(localChildOffsets.begin(), localChildOffsets.begin() + nodes, localChildOffsets.begin() + 1, &Dims[ibegin],
      [&](long long start, long long end) { return ranks_offsets[end - lowerBegin] - ranks_offsets[start - lowerBegin]; });

    lenX = ranks_offsets.back();
    LowerZ = std::reduce(lowerA.DimsLr.begin(), lowerA.DimsLr.begin() + lowerBegin, 0ll);
  }
  else {
    // only for leaf level
    // Dims stores the number of particels for each cell (multiplied by 3)
    std::transform(&cells[ybegin], &cells[ybegin + nodes], &Dims[ibegin], [](const Cell& c) { return (c.Body[1] - c.Body[0]) * 3; });
    // total number of particles on this process
    lenX = std::reduce(&Dims[ibegin], &Dims[ibegin + nodes]);
    LowerZ = 0;
  }

  std::vector<long long> neighbor_ones(xlen, 1ll);
  comm.dataSizesToNeighborOffsets(neighbor_ones.data());
  comm.neighbor_bcast(Dims.data(), neighbor_ones.data());
  X.alloc(xlen, Dims.data());
  Y.alloc(xlen, Dims.data());

  std::vector<long long> Qsizes(xlen, 0);
  // Qs and Rs are square matrices
  std::transform(Dims.begin(), Dims.end(), Qsizes.begin(), [](const long long d) { return d * d; });
  Q.alloc(xlen, Qsizes.data());
  R.alloc(xlen, Qsizes.data());

  // S stores the indices
  std::vector<long long> Ssizes(xlen);
  std::transform(Dims.begin(), Dims.end(), Ssizes.begin(), [](const long long d) { return d; });
  S_ind.alloc(xlen, Ssizes.data());

  // near field blocks (i.e. dense matrices on the leaf level), not necessarily square
  std::vector<long long> Asizes(ARows[nodes]);
  for (long long i = 0; i < nodes; i++)
    std::transform(ACols.begin() + ARows[i], ACols.begin() + ARows[i + 1], Asizes.begin() + ARows[i],
      [&](long long col) { return Dims[i + ibegin] * Dims[col]; });
  A.alloc(ARows[nodes], Asizes.data());

  typedef Eigen::Stride<Eigen::Dynamic, 1> Stride_t;
  typedef Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Stride_t> Matrix_t; 

  // skip blocks that contain no points (because they are split further)
  if (std::reduce(Dims.begin(), Dims.end())) {
    // index of the first cell for this process on this level
    long long pbegin = lowerComm.oLocal();
    // number of cells for this process/level
    long long pend = pbegin + lowerComm.lenLocal();

    // loop over all nodes
    for (long long i = 0; i < nodes; i++) {
      std::cout<<"Node "<<i<<std::endl;
      // number of rows in that cell
      long long M = Dims[i + ibegin];
      //std::cout<<"Rows "<<M<<std::endl;
      long long childi = localChildOffsets[i];
      long long cendi = localChildOffsets[i + 1];
      // get the corresponding Q matrix (as a reference)
      Eigen::Map<Eigen::MatrixXcd> Qi(Q[i + ibegin], M, M);

      // for all children (i.e. only on the intermediate levels)
      for (long long y = childi; y < cendi; y++) {
        long long offset_y = std::reduce(&lowerA.DimsLr[childi], &lowerA.DimsLr[y]);
        long long ny = lowerA.DimsLr[y];
        std::copy(lowerA.S_ind[y], lowerA.S_ind[y] + (ny), &(S_ind[i + ibegin])[offset_y]);

        // todo check the lr dimensions
        Matrix_t Ry(lowerA.R[y], ny, ny, Stride_t(lowerA.Dims[y], 1));
        Qi.block(offset_y, offset_y, ny, ny) = Ry;

        if (pbegin <= y && y < pend && 0 < M) {
          long long py = y - pbegin;
          lowerA.UpperStride[py] = M;

          for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
            long long j_global = Near.ColIndex[ij + Near.RowIndex[ybegin]];
            long long childj = lowerComm.iLocal(cells[j_global].Child[0]);
            long long cendj = (0 <= childj) ? (childj + cells[j_global].Child[1] - cells[j_global].Child[0]) : -1;

            for (long long x = childj; x < cendj; x++) {
              long long offset_x = std::reduce(&lowerA.DimsLr[childj], &lowerA.DimsLr[x]);
              long long nx = lowerA.DimsLr[x];
              long long lowN = lookupIJ(lowerA.ARows, lowerA.ACols, py, x);
              long long lowC = lookupIJ(lowerA.CRows, lowerA.CCols, py, x);
              std::complex<double>* dp = A[ij] + offset_y + offset_x * M;
              if (0 <= lowN)
                lowerA.NA[lowN] = std::distance(A[0], dp);
              else if (0 <= lowC)
                Matrix_t(dp, ny, nx, Stride_t(M, 1)) = Eigen::Map<Eigen::MatrixXcd>(lowerA.C[lowC], ny, nx);
            }
          }
        }
      }
       
      // leaf level (i.e. no children)
      if (cendi <= childi) {
        long long ci = i + ybegin;
        std::iota(S_ind[i + ibegin], S_ind[i + ibegin + 1], cells[ci].Body[0] * 3);
        Qi = Eigen::MatrixXcd::Identity(M, M);

        long long far_rows = mat.rows();
        // generate the near field aka dense matrices in A
        for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
          //std::cout<<"J "<<ij<<std::endl;
          long long N = Dims[ACols[ij]];
          far_rows -= N;
          long long cj = Near.ColIndex[ij + Near.RowIndex[ybegin]];
          Eigen::Map<Eigen::MatrixXcd> A_ij(A[ij], M, N);
          A_ij = mat.block(cells[ci].Body[0] * 3, cells[cj].Body[0] * 3, M, N);
        }
        double threshold = 1e-2;
        if (1. <= epi) {
          // build an HSS basis
          far_rows = mat.rows() - M;
          Eigen::MatrixXcd far(far_rows, M);
          long long diag = Near.ColIndex[ARows[i] + Near.RowIndex[ybegin]];
          long long top = cells[ci].Body[0] * 3;
          long long bottom = cells[ci].Body[1] * 3;
          //std::cout<<"Top Rows "<<0<<" "<<cells[ci].Body[0] * 3<<" | "<<current_rows<<" "<<M<<std::endl;
          far.topRows(top) = mat.block(0, top, top, M);
          far.bottomRows(mat.rows() - bottom) = mat.block(bottom, top, mat.rows() - bottom, M);
          std::vector<Eigen::Triplet<std::complex<double>>> nonzero_list;
          for (int c = 0; c < far.cols(); ++c) {
            double col_norm = far.col(c).norm();
            for (int r = 0; r < far.rows(); ++r)
              if (std::abs(far(r, c)) >= threshold * col_norm)
                nonzero_list.push_back(Eigen::Triplet<std::complex<double>>(r, c, far(r, c)));
          }
          Eigen::SparseMatrix<std::complex<double>> far_sparse(far.rows(), far.cols());
          far_sparse.setFromTriplets(nonzero_list.begin(), nonzero_list.end());
          long long rank = compute_basis_rid(far_sparse, epi, S_ind[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
          std::cout<<"Rank "<<rank<<std::endl;
          /*for (int c = 0; c < far.cols(); ++c) {
            double col_norm = far.col(c).norm();
            long long count = 0;
            for (int r = 0; r < far.rows(); ++r)
              if (std::abs(far(r,c)) >= threshold * col_norm)
                count++;
            std::cout<<"Col "<< c <<": " << count<<", Density: "<< ((double)count)/(far.rows())<<std::endl;
          }*/
          DimsLr[i + ibegin] = rank;
        } else {
          // build an H2 basis
          // generate the far field only if it exists
          if (far_rows > 0) {
            Eigen::MatrixXcd far(far_rows, M);
            long long current_near = Near.ColIndex[ARows[i] + Near.RowIndex[ybegin]];
            //std::cout<<"Current Near "<<current_near<<std::endl;
            //std::cout<<ARows[i]<<" "<<ARows[i+1]<<std::endl;
            long long current_rows = cells[current_near].Body[0] * 3;
            //std::cout<<"Top Rows "<<0<<" "<<cells[ci].Body[0] * 3<<" | "<<current_rows<<" "<<M<<std::endl;
            far.topRows(current_rows) = mat.block(0, cells[ci].Body[0] * 3, current_rows, M);
            for (long long ij = ARows[i]; ij < ARows[i + 1] - 1; ij++) {
              current_near = Near.ColIndex[ij + Near.RowIndex[ybegin]];
              //std::cout<<"Current Near "<<current_near<<std::endl;
              long long next_near = Near.ColIndex[ij + 1 + Near.RowIndex[ybegin]];
              //std::cout<<"Next Near "<<next_near<<std::endl;
              long long add_rows = cells[next_near].Body[0] * 3 - cells[current_near].Body[1] * 3;
              //std::cout<<"Middle Rows "<<current_rows<<" "<<add_rows<<std::endl;
              //std::cout<<cells[current_near].Body[1] * 3<<" "<<cells[ci].Body[0] * 3<<" | "<<add_rows<<" "<<M<<std::endl;
              far.middleRows(current_rows, add_rows) = mat.block(cells[current_near].Body[1] * 3, cells[ci].Body[0] * 3, add_rows, M);
              current_rows += add_rows;
            }
            current_near = Near.ColIndex[ARows[i + 1] - 1 + Near.RowIndex[ybegin]];
            long long add_rows = mat.rows() - cells[current_near].Body[1] * 3;
            //std::cout<<"Bottom Rows "<<cells[current_near].Body[1] * 3<<" "<<cells[ci].Body[0] * 3<<" | "<<add_rows<<" "<<M<<std::endl;
            //std::cout<<"Current Near "<<current_near<<std::endl;
            far.bottomRows(add_rows) = mat.block(cells[current_near].Body[1] * 3, cells[ci].Body[0] * 3, add_rows, M);
            std::vector<Eigen::Triplet<std::complex<double>>> nonzero_list;
            for (int c = 0; c < far.cols(); ++c) {
              double col_norm = far.col(c).norm();
              for (int r = 0; r < far.rows(); ++r)
                if (std::abs(far(r, c)) >= threshold * col_norm)
                  nonzero_list.push_back(Eigen::Triplet<std::complex<double>>(r, c, far(r, c)));
            }
            Eigen::SparseMatrix<std::complex<double>> far_sparse(far.rows(), far.cols());
            far_sparse.setFromTriplets(nonzero_list.begin(), nonzero_list.end());
            long long rank = compute_basis_rid(far_sparse, epi, S_ind[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
            std::cout<<"Rank "<<rank<<std::endl;
            /*for (int c = 0; c < far.cols(); ++c) {
              double col_norm = far.col(c).norm();
              long long count = 0;
              for (int r = 0; r < far.rows(); ++r)
                if (std::abs(far(r,c)) >= threshold * col_norm)
                  count++;
              std::cout<<"Col "<< c <<": " << count<<", Density: "<< ((double)count)/(far.rows())<<std::endl;
            }*/
            DimsLr[i + ibegin] = rank;
          }
        }
      }
    }

    //comm.dataSizesToNeighborOffsets(Ssizes.data());
    //comm.neighbor_bcast(S[0], Ssizes.data());

    for (long long i = 0; i < nodes; i++) {
      // Generate far field for the upper levels
      if (localChildOffsets[i+1] <= localChildOffsets[i]) {
        continue;
      }

      long long M = Dims[i + ibegin];
      std::vector<long long> far_field(Dims);
      if (1. <= epi) {
        // HSS basis
        far_field[i] = 0;
      } else {
        // H2 basis
        for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
          far_field[ACols[ij]] = 0;
        }
      }
      auto far_rows = std::reduce(far_field.begin(), far_field.end());
      // only compute the far field if it exists
      if (far_rows) {
        std::vector<long long> FS_ind(far_rows);
        long long start = 0;
        for (long long ij = 0; ij < nodes; ij++) {
          if (far_field[ij]) {
            std::copy(S_ind[ij + ibegin], S_ind[ij + ibegin] + far_field[ij], &FS_ind[start]);
            start += far_field[ij];
          }
        }
        Eigen::MatrixXcd F(far_rows, M);
        gen_matrix(mat, far_rows, M, FS_ind.data(), S_ind[i + ibegin], F);
        long long rank = compute_basis(F, epi, S_ind[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
        //std::cout<<"Rank "<<rank<<std::endl;
        DimsLr[i + ibegin] = rank;
      }
    }

    comm.dataSizesToNeighborOffsets(Qsizes.data());
    comm.neighbor_bcast(DimsLr.data(), neighbor_ones.data());
    comm.neighbor_bcast(S_ind[0], Ssizes.data());
    comm.neighbor_bcast(Q[0], Qsizes.data());
    comm.neighbor_bcast(R[0], Qsizes.data());
  }

  if (std::reduce(DimsLr.begin(), DimsLr.end())) {
    std::vector<long long> Csizes(CRows[nodes]);
    for (long long i = 0; i < nodes; i++)
      std::transform(CCols.begin() + CRows[i], CCols.begin() + CRows[i + 1], Csizes.begin() + CRows[i],
        [&](long long col) { return DimsLr[i + ibegin] * DimsLr[col]; });
    C.alloc(CRows[nodes], Csizes.data());

    std::vector<long long> Usizes(nodes);
    std::transform(&Dims[ibegin], &Dims[ibegin + nodes], &DimsLr[ibegin], Usizes.begin(), std::multiplies<long long>());
    U.alloc(nodes, Usizes.data());
    Z.alloc(xlen, DimsLr.data());
    W.alloc(xlen, DimsLr.data());

    //std::cout<<"Part2"<<std::endl;
    for (long long i = 0; i < nodes; i++) {
      //std::cout<<"Node "<<i<<std::endl;
      long long y = i + ibegin;
      long long M = DimsLr[y];
      //std::cout<<"M "<<M<<std::endl;
      Matrix_t Ry(R[y], M, M, Stride_t(Dims[y], 1));
      Eigen::Map<Eigen::MatrixXcd>(U[i], Dims[y], M) = Eigen::Map<Eigen::MatrixXcd>(Q[y], Dims[y], M);

      for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
        long long x = CCols[ij];
        //std::cout<<"x "<<x<<std::endl;
        long long N = DimsLr[CCols[ij]];
        //std::cout<<"N "<<N<<std::endl;
        Matrix_t Rx(R[x], N, N, Stride_t(Dims[x], 1));

        Eigen::Map<Eigen::MatrixXcd> Cyx(C[ij], M, N);
        if (1. <= epi) {
          Eigen::MatrixXcd Ayx(M, N);
          //gen_matrix(eval, M, N, S[y], S[x], Ayx.data());
          gen_matrix(mat, M, N, S_ind[y], S_ind[x], Ayx);
          Cyx.noalias() = Ry.triangularView<Eigen::Upper>() * Ayx * Rx.transpose().triangularView<Eigen::Lower>();
        }
        else
          gen_matrix(mat, M, N, S_ind[y], S_ind[x], Cyx);
          //gen_matrix(eval, M, N, S[y], S[x], Cyx.data());
      }
    }
  }

  NbXoffsets.insert(NbXoffsets.begin(), Dims.begin(), Dims.end());
  NbXoffsets.erase(NbXoffsets.begin() + comm.dataSizesToNeighborOffsets(NbXoffsets.data()), NbXoffsets.end());
  NbZoffsets.insert(NbZoffsets.begin(), DimsLr.begin(), DimsLr.end());
  NbZoffsets.erase(NbZoffsets.begin() + comm.dataSizesToNeighborOffsets(NbZoffsets.data()), NbZoffsets.end());
  //*/
}

void H2Matrix::construct(const Eigen::Ref<const Eigen::MatrixXcd> &mat, double epi, const Cell cells[], const CSR& Near, const HiDR& hidr, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm) {
  // number of cells on this level
  long long xlen = comm.lenNeighbors();
  // index of the first cell for this porcess on this level
  long long ibegin = comm.oLocal();
  // number of cells for this process on this level
  long long nodes = comm.lenLocal();
  // index of the first cell for this process in the global cell array
  long long ybegin = comm.oGlobal();

  // dimensions for each cell on this level
  Dims.resize(xlen, 0);
  // LR dimensions for each cell on this level
  DimsLr.resize(xlen, 0);
  // stide for what?
  UpperStride.resize(nodes, 0);

  // get the Nearfield indices CSR
  ARows.insert(ARows.begin(), comm.ARowOffsets.begin(), comm.ARowOffsets.end());
  ACols.insert(ACols.begin(), comm.AColumns.begin(), comm.AColumns.end());
  // get the far-field indices CSR
  CRows.insert(CRows.begin(), comm.CRowOffsets.begin(), comm.CRowOffsets.end());
  CCols.insert(CCols.begin(), comm.CColumns.begin(), comm.CColumns.end());
  // ?
  NA.resize(ARows[nodes], -1);

  // get the number of local children
  long long localChildLen = cells[ybegin + nodes - 1].Child[1] - cells[ybegin].Child[0];
  std::vector<long long> localChildOffsets(nodes + 1, -1);
  
  if (0 < localChildLen) {
    long long lowerBegin = lowerComm.oLocal() + comm.LowerX;
    long long localChildIndex = lowerBegin - cells[ybegin].Child[0];
    std::transform(&cells[ybegin], &cells[ybegin + nodes], localChildOffsets.begin() + 1, [=](const Cell& c) { return localChildIndex + c.Child[1]; });
    localChildOffsets[0] = lowerBegin;

    std::vector<long long> ranks_offsets(localChildLen + 1);
    std::inclusive_scan(lowerA.DimsLr.begin() + localChildOffsets[0], lowerA.DimsLr.begin() + localChildOffsets[nodes], ranks_offsets.begin() + 1);
    ranks_offsets[0] = 0;

    std::transform(localChildOffsets.begin(), localChildOffsets.begin() + nodes, localChildOffsets.begin() + 1, &Dims[ibegin],
      [&](long long start, long long end) { return ranks_offsets[end - lowerBegin] - ranks_offsets[start - lowerBegin]; });

    lenX = ranks_offsets.back();
    LowerZ = std::reduce(lowerA.DimsLr.begin(), lowerA.DimsLr.begin() + lowerBegin, 0ll);
  }
  else {
    // only for leaf level
    // Dims stores the number of particels for each cell (multiplied by 3)
    std::transform(&cells[ybegin], &cells[ybegin + nodes], &Dims[ibegin], [](const Cell& c) { return (c.Body[1] - c.Body[0]) * 3; });
    // total number of particles on this process
    lenX = std::reduce(&Dims[ibegin], &Dims[ibegin + nodes]);
    LowerZ = 0;
  }

  std::vector<long long> neighbor_ones(xlen, 1ll);
  comm.dataSizesToNeighborOffsets(neighbor_ones.data());
  comm.neighbor_bcast(Dims.data(), neighbor_ones.data());
  X.alloc(xlen, Dims.data());
  Y.alloc(xlen, Dims.data());

  std::vector<long long> Qsizes(xlen, 0);
  // Qs and Rs are square matrices
  std::transform(Dims.begin(), Dims.end(), Qsizes.begin(), [](const long long d) { return d * d; });
  Q.alloc(xlen, Qsizes.data());
  R.alloc(xlen, Qsizes.data());

  // S stores the indices
  std::vector<long long> Ssizes(xlen);
  std::transform(Dims.begin(), Dims.end(), Ssizes.begin(), [](const long long d) { return d; });
  S_ind.alloc(xlen, Ssizes.data());

  // near field blocks (i.e. dense matrices on the leaf level), not necessarily square
  std::vector<long long> Asizes(ARows[nodes]);
  for (long long i = 0; i < nodes; i++)
    std::transform(ACols.begin() + ARows[i], ACols.begin() + ARows[i + 1], Asizes.begin() + ARows[i],
      [&](long long col) { return Dims[i + ibegin] * Dims[col]; });
  A.alloc(ARows[nodes], Asizes.data());

  typedef Eigen::Stride<Eigen::Dynamic, 1> Stride_t;
  typedef Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Stride_t> Matrix_t; 

  // skip blocks that contain no points (because they are split further)
  if (std::reduce(Dims.begin(), Dims.end())) {
    // index of the first cell for this process on this level
    long long pbegin = lowerComm.oLocal();
    // number of cells for this process/level
    long long pend = pbegin + lowerComm.lenLocal();

    // loop over all nodes
    for (long long i = 0; i < nodes; i++) {
      std::cout<<"Node "<<i<<std::endl;
      // number of rows in that cell
      long long M = Dims[i + ibegin];
      //std::cout<<"Rows "<<M<<std::endl;
      long long childi = localChildOffsets[i];
      long long cendi = localChildOffsets[i + 1];
      // get the corresponding Q matrix (as a reference)
      Eigen::Map<Eigen::MatrixXcd> Qi(Q[i + ibegin], M, M);

      // for all children (i.e. only on the intermediate levels)
      for (long long y = childi; y < cendi; y++) {
        long long offset_y = std::reduce(&lowerA.DimsLr[childi], &lowerA.DimsLr[y]);
        long long ny = lowerA.DimsLr[y];
        std::copy(lowerA.S_ind[y], lowerA.S_ind[y] + (ny), &(S_ind[i + ibegin])[offset_y]);

        // todo check the lr dimensions
        Matrix_t Ry(lowerA.R[y], ny, ny, Stride_t(lowerA.Dims[y], 1));
        Qi.block(offset_y, offset_y, ny, ny) = Ry;

        if (pbegin <= y && y < pend && 0 < M) {
          long long py = y - pbegin;
          lowerA.UpperStride[py] = M;

          for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
            long long j_global = Near.ColIndex[ij + Near.RowIndex[ybegin]];
            long long childj = lowerComm.iLocal(cells[j_global].Child[0]);
            long long cendj = (0 <= childj) ? (childj + cells[j_global].Child[1] - cells[j_global].Child[0]) : -1;

            for (long long x = childj; x < cendj; x++) {
              long long offset_x = std::reduce(&lowerA.DimsLr[childj], &lowerA.DimsLr[x]);
              long long nx = lowerA.DimsLr[x];
              long long lowN = lookupIJ(lowerA.ARows, lowerA.ACols, py, x);
              long long lowC = lookupIJ(lowerA.CRows, lowerA.CCols, py, x);
              std::complex<double>* dp = A[ij] + offset_y + offset_x * M;
              if (0 <= lowN)
                lowerA.NA[lowN] = std::distance(A[0], dp);
              else if (0 <= lowC)
                Matrix_t(dp, ny, nx, Stride_t(M, 1)) = Eigen::Map<Eigen::MatrixXcd>(lowerA.C[lowC], ny, nx);
            }
          }
        }
      }
       
      // leaf level (i.e. no children)
      if (cendi <= childi) {
        long long ci = i + ybegin;
        std::iota(S_ind[i + ibegin], S_ind[i + ibegin + 1], cells[ci].Body[0] * 3);
        Qi = Eigen::MatrixXcd::Identity(M, M);

        //long long far_rows = mat.rows();
        // generate the near field aka dense matrices in A
        for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
          //std::cout<<"J "<<ij<<std::endl;
          long long N = Dims[ACols[ij]];
          //far_rows -= N;
          long long cj = Near.ColIndex[ij + Near.RowIndex[ybegin]];
          Eigen::Map<Eigen::MatrixXcd> A_ij(A[ij], M, N);
          A_ij = mat.block(cells[ci].Body[0] * 3, cells[cj].Body[0] * 3, M, N);
        }
        long long far_rows = hidr.fbodies_size_at_i(i);
        if (1. <= epi) {
          // build an HSS basis
          //far_rows = mat.rows() - M;
          // std::cout<<"Far frows "<< far_rows<<" vs "<<hidr.fbodies_size_at_i(i) * 3 <<std::endl;
          //far_rows = hidr.fbodies_size_at_i(i);
          Eigen::MatrixXcd far(far_rows * 3, M);
          /*long long diag = Near.ColIndex[ARows[i] + Near.RowIndex[ybegin]];
          long long top = cells[ci].Body[0] * 3;
          long long bottom = cells[ci].Body[1] * 3;
          //std::cout<<"Top Rows "<<0<<" "<<cells[ci].Body[0] * 3<<" | "<<current_rows<<" "<<M<<std::endl;
          far.topRows(top) = mat.block(0, top, top, M);
          far.bottomRows(mat.rows() - bottom) = mat.block(bottom, top, mat.rows() - bottom, M);*/
          gen_matrix_hidr(mat, far_rows, M, hidr.fbodies_at_i(i), S_ind[i + ibegin], far);
          long long rank = compute_basis(far, epi, S_ind[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
          std::cout<<"Rank "<<rank<<std::endl;
          DimsLr[i + ibegin] = rank;
        } else {
          // build an H2 basis
          // generate the far field only if it exists
          // todo this is superficial no?
          //far_rows = hidr.fbodies_size_at_i(i);
          if (far_rows > 0) {
            Eigen::MatrixXcd far(far_rows * 3, M);
            //std::cout<<"Far frows "<< far_rows<<" vs "<<hidr.fbodies_size_at_i(i) * 3 <<std::endl;
            /*long long current_near = Near.ColIndex[ARows[i] + Near.RowIndex[ybegin]];
            //std::cout<<"Current Near "<<current_near<<std::endl;
            //std::cout<<ARows[i]<<" "<<ARows[i+1]<<std::endl;
            long long current_rows = cells[current_near].Body[0] * 3;
            //std::cout<<"Top Rows "<<0<<" "<<cells[ci].Body[0] * 3<<" | "<<current_rows<<" "<<M<<std::endl;
            far.topRows(current_rows) = mat.block(0, cells[ci].Body[0] * 3, current_rows, M);
            for (long long ij = ARows[i]; ij < ARows[i + 1] - 1; ij++) {
              current_near = Near.ColIndex[ij + Near.RowIndex[ybegin]];
              //std::cout<<"Current Near "<<current_near<<std::endl;
              long long next_near = Near.ColIndex[ij + 1 + Near.RowIndex[ybegin]];
              //std::cout<<"Next Near "<<next_near<<std::endl;
              long long add_rows = cells[next_near].Body[0] * 3 - cells[current_near].Body[1] * 3;
              //std::cout<<"Middle Rows "<<current_rows<<" "<<add_rows<<std::endl;
              //std::cout<<cells[current_near].Body[1] * 3<<" "<<cells[ci].Body[0] * 3<<" | "<<add_rows<<" "<<M<<std::endl;
              far.middleRows(current_rows, add_rows) = mat.block(cells[current_near].Body[1] * 3, cells[ci].Body[0] * 3, add_rows, M);
              current_rows += add_rows;
            }
            current_near = Near.ColIndex[ARows[i + 1] - 1 + Near.RowIndex[ybegin]];
            long long add_rows = mat.rows() - cells[current_near].Body[1] * 3;
            //std::cout<<"Bottom Rows "<<cells[current_near].Body[1] * 3<<" "<<cells[ci].Body[0] * 3<<" | "<<add_rows<<" "<<M<<std::endl;
            //std::cout<<"Current Near "<<current_near<<std::endl;
            far.bottomRows(add_rows) = mat.block(cells[current_near].Body[1] * 3, cells[ci].Body[0] * 3, add_rows, M);*/
            gen_matrix_hidr(mat, far_rows, M, hidr.fbodies_at_i(i), S_ind[i + ibegin], far);
            long long rank = compute_basis(far, epi, S_ind[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
            std::cout<<"Rank "<<rank<<std::endl;
            DimsLr[i + ibegin] = rank;
          }
        }
      }
    }

    //comm.dataSizesToNeighborOffsets(Ssizes.data());
    //comm.neighbor_bcast(S[0], Ssizes.data());

    for (long long i = 0; i < nodes; i++) {
      // Generate far field for the upper levels
      if (localChildOffsets[i+1] <= localChildOffsets[i]) {
        continue;
      }

      long long M = Dims[i + ibegin];
      std::vector<long long> far_field(Dims);
      if (1. <= epi) {
        // HSS basis
        far_field[i] = 0;
      } else {
        // H2 basis
        for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
          far_field[ACols[ij]] = 0;
        }
      }
      //auto far_rows = hidr.fbodies_size_at_i(i);
      auto far_rows = std::reduce(far_field.begin(), far_field.end());
      // only compute the far field if it exists
      if (far_rows) {
        std::vector<long long> FS_ind(far_rows);
        long long start = 0;
        for (long long ij = 0; ij < nodes; ij++) {
          if (far_field[ij]) {
            std::copy(S_ind[ij + ibegin], S_ind[ij + ibegin] + far_field[ij], &FS_ind[start]);
            start += far_field[ij];
          }
        }
        Eigen::MatrixXcd F(far_rows, M);
        gen_matrix(mat, far_rows, M, FS_ind.data(), S_ind[i + ibegin], F);
        //gen_matrix(mat, far_rows, M, hidr.fbodies_at_i(i), S_ind[i + ibegin], F);
        long long rank = compute_basis(F, epi, S_ind[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
        //std::cout<<"Rank "<<rank<<std::endl;
        DimsLr[i + ibegin] = rank;
      }
    }

    comm.dataSizesToNeighborOffsets(Qsizes.data());
    comm.neighbor_bcast(DimsLr.data(), neighbor_ones.data());
    comm.neighbor_bcast(S_ind[0], Ssizes.data());
    comm.neighbor_bcast(Q[0], Qsizes.data());
    comm.neighbor_bcast(R[0], Qsizes.data());
  }

  if (std::reduce(DimsLr.begin(), DimsLr.end())) {
    std::vector<long long> Csizes(CRows[nodes]);
    for (long long i = 0; i < nodes; i++)
      std::transform(CCols.begin() + CRows[i], CCols.begin() + CRows[i + 1], Csizes.begin() + CRows[i],
        [&](long long col) { return DimsLr[i + ibegin] * DimsLr[col]; });
    C.alloc(CRows[nodes], Csizes.data());

    std::vector<long long> Usizes(nodes);
    std::transform(&Dims[ibegin], &Dims[ibegin + nodes], &DimsLr[ibegin], Usizes.begin(), std::multiplies<long long>());
    U.alloc(nodes, Usizes.data());
    Z.alloc(xlen, DimsLr.data());
    W.alloc(xlen, DimsLr.data());

    //std::cout<<"Part2"<<std::endl;
    for (long long i = 0; i < nodes; i++) {
      //std::cout<<"Node "<<i<<std::endl;
      long long y = i + ibegin;
      long long M = DimsLr[y];
      //std::cout<<"M "<<M<<std::endl;
      Matrix_t Ry(R[y], M, M, Stride_t(Dims[y], 1));
      Eigen::Map<Eigen::MatrixXcd>(U[i], Dims[y], M) = Eigen::Map<Eigen::MatrixXcd>(Q[y], Dims[y], M);

      for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
        long long x = CCols[ij];
        //std::cout<<"x "<<x<<std::endl;
        long long N = DimsLr[CCols[ij]];
        //std::cout<<"N "<<N<<std::endl;
        Matrix_t Rx(R[x], N, N, Stride_t(Dims[x], 1));

        Eigen::Map<Eigen::MatrixXcd> Cyx(C[ij], M, N);
        if (1. <= epi) {
          Eigen::MatrixXcd Ayx(M, N);
          //gen_matrix(eval, M, N, S[y], S[x], Ayx.data());
          gen_matrix(mat, M, N, S_ind[y], S_ind[x], Ayx);
          Cyx.noalias() = Ry.triangularView<Eigen::Upper>() * Ayx * Rx.transpose().triangularView<Eigen::Lower>();
        }
        else
          gen_matrix(mat, M, N, S_ind[y], S_ind[x], Cyx);
          //gen_matrix(eval, M, N, S[y], S[x], Cyx.data());
      }
    }
  }

  NbXoffsets.insert(NbXoffsets.begin(), Dims.begin(), Dims.end());
  NbXoffsets.erase(NbXoffsets.begin() + comm.dataSizesToNeighborOffsets(NbXoffsets.data()), NbXoffsets.end());
  NbZoffsets.insert(NbZoffsets.begin(), DimsLr.begin(), DimsLr.end());
  NbZoffsets.erase(NbZoffsets.begin() + comm.dataSizesToNeighborOffsets(NbZoffsets.data()), NbZoffsets.end());
  //*/
}

void H2Matrix::constructBLR(const Eigen::Ref<const Eigen::MatrixXcd> &mat, double epi, const Cell cells[], const CSR& Near, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm) {
  // number of cells on this level
  long long xlen = comm.lenNeighbors();
  // index of the first cell for this porcess on this level
  long long ibegin = comm.oLocal();
  // number of cells for this process on this level
  long long nodes = comm.lenLocal();
  // index of the first cell for this process in the global cell array
  long long ybegin = comm.oGlobal();

  // dimensions for each cell on this level
  Dims.resize(xlen, 0);
  // LR dimensions for each cell on this level
  DimsLr.resize(xlen, 0);
  // stide for what?
  UpperStride.resize(nodes, 0);

  // get the Nearfield indices CSR
  ARows.insert(ARows.begin(), comm.ARowOffsets.begin(), comm.ARowOffsets.end());
  ACols.insert(ACols.begin(), comm.AColumns.begin(), comm.AColumns.end());
  // get the far-field indices CSR
  CRows.insert(CRows.begin(), comm.CRowOffsets.begin(), comm.CRowOffsets.end());
  CCols.insert(CCols.begin(), comm.CColumns.begin(), comm.CColumns.end());
  // ?
  NA.resize(ARows[nodes], -1);

  // get the number of local children
  long long localChildLen = cells[ybegin + nodes - 1].Child[1] - cells[ybegin].Child[0];
  std::vector<long long> localChildOffsets(nodes + 1, -1);
  
  if (0 < localChildLen) {
    long long lowerBegin = lowerComm.oLocal() + comm.LowerX;
    long long localChildIndex = lowerBegin - cells[ybegin].Child[0];
    std::transform(&cells[ybegin], &cells[ybegin + nodes], localChildOffsets.begin() + 1, [=](const Cell& c) { return localChildIndex + c.Child[1]; });
    localChildOffsets[0] = lowerBegin;

    std::vector<long long> ranks_offsets(localChildLen + 1);
    std::inclusive_scan(lowerA.DimsLr.begin() + localChildOffsets[0], lowerA.DimsLr.begin() + localChildOffsets[nodes], ranks_offsets.begin() + 1);
    ranks_offsets[0] = 0;

    std::transform(localChildOffsets.begin(), localChildOffsets.begin() + nodes, localChildOffsets.begin() + 1, &Dims[ibegin],
      [&](long long start, long long end) { return ranks_offsets[end - lowerBegin] - ranks_offsets[start - lowerBegin]; });

    lenX = ranks_offsets.back();
    LowerZ = std::reduce(lowerA.DimsLr.begin(), lowerA.DimsLr.begin() + lowerBegin, 0ll);
  }
  else {
    // only for leaf level
    // Dims stores the number of particels for each cell
    std::transform(&cells[ybegin], &cells[ybegin + nodes], &Dims[ibegin], [](const Cell& c) { return c.Body[1] - c.Body[0]; });
    // total number of particles on this process
    lenX = std::reduce(&Dims[ibegin], &Dims[ibegin + nodes]);
    LowerZ = 0;
  }

  std::vector<long long> neighbor_ones(xlen, 1ll);
  comm.dataSizesToNeighborOffsets(neighbor_ones.data());
  comm.neighbor_bcast(Dims.data(), neighbor_ones.data());
  X.alloc(xlen, Dims.data());
  Y.alloc(xlen, Dims.data());

  std::vector<long long> Qsizes(xlen, 0);
  // Qs and Rs are square matrices
  //for (size_t i = 0; i<Dims.size(); ++i) {
  //  std::cout<<Dims[i]<<", ";
  //}
  std::cout<<std::endl;
  std::transform(Dims.begin(), Dims.end(), Qsizes.begin(), [](const long long d) { return d * d; });
  Q.alloc(xlen, Qsizes.data());
  R.alloc(xlen, Qsizes.data());

  // we don't really need S
  std::vector<long long> Ssizes(xlen);
  std::transform(Dims.begin(), Dims.end(), Ssizes.begin(), [](const long long d) { return 3 * d; });
  S.alloc(xlen, Ssizes.data());

  // near field blocks (i.e. dense matrices on the leaf level), not necessarily square
  std::vector<long long> Asizes(ARows[nodes]);
  for (long long i = 0; i < nodes; i++)
    std::transform(ACols.begin() + ARows[i], ACols.begin() + ARows[i + 1], Asizes.begin() + ARows[i],
      [&](long long col) { return Dims[i + ibegin] * Dims[col] * 9; });
  A.alloc(ARows[nodes], Asizes.data());
  std::cout<<"A_size "<<A.size()<<std::endl;

  typedef Eigen::Stride<Eigen::Dynamic, 1> Stride_t;
  //typedef Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Stride_t> Matrix_t; 

  // skip blocks that contain no points (because they are split further)
  if (std::reduce(Dims.begin(), Dims.end())) {
    // index of the first cell for this process on this level
    long long pbegin = lowerComm.oLocal();
    // number of cells for this process/level
   //long long pend = pbegin + lowerComm.lenLocal();

    // loop over all nodes
    for (long long i = 0; i < nodes; i++) {
      //std::cout<<"Node "<<i<<std::endl;
      // number of rows in that cell
      long long M = Dims[i + ibegin];
      //std::cout<<"Rows "<<M<<std::endl;
      long long childi = localChildOffsets[i];
      long long cendi = localChildOffsets[i + 1];
      // get the corresponding Q matrix (as a reference)
      //Eigen::Map<Eigen::MatrixXcd> Qi(Q[i + ibegin], M, M);
      
       
      // leaf level (i.e. no children)
      if (cendi <= childi) {
        long long ci = i + ybegin;
        //std::copy(&bodies[3 * cells[ci].Body[0]], &bodies[3 * cells[ci].Body[1]], S[i + ibegin]);
        //Qi = Eigen::MatrixXcd::Identity(M, M);

        // generate the near field aka dense matrices in A
        for (long long j = 0; j < nodes; j++)  {
          //std::cout<<"J "<<j<<std::endl;
          long long N = Dims[ACols[j]];
          //std::cout<<"Cols "<<N<<std::endl;
          //Eigen::Map<Eigen::MatrixXcd> A_ij(A[j], M * 3, N * 3);
          Eigen::MatrixXcd A_ij(M * 3, N * 3);
          long long cj = j + ybegin;
          //std::cout<<"offset "<<3 * cells[ci].Body[0]<<", "<<3 * cells[cj].Body[0]<<std::endl;
          std::vector<long long> near_rows = {cells[ci].Body[0], cells[ci].Body[1]};
          std::vector<long long> near_cols = {cells[cj].Body[0], cells[cj].Body[1]};
          gen_matrix(mat, near_rows, near_cols, A_ij);
          //A_ij = mat.block(3 * cells[ci].Body[0], 3 * cells[cj].Body[0], M * 3, N * 3);
          Eigen::ColPivHouseholderQR<Eigen::Ref<Eigen::MatrixXcd>> rrqr(A_ij);
          rrqr.setThreshold(epi);
          auto rank = rrqr.rank();
          //std::cout<<"Epi "<<epi<<std::endl;
          std::cout<<rank<<" ";
        }
        std::cout<<std::endl;
      }//std::cout<<std::endl;
    }
  }
}

void H2Matrix::matVecDense(const std::complex<double>* X_in, std::complex<double>* X_out, const ColCommMPI& comm) {
  typedef Eigen::Map<Eigen::VectorXcd> Vector_t;
  typedef Eigen::Map<const Eigen::MatrixXcd> Matrix_t;
  // if we have a dedicated output vector we don't need a barrier
  // before writing

  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  Eigen::Map<const Eigen::VectorXcd> x_in(&X_in[0], n_mat);
  //Eigen::VectorXcd xin2(x_in);
  //Matrix_t A2(Mat[0], n_mat, n_mat);
  //Eigen::VectorXcd xout2 = A2 * xin2;

  // there really should be no need to do this in a loop
  // there is because of the data layout
  long long offset = 0;
  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    Matrix_t A(Mat[i], M, n_mat);
    Vector_t x_out(&X_out[offset], M);

    x_out = A * x_in;
    offset += M;
  }
}

void H2Matrix::matVecUpwardPass(const std::complex<double>* X_in, const ColCommMPI& comm) {
  typedef Eigen::Map<Eigen::VectorXcd> Vector_t;
  typedef Eigen::Map<const Eigen::MatrixXcd> Matrix_t;

  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  std::copy(&X_in[LowerZ], &X_in[LowerZ + lenX], X[ibegin]);

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    long long N = DimsLr[i + ibegin];
    Vector_t x(X[i + ibegin], M);
    if (0 < N) {
      Vector_t z(Z[i + ibegin], N);
      Matrix_t q(Q[i + ibegin], M, N);
      z = q.transpose() * x;
    }
  }

  comm.neighbor_bcast(Z[0], NbZoffsets.data());
}

void H2Matrix::matVecHorizontalandDownwardPass(std::complex<double>* Y_out, const ColCommMPI& comm) {
  typedef Eigen::Map<Eigen::VectorXcd> Vector_t;
  typedef Eigen::Map<const Eigen::MatrixXcd> Matrix_t;

  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    long long K = DimsLr[i + ibegin];
    if (0 < K) {
      Vector_t w(W[i + ibegin], K);
      for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
        long long j = CCols[ij];
        long long N = DimsLr[j];

        Vector_t z(Z[j], N);
        Matrix_t c(C[ij], K, N);
        w.noalias() += c * z;
      }

      Matrix_t q(Q[i + ibegin], M, K);
      Vector_t y(Y[i + ibegin], M);
      y.noalias() = q * w;
    }
  }

  std::copy(Y[ibegin], Y[ibegin + nodes], &Y_out[LowerZ]);
}

void H2Matrix::matVecLeafHorizontalPass(std::complex<double>* X_io, const ColCommMPI& comm) {
  typedef Eigen::Map<Eigen::VectorXcd> Vector_t;
  typedef Eigen::Map<Eigen::MatrixXcd> Matrix_t;

  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  std::copy(&X_io[0], &X_io[lenX], X[ibegin]);
  comm.neighbor_bcast(X[0], NbXoffsets.data());

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    long long K = DimsLr[i + ibegin];
    Vector_t y(Y[i + ibegin], M);
    y.setZero();

    if (0 < K) {
      Vector_t w(W[i + ibegin], K);
      for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
        long long j = CCols[ij];
        long long N = DimsLr[j];

        Vector_t z(Z[j], N);
        Matrix_t c(C[ij], K, N);
        w.noalias() += c * z;
      }

      Matrix_t q(Q[i + ibegin], M, K);
      y.noalias() += q * w;
    }

    for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
      long long j = ACols[ij];
      long long N = Dims[j];

      Vector_t x(X[j], N);
      Matrix_t c(A[ij], M, N);
      y.noalias() += c * x;
    }
  }

  std::copy(Y[ibegin], Y[ibegin + nodes], X_io);
}

void H2Matrix::factorize(const ColCommMPI& comm) {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long xlen = comm.lenNeighbors();
  long long dims_max = *std::max_element(Dims.begin(), Dims.end());
  typedef Eigen::Map<Eigen::MatrixXcd> Matrix_t;

  std::vector<long long> Bsizes(xlen);
  std::fill(Bsizes.begin(), Bsizes.end(), dims_max * dims_max);
  MatrixDataContainer<std::complex<double>> B;
  B.alloc(xlen, Bsizes.data());
  info = 0;

  if (nodes == 1)
    comm.level_merge(A[0], A.size());

  for (long long i = 0; i < nodes; i++) {
    long long diag = lookupIJ(ARows, ACols, i, i + ibegin);
    long long M = Dims[i + ibegin];
    long long Ms = DimsLr[i + ibegin];
    long long Mr = M - Ms;

    Matrix_t Ui(Q[i + ibegin], M, M);
    Matrix_t V(R[i + ibegin], M, M);
    Matrix_t Aii(A[diag], M, M);
    Matrix_t b(B[i + ibegin], dims_max, M);

    b.topRows(M).noalias() = Ui.adjoint() * Aii.transpose();
    Aii.noalias() = Ui.adjoint() * b.topRows(M).transpose();
    V.topRows(Ms) = Ui.leftCols(Ms).adjoint();

    if (0 < Mr) {
      // this is only a check for singularity, not needed for the actual computation
      std::vector<int> ipiv(Mr);
      Eigen::MatrixXcd test = Aii.bottomRightCorner(Mr, Mr);
      auto error = LAPACKE_zgetrf(LAPACK_COL_MAJOR, Mr, Mr, reinterpret_cast<__complex__ double*>(test.data()), Mr, ipiv.data());

      Eigen::PartialPivLU<Eigen::MatrixXcd> fac(Aii.bottomRightCorner(Mr, Mr));
      V.bottomRows(Mr) = fac.solve(Ui.rightCols(Mr).adjoint());
      if (0 < Ms) {
        Aii.bottomLeftCorner(Mr, Ms).noalias() = V.bottomRows(Mr) * b.topRows(Ms).transpose();
        Aii.topLeftCorner(Ms, Ms).noalias() -= Aii.topRightCorner(Ms, Mr) * Aii.bottomLeftCorner(Mr, Ms);
      }
      info += error; //(std::abs(fac.determinant()) <= std::numeric_limits<double>::min());
    }

    for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) 
      if (ij != diag) {
        long long j = ACols[ij];
        long long N = Dims[j];

        Matrix_t Uj(Q[j], N, N);
        Matrix_t Aij(A[ij], M, N);

        b.topRows(N) = Uj.adjoint() * Aij.transpose();
        Aij.noalias() = V * b.topRows(N).transpose();
      }
    
    b.topLeftCorner(Mr, Ms) = Aii.bottomLeftCorner(Mr, Ms);
    b.topRightCorner(Mr, Mr) = V.bottomRows(Mr) * Ui.rightCols(Mr);
  }
  comm.dataSizesToNeighborOffsets(Bsizes.data());
  comm.neighbor_bcast(B[0], Bsizes.data());

  for (long long i = 0; i < nodes; i++) {
    long long diag = lookupIJ(ARows, ACols, i, i + ibegin);
    long long M = Dims[i + ibegin];
    long long Ms = DimsLr[i + ibegin];
    long long Mr = M - Ms;
    Matrix_t Aii(A[diag], M, M);

    for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++)
      if (ij != diag) {
        long long j = ACols[ij];
        long long N = Dims[j];
        long long Ns = DimsLr[j];
        long long Nr = N - Ns;
        
        Matrix_t Aij(A[ij], M, N);
        Matrix_t Bj(B[j], dims_max, N);
        Aij.topLeftCorner(Ms, Ns) -= Aii.topRightCorner(Ms, Mr) * Aij.bottomLeftCorner(Mr, Ns) + Aij.topRightCorner(Ms, Nr) * Bj.topLeftCorner(Nr, Ns);
        Aii.topLeftCorner(Ms, Ms) -= Aij.topRightCorner(Ms, Nr) * Bj.topRightCorner(Nr, Nr) * Aij.topRightCorner(Ms, Nr).transpose();
      }
  }
}

void H2Matrix::factorizeCopyNext(const H2Matrix& lowerA, const ColCommMPI& lowerComm) {
  long long ibegin = lowerComm.oLocal();
  long long nodes = lowerComm.lenLocal();
  typedef Eigen::Map<const Eigen::MatrixXcd> Matrix_t;

  for (long long i = 0; i < nodes; i++)
    for (long long ij = lowerA.ARows[i]; ij < lowerA.ARows[i + 1]; ij++) {
      long long j = lowerA.ACols[ij];
      long long M = lowerA.Dims[i + ibegin];
      long long N = lowerA.Dims[j];
      long long Ms = lowerA.DimsLr[i + ibegin];
      long long Ns = lowerA.DimsLr[j];

      Matrix_t Aij(lowerA.A[ij], M, N);
      if (0 < Ms && 0 < Ns) {
        Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> An(A[0] + lowerA.NA[ij], Ms, Ns, Eigen::Stride<Eigen::Dynamic, 1>(lowerA.UpperStride[i], 1));
        An = Aij.topLeftCorner(Ms, Ns);
      }
    }
}

void H2Matrix::forwardSubstitute(const std::complex<double>* X_in, const ColCommMPI& comm) {
  typedef Eigen::Map<Eigen::VectorXcd> Vector_t;
  typedef Eigen::Map<const Eigen::MatrixXcd> Matrix_t;

  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  std::copy(&X_in[LowerZ], &X_in[LowerZ + lenX], Y[ibegin]);

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];

    if (0 < M) {
      Vector_t x(X[i + ibegin], M);
      Vector_t y(Y[i + ibegin], M);
      Matrix_t q(R[i + ibegin], M, M);
      x.noalias() = q * y;
    }
  }

  comm.neighbor_bcast(X[0], NbXoffsets.data());

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    long long Ms = DimsLr[i + ibegin];

    if (0 < Ms) {
      Vector_t z(Z[i + ibegin], Ms);
      z = Vector_t(X[i + ibegin], Ms);

      for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
        long long j = ACols[ij];
        long long N = Dims[j];
        long long Ns = DimsLr[j];
        long long Nr = N - Ns;

        if (0 < Nr) {
          Vector_t xj(X[j], N);
          Matrix_t Aij(A[ij], M, N);
          z.noalias() -= Aij.topRightCorner(Ms, Nr) * xj.bottomRows(Nr);
        }
      }
    }
  }

  comm.neighbor_bcast(Z[0], NbZoffsets.data());
}

void H2Matrix::backwardSubstitute(std::complex<double>* Y_out, const ColCommMPI& comm) {
  typedef Eigen::Map<Eigen::VectorXcd> Vector_t;
  typedef Eigen::Map<const Eigen::MatrixXcd> Matrix_t;

  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  comm.neighbor_bcast(W[0], NbZoffsets.data());

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    long long Ms = DimsLr[i + ibegin];
    long long Mr = M - Ms;

    Vector_t x(X[i + ibegin], M);
    x.topRows(Ms) = Vector_t(W[i + ibegin], Ms);
      
    if (0 < Mr) {
      for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
        long long j = ACols[ij];
        long long N = Dims[j];
        long long Ns = DimsLr[j];

        if (0 < Ns) {
          Vector_t wj(W[j], Ns);
          Matrix_t Aij(A[ij], M, N);
          x.bottomRows(Mr).noalias() -= Aij.bottomLeftCorner(Mr, Ns) * wj;
        }
      }
    }
  }

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    if (0 < M) {
      Vector_t x(X[i + ibegin], M);
      Vector_t y(Y[i + ibegin], M);
      Matrix_t q(Q[i + ibegin], M, M);
      y.noalias() = q.conjugate() * x;
    }
  }

  std::copy(Y[ibegin], Y[ibegin + nodes], &Y_out[LowerZ]);
}
