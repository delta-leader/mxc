#include <h2matrix.hpp>
#include <h-matrix.hpp>
#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <kernel.hpp>

#include <numeric>
#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

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
      Eigen::PartialPivLU<Eigen::MatrixXcd> fac(Aii.bottomRightCorner(Mr, Mr));
      V.bottomRows(Mr) = fac.solve(Ui.rightCols(Mr).adjoint());
      if (0 < Ms) {
        Aii.bottomLeftCorner(Mr, Ms).noalias() = V.bottomRows(Mr) * b.topRows(Ms).transpose();
        Aii.topLeftCorner(Ms, Ms).noalias() -= Aii.topRightCorner(Ms, Mr) * Aii.bottomLeftCorner(Mr, Ms);
      }
      info += (std::abs(fac.determinant()) <= std::numeric_limits<double>::min());
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
