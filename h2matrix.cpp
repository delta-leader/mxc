#include <h2matrix.hpp>
#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <kernel.hpp>

#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>

WellSeparatedApproximation::WellSeparatedApproximation(const MatrixAccessor& eval, double epi, long long rank, long long lbegin, long long len, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& upper) :
  lbegin(lbegin), lend(lbegin + len), M(len) {
  std::vector<std::vector<double>> Fbodies(len);
  // loop over the cells in the upper level
  for (long long i = upper.lbegin; i < upper.lend; i++)
    // loop over all children
    for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; c++)
      if (lbegin <= c && c < lend)
        M[c - lbegin] = std::vector<double>(upper.M[i - upper.lbegin].begin(), upper.M[i - upper.lbegin].end());

  for (long long y = lbegin; y < lend; y++) {
    for (long long yx = Far.RowIndex[y]; yx < Far.RowIndex[y + 1]; yx++) {
      long long x = Far.ColIndex[yx];
      long long m = cells[y].Body[1] - cells[y].Body[0];
      long long n = cells[x].Body[1] - cells[x].Body[0];
      const double* xbodies = &bodies[3 * cells[x].Body[0]];
      const double* ybodies = &bodies[3 * cells[y].Body[0]];

      long long k = std::min(rank, std::min(m, n));
      std::vector<long long> ipiv(k);
      long long iters = adaptive_cross_approximation(epi, eval, m, n, k, ybodies, xbodies, nullptr, &ipiv[0]);
      ipiv.resize(iters);

      Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic>> Xbodies(xbodies, 3, n);
      Eigen::VectorXd Fbodies(3 * iters);

      Fbodies = Xbodies(Eigen::all, ipiv).reshaped();
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

inline long long lookupIJ(const std::vector<long long>& RowIndex, const std::vector<long long>& ColIndex, long long i, long long j) {
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
  NA = std::vector<std::complex<double>*>(ARows[nodes], nullptr);

  CRows = std::vector<long long>(Far.RowIndex.begin() + ybegin, Far.RowIndex.begin() + ybegin + nodes + 1);
  CCols = std::vector<long long>(Far.ColIndex.begin() + CRows[0], Far.ColIndex.begin() + CRows[nodes]);
  offset = CRows[0];
  std::for_each(CRows.begin(), CRows.end(), [=](long long& i) { i = i - offset; });
  std::for_each(CCols.begin(), CCols.end(), [&](long long& col) { col = comm.iLocal(col); });
  C = std::vector<std::complex<double>*>(CRows[nodes], nullptr);

  if (localChildOffsets.back() == 0)
    std::transform(&cells[ybegin], &cells[ybegin + nodes], &Dims[ibegin], [](const Cell& c) { return c.Body[1] - c.Body[0]; });
  else {
    std::vector<long long>::const_iterator iter = lowerA.DimsLr.begin() + localChildIndex;
    std::transform(localChildOffsets.begin(), localChildOffsets.begin() + nodes, localChildOffsets.begin() + 1, &Dims[ibegin],
      [&](long long start, long long end) { return std::reduce(iter + start, iter + end); });
  }

  comm.neighbor_bcast(Dims.data());
  X = MatrixDataContainer<std::complex<double>>(xlen, Dims.data());
  Y = MatrixDataContainer<std::complex<double>>(xlen, Dims.data());
  NX = std::vector<std::complex<double>*>(xlen, nullptr);
  NY = std::vector<std::complex<double>*>(xlen, nullptr);

  for (long long i = 0; i < xlen; i++) {
    long long ci = comm.iGlobal(i);
    long long child = lowerComm.iLocal(cells[ci].Child[0]);
    long long cend = 0 <= child ? (child + cells[ci].Child[1] - cells[ci].Child[0]) : -1;
    for (long long y = child; y < cend; y++) {
      long long offset_y = std::reduce(&lowerA.DimsLr[child], &lowerA.DimsLr[y]);
      lowerA.NX[y] = X[i] + offset_y;
      lowerA.NY[y] = Y[i] + offset_y;
    }
  }

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
  Ipivots = std::vector<int>(std::reduce(Dims.begin() + ibegin, Dims.begin() + (ibegin + nodes)));

  if (std::reduce(Dims.begin(), Dims.end())) {
    typedef Eigen::Stride<Eigen::Dynamic, 1> Stride_t;
    typedef Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Stride_t> Matrix_t; 
    long long pbegin = lowerComm.oLocal();
    long long pend = pbegin + lowerComm.lenLocal();

    for (long long i = 0; i < nodes; i++) {
      long long M = Dims[i + ibegin];
      long long childi = localChildIndex + localChildOffsets[i];
      long long cendi = localChildIndex + localChildOffsets[i + 1];
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
                lowerA.NA[lowN] = dp;
              if (0 <= lowC) {
                lowerA.C[lowC] = dp;
                Matrix_t Rx(lowerA.R[x], nx, nx, Stride_t(lowerA.Dims[x], 1));
                Matrix_t Cyx(dp, ny, nx, Stride_t(M, 1));
                Eigen::MatrixXcd Ayx(ny, nx);
                gen_matrix(eval, ny, nx, lowerA.S[y], lowerA.S[x], Ayx.data());
                Cyx.noalias() = Ry.triangularView<Eigen::Upper>() * Ayx * Rx.transpose().triangularView<Eigen::Lower>();
              }
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
    comm.neighbor_bcast(S);
    std::vector<std::vector<double>> cbodies(nodes);
    for (long long i = 0; i < nodes; i++) {
      long long fsize = wsa.fbodies_size_at_i(i);
      const double* fbodies = wsa.fbodies_at_i(i);

      if (use_near_bodies) {
        std::vector<long long>::iterator neighbors = ACols.begin() + ARows[i];
        std::vector<long long>::iterator neighbors_end = ACols.begin() + ARows[i + 1];
        long long csize = std::transform_reduce(neighbors, neighbors_end, -Dims[i + ibegin], std::plus<long long>(), [&](long long col) { return Dims[col]; });

        cbodies[i] = std::vector<double>(3 * (fsize + csize));
        long long loc = 0;
        for (long long n = 0; n < (ARows[i + 1] - ARows[i]); n++) {
          long long col = neighbors[n];
          long long len = 3 * Dims[col];
          if (col != (i + ibegin)) {
            std::copy(S[col], S[col] + len, cbodies[i].begin() + loc);
            loc += len;
          }
        }
        std::copy(fbodies, &fbodies[3 * fsize], cbodies[i].begin() + loc);
      }
      else {
        cbodies[i] = std::vector<double>(3 * fsize);
        std::copy(fbodies, &fbodies[3 * fsize], cbodies[i].begin());
      }
    }

    for (long long i = 0; i < nodes; i++) {
      long long fsize = cbodies[i].size() / 3;
      const double* fbodies = cbodies[i].data();
      long long rank = compute_basis(eval, epi, Dims[i + ibegin], fsize, S[i + ibegin], fbodies, Q[i + ibegin], R[i + ibegin]);
      DimsLr[i + ibegin] = rank;
    }

    comm.neighbor_bcast(DimsLr.data());
    comm.neighbor_bcast(S);
    comm.neighbor_bcast(Q);
    comm.neighbor_bcast(R);
  }
}

void H2Matrix::matVecUpwardPass(const ColCommMPI& comm) {
  typedef Eigen::Map<Eigen::VectorXcd> Vector_t;
  typedef Eigen::Map<const Eigen::MatrixXcd> Matrix_t;

  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  comm.level_merge(X[0], X.size());
  comm.neighbor_bcast(X);

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    long long N = DimsLr[i + ibegin];
    if (0 < N) {
      Vector_t x(X[i + ibegin], M);
      Vector_t xo(NX[i + ibegin], N);
      Matrix_t q(Q[i + ibegin], M, N);
      xo.noalias() = q.transpose() * x;
    }
  }
}

void H2Matrix::matVecHorizontalandDownwardPass(const ColCommMPI& comm) {
  typedef Eigen::Map<Eigen::VectorXcd> Vector_t;
  typedef Eigen::Stride<Eigen::Dynamic, 1> Stride_t;
  typedef Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Stride_t> Matrix_t;

  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    long long K = DimsLr[i + ibegin];
    if (0 < K) {
      Vector_t y(Y[i + ibegin], M);
      Vector_t yo(NY[i + ibegin], K);

      for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
        long long j = CCols[ij];
        long long N = DimsLr[j];

        Vector_t xo(NX[j], N);
        Matrix_t c(C[ij], K, N, Stride_t(UpperStride[i], 1));
        yo.noalias() += c * xo;
      }
      Matrix_t q(Q[i + ibegin], M, K, Stride_t(M, 1));
      y.noalias() = q * yo;
    }
  }
}

void H2Matrix::matVecLeafHorizontalPass(const ColCommMPI& comm) {
  typedef Eigen::Map<Eigen::VectorXcd> Vector_t;
  typedef Eigen::Map<Eigen::MatrixXcd> Matrix_t;

  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    Vector_t y(Y[i + ibegin], M);

    if (0 < M)
      for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
        long long j = ACols[ij];
        long long N = Dims[j];

        Vector_t x(X[j], N);
        Matrix_t c(A[ij], M, N);
        y.noalias() += c * x;
      }
  }
}

void H2Matrix::resetX() {
  std::fill(X[0], X[0] + X.size(), std::complex<double>(0., 0.));
  std::fill(Y[0], Y[0] + Y.size(), std::complex<double>(0., 0.));
}

void H2Matrix::factorize(const ColCommMPI& comm) {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long dims_max = *std::max_element(Dims.begin(), Dims.end());

  typedef Eigen::Map<Eigen::MatrixXcd> Matrix_t;
  std::vector<long long> ipiv_offsets(nodes);
  std::exclusive_scan(Dims.begin() + ibegin, Dims.begin() + (ibegin + nodes), ipiv_offsets.begin(), 0ll);

  comm.level_merge(A[0], A.size());
  for (long long i = 0; i < nodes; i++) {
    long long diag = lookupIJ(ARows, ACols, i, i + ibegin);
    long long M = Dims[i + ibegin];
    long long Ms = DimsLr[i + ibegin];
    long long Mr = M - Ms;

    Matrix_t Ui(Q[i + ibegin], M, M);
    Matrix_t V(R[i + ibegin], M, M);
    Matrix_t Aii(A[diag], M, M);
    V.noalias() = Ui.adjoint() * Aii.transpose();
    Aii.noalias() = Ui.adjoint() * V.transpose();

    Eigen::PartialPivLU<Eigen::MatrixXcd> plu = Aii.bottomRightCorner(Mr, Mr).lu();
    Eigen::MatrixXcd b(dims_max, M);
    Eigen::Map<Eigen::VectorXi> ipiv(Ipivots.data() + ipiv_offsets[i], Mr);

    Aii.bottomRightCorner(Mr, Mr) = plu.matrixLU();
    ipiv = plu.permutationP().indices();
    V.bottomRows(Mr) = plu.solve(Ui.rightCols(Mr).adjoint());

    if (0 < Ms) {
      Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> An(NA[diag], Ms, Ms, Eigen::Stride<Eigen::Dynamic, 1>(UpperStride[i], 1));
      Aii.bottomLeftCorner(Mr, Ms) = plu.solve(Aii.bottomLeftCorner(Mr, Ms));
      An.noalias() = Aii.topLeftCorner(Ms, Ms) - Aii.topRightCorner(Ms, Mr) * Aii.bottomLeftCorner(Mr, Ms);
      V.topRows(Ms) = Ui.leftCols(Ms).adjoint();
    }

    for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) 
      if (ij != diag) {
        long long j = ACols[ij];
        long long N = Dims[j];
        long long Ns = DimsLr[j];

        Matrix_t Uj(Q[j], N, N);
        Matrix_t Aij(A[ij], M, N);

        b.topRows(N) = Uj.adjoint() * Aij.transpose();
        Aij.noalias() = V * b.topRows(N).transpose();

        if (0 < Ms && 0 < Ns) {
          Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> An(NA[ij], Ms, Ns, Eigen::Stride<Eigen::Dynamic, 1>(UpperStride[i], 1));
          An = Aij.topLeftCorner(Ms, Ns);
        }
      }
  }
}

void H2Matrix::forwardSubstitute(const ColCommMPI& comm) {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  std::vector<long long> ipiv_offsets(nodes);
  std::exclusive_scan(Dims.begin() + ibegin, Dims.begin() + (ibegin + nodes), ipiv_offsets.begin(), 0ll);

  typedef Eigen::Map<Eigen::VectorXcd> Vector_t;
  typedef Eigen::Map<const Eigen::MatrixXcd> Matrix_t;

  comm.level_merge(X[0], X.size());
  for (long long i = 0; i < nodes; i++) {
    long long diag = lookupIJ(ARows, ACols, i, i + ibegin);
    long long M = Dims[i + ibegin];
    long long Ms = DimsLr[i + ibegin];
    long long Mr = M - Ms;

    Vector_t x(X[i + ibegin], M);
    Vector_t y(Y[i + ibegin], M);
    Matrix_t q(Q[i + ibegin], M, M);
    Matrix_t Aii(A[diag], M, M);
    Eigen::PermutationMatrix<Eigen::Dynamic> p(Eigen::Map<Eigen::VectorXi>(Ipivots.data() + ipiv_offsets[i], Mr));
    
    y.noalias() = q.adjoint() * x;
    y.bottomRows(Mr).applyOnTheLeft(p);
    Aii.bottomRightCorner(Mr, Mr).triangularView<Eigen::UnitLower>().solveInPlace(y.bottomRows(Mr));
    Aii.bottomRightCorner(Mr, Mr).triangularView<Eigen::Upper>().solveInPlace(y.bottomRows(Mr));
    x = y;
  }

  comm.neighbor_bcast(X);
  for (long long i = 0; i < nodes; i++) {
    long long diag = lookupIJ(ARows, ACols, i, i + ibegin);
    long long M = Dims[i + ibegin];
    long long Ms = DimsLr[i + ibegin];
    long long Mr = M - Ms;

    Vector_t x(X[i + ibegin], M);
    if (0 < Ms) {
      for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
        long long j = ACols[ij];
        long long N = Dims[j];
        long long Ns = DimsLr[j];
        long long Nr = N - Ns;

        Vector_t xj(X[j], N);
        Matrix_t Aij(A[ij], M, N);
        x.topRows(Ms).noalias() -= Aij.topRightCorner(Ms, Nr) * xj.bottomRows(Nr);
      }
      Vector_t xo(NX[i + ibegin], Ms);
      xo = x.topRows(Ms);
    }

    Vector_t y(Y[i + ibegin], M);
    y = x;
    for (long long ij = ARows[i]; ij < diag; ij++) {
      long long j = ACols[ij];
      long long N = Dims[j];
      long long Ns = DimsLr[j];
      long long Nr = N - Ns;

      Vector_t xj(X[j], N);
      Matrix_t Aij(A[ij], M, N);
      y.bottomRows(Mr).noalias() -= Aij.bottomRightCorner(Mr, Nr) * xj.bottomRows(Nr);
    }
  }

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    Vector_t x(X[i + ibegin], M);
    Vector_t y(Y[i + ibegin], M);
    x = y;
  }
}

void H2Matrix::backwardSubstitute(const ColCommMPI& comm) {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();

  typedef Eigen::Map<Eigen::VectorXcd> Vector_t;
  typedef Eigen::Map<const Eigen::MatrixXcd> Matrix_t;

  comm.neighbor_bcast(X);
  for (long long i = 0; i < nodes; i++) {
    long long diag = lookupIJ(ARows, ACols, i, i + ibegin);
    long long M = Dims[i + ibegin];
    long long Ms = DimsLr[i + ibegin];
    long long Mr = M - Ms;

    Vector_t x(X[i + ibegin], M);
    Vector_t y(Y[i + ibegin], M);

    y.bottomRows(Mr) = x.bottomRows(Mr);
    if (0 < Ms) {
      Vector_t xo(NX[i + ibegin], Ms);
      y.topRows(Ms) = xo;
    }

    for (long long ij = diag + 1; ij < ARows[i + 1]; ij++) {
      long long j = ACols[ij];
      long long N = Dims[j];
      long long Ns = DimsLr[j];
      long long Nr = N - Ns;

      Vector_t xj(X[j], N);
      Matrix_t Aij(A[ij], M, N);
      y.bottomRows(Mr).noalias() -= Aij.bottomRightCorner(Mr, Nr) * xj.bottomRows(Nr);
    }

    for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
      long long j = ACols[ij];
      long long N = Dims[j];
      long long Ns = DimsLr[j];

      if (0 < Ns) {
        Vector_t xo(NX[j], Ns);
        Matrix_t Aij(A[ij], M, N);
        y.bottomRows(Mr).noalias() -= Aij.bottomLeftCorner(Mr, Ns) * xo;
      }
    }
  }

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    Vector_t x(X[i + ibegin], M);
    Vector_t y(Y[i + ibegin], M);
    Matrix_t q(Q[i + ibegin], M, M);
    x.noalias() = q.conjugate() * y;
  }

  comm.neighbor_bcast(X);
}
