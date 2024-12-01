#include <h2matrix.hpp>
#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <kernel.hpp>

#include <numeric>
#include <algorithm>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/LU>


/* explicit template instantiation */
// complex double
template class WellSeparatedApproximation<std::complex<double>>;
template class MatrixDataContainer<std::complex<double>>;
template class H2Matrix<std::complex<double>>;
// complex float
template class WellSeparatedApproximation<std::complex<float>>;
template class MatrixDataContainer<std::complex<float>>;
template class H2Matrix<std::complex<float>>;
// complex half
//template class WellSeparatedApproximation<std::complex<Eigen::half>>;
//template class H2Matrix<std::complex<Eigen::half>>;
// double
template class WellSeparatedApproximation<double>;
template class MatrixDataContainer<double>;
template class H2Matrix<double>;
// float
template class WellSeparatedApproximation<float>;
template class MatrixDataContainer<float>;
template class H2Matrix<float>;

/* supported type conversions */
// (complex) double to float
template H2Matrix<std::complex<float>>::H2Matrix(const H2Matrix<std::complex<double>>&);
template H2Matrix<float>::H2Matrix(const H2Matrix<double>&);
// (complex) float to double
template H2Matrix<std::complex<double>>::H2Matrix(const H2Matrix<std::complex<float>>&);
template H2Matrix<double>::H2Matrix(const H2Matrix<float>&);

template <typename DT>
void WellSeparatedApproximation<DT>::construct(const MatrixAccessor<DT>& eval, double epi, long long rank, long long lbegin, long long len, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation<DT>& upper) {
  WellSeparatedApproximation::lbegin = lbegin;
  lend = lbegin + len;
  M.resize(len);
  for (long long i = upper.lbegin; i < upper.lend; i++)
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
      long long iters = adaptive_cross_approximation<DT>(epi, eval, m, n, k, ybodies, xbodies, nullptr, &ipiv[0], nullptr, nullptr);
      ipiv.resize(iters);

      Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic>> Xbodies(xbodies, 3, n);
      Eigen::VectorXd Fbodies(3 * iters);

      Fbodies = Xbodies(Eigen::all, ipiv).reshaped();
      M[y - lbegin].insert(M[y - lbegin].end(), Fbodies.begin(), Fbodies.end());
    }
  }
}

template <typename DT>
long long WellSeparatedApproximation<DT>::fbodies_size_at_i(long long i) const {
  return 0 <= i && i < (long long)M.size() ? M[i].size() / 3 : 0;
}

template <typename DT>
const double* WellSeparatedApproximation<DT>::fbodies_at_i(long long i) const {
  return 0 <= i && i < (long long)M.size() ? M[i].data() : nullptr;
}

template <typename DT>
long long compute_basis(const MatrixAccessor<DT>& eval, double epi, long long M, long long N, double Xbodies[], const double Fbodies[], DT a[], DT c[], bool orth) {
  typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;
  long long K = std::min(M, N), rank = 0;
  if (0 < K) {
    Matrix_dt RX = Matrix_dt::Zero(K, M);

    if (K < N) {
      Matrix_dt XF(N, M);
      gen_matrix(eval, N, M, Fbodies, Xbodies, XF.data());
      Eigen::HouseholderQR<Matrix_dt> qr(XF);
      RX = qr.matrixQR().topRows(K).template triangularView<Eigen::Upper>();
    }
    else
      gen_matrix(eval, N, M, Fbodies, Xbodies, RX.data());
    
    Eigen::ColPivHouseholderQR<Eigen::Ref<Matrix_dt>> rrqr(RX);
    rank = std::min(K, (long long)std::floor(epi));
    if (epi < 1.) {
      rrqr.setThreshold(epi);
      rank = rrqr.rank();
    }

    Eigen::Map<Matrix_dt> A(a, M, M), C(c, M, M);
    if (0 < rank && rank < M) {
      C.topRows(rank) = rrqr.matrixR().topRows(rank);
      C.topLeftCorner(rank, rank).template triangularView<Eigen::Upper>().solveInPlace(C.topRightCorner(rank, M - rank));
      C.topLeftCorner(rank, rank) = Matrix_dt::Identity(rank, rank);

      Eigen::Map<Eigen::MatrixXd> body(Xbodies, 3, M);
      body = body * rrqr.colsPermutation();

      if (orth) {
        RX = A.template triangularView<Eigen::Upper>() * (rrqr.colsPermutation() * C.topRows(rank).transpose());
        Eigen::HouseholderQR<Eigen::Ref<Matrix_dt>> qr(RX);
        A = qr.householderQ();
        C.setZero();
        C.topLeftCorner(rank, rank) = qr.matrixQR().topRows(rank).template triangularView<Eigen::Upper>();
      }
      else {
        A.setZero();
        A.leftCols(rank) = rrqr.colsPermutation() * C.topRows(rank).transpose();
        C.setZero();
        C.topLeftCorner(rank, rank) = Matrix_dt::Identity(rank, rank);
      }
    }
    else {
      C = A.template triangularView<Eigen::Upper>();
      A = Matrix_dt::Identity(M, M);
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

template <typename DT>
H2Matrix<DT>::H2Matrix(const H2Matrix<DT>& h2matrix) : UpperStride(h2matrix.UpperStride), S(h2matrix.S),
  NA(h2matrix.NA), NbXoffsets(h2matrix.NbXoffsets), NbZoffsets(h2matrix.NbZoffsets), nodes(h2matrix.nodes),
  lenX(h2matrix.lenX), LowerZ(h2matrix.LowerZ), Dims(h2matrix.Dims), DimsLr(h2matrix.DimsLr), ARows(h2matrix.ARows), ACols(h2matrix.ACols),
  CRows(h2matrix.CRows), CCols(h2matrix.CCols), Q(h2matrix.Q), R(h2matrix.R), A(h2matrix.A), C(h2matrix.C), U(h2matrix.U),
  X(h2matrix.X), Y(h2matrix.Y), Z(h2matrix.Z), W(h2matrix.W) {}

template <typename DT> template <typename OT>
H2Matrix<DT>::H2Matrix(const H2Matrix<OT>& h2matrix) : UpperStride(h2matrix.UpperStride), S(h2matrix.S),
  NA(h2matrix.NA), NbXoffsets(h2matrix.NbXoffsets), NbZoffsets(h2matrix.NbZoffsets), nodes(h2matrix.nodes),
  lenX(h2matrix.lenX), LowerZ(h2matrix.LowerZ), Dims(h2matrix.Dims), DimsLr(h2matrix.DimsLr), ARows(h2matrix.ARows), ACols(h2matrix.ACols),
  CRows(h2matrix.CRows), CCols(h2matrix.CCols), Q(h2matrix.Q), R(h2matrix.R), A(h2matrix.A), C(h2matrix.C), U(h2matrix.U),
  X(h2matrix.X), Y(h2matrix.Y), Z(h2matrix.Z), W(h2matrix.W) {}

template <typename DT>
void H2Matrix<DT>::construct(const MatrixAccessor<DT>& eval, double epi, const Cell cells[], const CSR& Near, const CSR& Far, const double bodies[], const WellSeparatedApproximation<DT>& wsa, const long long nodes, H2Matrix<DT>& lowerA, const long long lowerNodes, const std::pair<long long, long long> Tree[]) {
  //long long nodes = comm.lenNeighbors();
  //long long ibegin = comm.oLocal();
  //long long nodes = comm.lenLocal();
  //std::cout<<"ilocal "<<lowerComm.lenLocal()<<std::endl;
  this->nodes = nodes;
  long long ybegin = nodes - 1;

  Dims.resize(nodes, 0);
  DimsLr.resize(nodes, 0);
  UpperStride.resize(nodes, 0);

  long long tbegin = nodes - 1;
  long long tend = tbegin + nodes;
  //ARows.insert(ARows.begin(), comm.ARowOffsets.begin(), comm.ARowOffsets.end());
  ARows.insert(ARows.begin(), Near.RowIndex.begin() + tbegin, Near.RowIndex.begin() + tend + 1);
  long long offset = ARows[0];
  std::for_each(ARows.begin(), ARows.end(), [=](long long& i) { i = i - offset; });
  //ACols.insert(ACols.begin(), comm.AColumns.begin(), comm.AColumns.end());
  ACols.insert(ACols.begin(), Near.ColIndex.begin() + Near.RowIndex[tbegin], Near.ColIndex.begin() + Near.RowIndex[tend]);
  std::for_each(ACols.begin(), ACols.end(), [=](long long& i) { i = i - tbegin; });
  //CRows.insert(CRows.begin(), comm.CRowOffsets.begin(), comm.CRowOffsets.end());
  CRows.insert(CRows.begin(), Far.RowIndex.begin() + tbegin, Far.RowIndex.begin() + tend + 1);
  offset = CRows[0];
  std::for_each(CRows.begin(), CRows.end(), [=](long long& i) { i = i - offset; });
  //CCols.insert(CCols.begin(), comm.CColumns.begin(), comm.CColumns.end());
  CCols.insert(CCols.begin(), Far.ColIndex.begin() + Far.RowIndex[tbegin], Far.ColIndex.begin() + Far.RowIndex[tend]);
  std::for_each(CCols.begin(), CCols.end(), [=](long long& i) { i = i - tbegin; });
  NA.resize(ARows[nodes], -1);
  
  //std::cout<<lowerNodes<<std::endl;
  //std::cout<<nodes<<std::endl;
  if (lowerNodes - nodes > 0) {
    long long lbegin = lowerNodes -1;
    long long lend = lbegin + lowerNodes;
    //std::cout<<"lbegin "<<lbegin<<std::endl;
    //std::cout<<"lend "<<lend<<std::endl;
    long long lenAl = Near.RowIndex[lend] - Near.RowIndex[lbegin];
    long long lenCl = Far.RowIndex[lend] - Far.RowIndex[lbegin];

    //LowerX = Tree[tbegin].first - lbegin;
    LowerIndA.resize(lenAl);
    LowerIndC.resize(lenCl);

    for (long long i = tbegin; i < tend; i++) {
      long long childi = Tree[i].first;
      long long cendi = Tree[i].second;

      for (long long ij = Near.RowIndex[i]; ij < Near.RowIndex[i + 1]; ij++) {
        long long j = Near.ColIndex[ij];
        long long childj = Tree[j].first;
        long long cendj = Tree[j].second;

        for (long long y = std::max(lbegin, childi); y < std::min(lend, cendi); y++)
          for (long long x = childj; x < cendj; x++) {
            long long A_yx = std::distance(&Near.ColIndex[Near.RowIndex[lbegin]], std::find(&Near.ColIndex[Near.RowIndex[y]], &Near.ColIndex[Near.RowIndex[y + 1]], x));
            long long C_yx = std::distance(&Far.ColIndex[Far.RowIndex[lbegin]], std::find(&Far.ColIndex[Far.RowIndex[y]], &Far.ColIndex[Far.RowIndex[y + 1]], x));

            if (A_yx < (Near.RowIndex[y + 1] - Near.RowIndex[lbegin]))
              LowerIndA[A_yx] = std::make_tuple(y - childi, x - childj, ij - Near.RowIndex[tbegin]);
            else if (C_yx < (Far.RowIndex[y + 1] - Far.RowIndex[lbegin]))
              LowerIndC[C_yx] = std::make_tuple(y - childi, x - childj, ij - Near.RowIndex[tbegin]);
          }
      }
    }
  }

  //std::cout<<ARows.size()<<std::endl;
  //std::cout<<ACols.size()<<std::endl;

  long long localChildLen = cells[ybegin + nodes - 1].Child[1] - cells[ybegin].Child[0];
  std::vector<long long> localChildOffsets(nodes + 1, -1);
  
  if (0 < localChildLen) {
    //long long lowerBegin = comm.LowerX;
    long long lowerBegin = 0;
    long long localChildIndex = lowerBegin - cells[ybegin].Child[0];
    std::transform(&cells[ybegin], &cells[ybegin + nodes], localChildOffsets.begin() + 1, [=](const Cell& c) { return localChildIndex + c.Child[1]; });
    localChildOffsets[0] = lowerBegin;

    std::vector<long long> ranks_offsets(localChildLen + 1);
    std::inclusive_scan(lowerA.DimsLr.begin() + localChildOffsets[0], lowerA.DimsLr.begin() + localChildOffsets[nodes], ranks_offsets.begin() + 1);
    ranks_offsets[0] = 0;

    std::transform(localChildOffsets.begin(), localChildOffsets.begin() + nodes, localChildOffsets.begin() + 1, &Dims[0],
      [&](long long start, long long end) { return ranks_offsets[end - lowerBegin] - ranks_offsets[start - lowerBegin]; });

    lenX = ranks_offsets.back();
    LowerZ = std::reduce(lowerA.DimsLr.begin(), lowerA.DimsLr.begin() + lowerBegin, 0ll);
  }
  else {
    std::transform(&cells[ybegin], &cells[ybegin + nodes], &Dims[0], [](const Cell& c) { return c.Body[1] - c.Body[0]; });
    lenX = std::reduce(&Dims[0], &Dims[nodes]);
    LowerZ = 0;
  }

  std::vector<long long> neighbor_ones(nodes, 1ll);
  //comm.dataSizesToNeighborOffsets(neighbor_ones.data());
  //comm.neighbor_bcast(Dims.data(), neighbor_ones.data());
  X.alloc(nodes, Dims.data());
  Y.alloc(nodes, Dims.data());

  std::vector<long long> Qsizes(nodes, 0);
  std::transform(Dims.begin(), Dims.end(), Qsizes.begin(), [](const long long d) { return d * d; });
  Q.alloc(nodes, Qsizes.data());
  R.alloc(nodes, Qsizes.data());

  std::vector<long long> Ssizes(nodes);
  std::transform(Dims.begin(), Dims.end(), Ssizes.begin(), [](const long long d) { return 3 * d; });
  S.alloc(nodes, Ssizes.data());

  std::vector<long long> Asizes(ARows[nodes]);
  for (long long i = 0; i < nodes; i++)
    std::transform(ACols.begin() + ARows[i], ACols.begin() + ARows[i + 1], Asizes.begin() + ARows[i],
      [&](long long col) { return Dims[i] * Dims[col]; });
  A.alloc(ARows[nodes], Asizes.data());

  typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;
  typedef Eigen::Stride<Eigen::Dynamic, 1> Stride_t;
  typedef Eigen::Map<Matrix_dt, Eigen::Unaligned, Stride_t> MatrixMap_dt;

  if (std::reduce(Dims.begin(), Dims.end())) {
    long long pbegin = 0;
    //long long pend = pbegin + lowerComm.lenLocal();
    long long pend = pbegin + lowerNodes;

    for (long long i = 0; i < nodes; i++) {
      long long M = Dims[i];
      long long childi = localChildOffsets[i];
      long long cendi = localChildOffsets[i + 1];
      Eigen::Map<Matrix_dt> Qi(Q[i], M, M);

      for (long long y = childi; y < cendi; y++) { // Intermediate levels
        long long offset_y = std::reduce(&lowerA.DimsLr[childi], &lowerA.DimsLr[y]);
        long long ny = lowerA.DimsLr[y];
        std::copy(lowerA.S[y], lowerA.S[y] + (ny * 3), &(S[i])[offset_y * 3]);

        MatrixMap_dt Ry(lowerA.R[y], ny, ny, Stride_t(lowerA.Dims[y], 1));
        Qi.block(offset_y, offset_y, ny, ny) = Ry;

        if (pbegin <= y && y < pend && 0 < M) {
          long long py = y - pbegin;
          lowerA.UpperStride[py] = M;

          for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
            long long j_global = Near.ColIndex[ij + Near.RowIndex[ybegin]];
            //long long childj = lowerComm.iLocal(cells[j_global].Child[0]);
            long long childj = cells[j_global].Child[0] - lowerNodes + 1; 
            long long cendj = (0 <= childj) ? (childj + cells[j_global].Child[1] - cells[j_global].Child[0]) : -1;

            for (long long x = childj; x < cendj; x++) {
              long long offset_x = std::reduce(&lowerA.DimsLr[childj], &lowerA.DimsLr[x]);
              long long nx = lowerA.DimsLr[x];
              long long lowN = lookupIJ(lowerA.ARows, lowerA.ACols, py, x);
              long long lowC = lookupIJ(lowerA.CRows, lowerA.CCols, py, x);
              DT* dp = A[ij] + offset_y + offset_x * M;
              if (0 <= lowN)
                lowerA.NA[lowN] = std::distance(A[0], dp);
              else if (0 <= lowC)
                MatrixMap_dt(dp, ny, nx, Stride_t(M, 1)) = Eigen::Map<Matrix_dt>(lowerA.C[lowC], ny, nx);
            }
          }
        }
      }

      if (cendi <= childi) { // Leaf level
        long long ci = i + ybegin;
        std::copy(&bodies[3 * cells[ci].Body[0]], &bodies[3 * cells[ci].Body[1]], S[i]);
        Qi = Matrix_dt::Identity(M, M);

        for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
          long long N = Dims[ACols[ij]];
          long long cj = Near.ColIndex[ij + Near.RowIndex[ybegin]];
          gen_matrix(eval, M, N, &bodies[3 * cells[ci].Body[0]], &bodies[3 * cells[cj].Body[0]], A[ij]);
        }
      }
    }

    //comm.dataSizesToNeighborOffsets(Ssizes.data());
    //comm.neighbor_bcast(S[0], Ssizes.data());
    std::vector<std::vector<double>> cbodies(nodes);

    // epi is not accessed until here
    // same goes for wsa, so until here everything should be fine
    for (long long i = 0; i < nodes; i++) {
      // gets the number of sampled points
      long long fsize = wsa.fbodies_size_at_i(i);
      const double* fbodies = wsa.fbodies_at_i(i);

      //when the rank is fixed
      if (1. <= epi) {
        std::vector<long long>::iterator neighbors = ACols.begin() + ARows[i];
        std::vector<long long>::iterator neighbors_end = ACols.begin() + ARows[i + 1];
        long long csize = std::transform_reduce(neighbors, neighbors_end, -Dims[i], std::plus<long long>(), [&](long long col) { return Dims[col]; });

        cbodies[i] = std::vector<double>(3 * (fsize + csize));
        long long loc = 0;
        for (long long n = 0; n < (ARows[i + 1] - ARows[i]); n++) {
          long long col = neighbors[n];
          long long len = 3 * Dims[col];
          if (col != i) {
            std::copy(S[col], S[col] + len, cbodies[i].begin() + loc);
            loc += len;
          }
        }
        std::copy(fbodies, &fbodies[3 * fsize], cbodies[i].begin() + loc);
      }
      //when the accuracy is fixed
      else {
        cbodies[i] = std::vector<double>(3 * fsize);
        std::copy(fbodies, &fbodies[3 * fsize], cbodies[i].begin());
      }
    }

    // fbodies is different
    for (long long i = 0; i < nodes; i++) {
      long long fsize = cbodies[i].size() / 3;
      const double* fbodies = cbodies[i].data();
      long long rank = compute_basis(eval, epi, Dims[i], fsize, S[i], fbodies, Q[i], R[i], 1. <= epi);
      DimsLr[i] = rank;
    }

    //comm.dataSizesToNeighborOffsets(Qsizes.data());
    //comm.neighbor_bcast(DimsLr.data(), neighbor_ones.data());
    //comm.neighbor_bcast(S[0], Ssizes.data());
    //comm.neighbor_bcast(Q[0], Qsizes.data());
    //comm.neighbor_bcast(R[0], Qsizes.data());
  }

  if (std::reduce(DimsLr.begin(), DimsLr.end())) {
    std::vector<long long> Csizes(CRows[nodes]);
    for (long long i = 0; i < nodes; i++)
      std::transform(CCols.begin() + CRows[i], CCols.begin() + CRows[i + 1], Csizes.begin() + CRows[i],
        [&](long long col) { return DimsLr[i] * DimsLr[col]; });
    C.alloc(CRows[nodes], Csizes.data());

    std::vector<long long> Usizes(nodes);
    std::transform(&Dims[0], &Dims[nodes], &DimsLr[0], Usizes.begin(), std::multiplies<long long>());
    U.alloc(nodes, Usizes.data());
    Z.alloc(nodes, DimsLr.data());
    W.alloc(nodes, DimsLr.data());

    for (long long i = 0; i < nodes; i++) {
      long long y = i;
      long long M = DimsLr[y];
      MatrixMap_dt Ry(R[y], M, M, Stride_t(Dims[y], 1));
      Eigen::Map<Matrix_dt>(U[i], Dims[y], M) = Eigen::Map<Matrix_dt>(Q[y], Dims[y], M);

      for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
        long long x = CCols[ij];
        long long N = DimsLr[CCols[ij]];
        MatrixMap_dt Rx(R[x], N, N, Stride_t(Dims[x], 1));

        Eigen::Map<Matrix_dt> Cyx(C[ij], M, N);
        if (1. <= epi) {
          Matrix_dt Ayx(M, N);
          gen_matrix(eval, M, N, S[y], S[x], Ayx.data());
          Cyx.noalias() = Ry.template triangularView<Eigen::Upper>() * Ayx * Rx.transpose().template triangularView<Eigen::Lower>();
        }
        else
          gen_matrix(eval, M, N, S[y], S[x], Cyx.data());
      }
    }
  }

  NbXoffsets.insert(NbXoffsets.begin(), Dims.begin(), Dims.end());
  //NbXoffsets.erase(NbXoffsets.begin() + comm.dataSizesToNeighborOffsets(NbXoffsets.data()), NbXoffsets.end());
  NbZoffsets.insert(NbZoffsets.begin(), DimsLr.begin(), DimsLr.end());
  //NbZoffsets.erase(NbZoffsets.begin() + comm.dataSizesToNeighborOffsets(NbZoffsets.data()), NbZoffsets.end());
}

template <typename DT>
void H2Matrix<DT>::matVecUpwardPass(const DT* X_in) {
  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, 1>> VectorMap_dt;
  typedef Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> MatrixMap_dt;

  //long long ibegin = comm.oLocal();
  //long long nodes = comm.lenLocal();
  std::copy(&X_in[LowerZ], &X_in[LowerZ + lenX], X[0]);

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i];
    long long N = DimsLr[i];
    VectorMap_dt x(X[i], M);
    if (0 < N) {
      VectorMap_dt z(Z[i], N);
      MatrixMap_dt q(Q[i], M, N);
      z = q.transpose() * x;
    }
  }

  //comm.neighbor_bcast(Z[0], NbZoffsets.data());
}

template <typename DT>
void H2Matrix<DT>::matVecHorizontalandDownwardPass(DT* Y_out) {
  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, 1>> VectorMap_dt;
  typedef Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> MatrixMap_dt;

  //long long ibegin = comm.oLocal();
  //long long nodes = comm.lenLocal();

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i];
    long long K = DimsLr[i];
    if (0 < K) {
      VectorMap_dt w(W[i], K);
      for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
        long long j = CCols[ij];
        long long N = DimsLr[j];

        VectorMap_dt z(Z[j], N);
        MatrixMap_dt c(C[ij], K, N);
        w.noalias() += c * z;
      }

      MatrixMap_dt q(Q[i], M, K);
      VectorMap_dt y(Y[i], M);
      y.noalias() = q * w;
    }
  }

  std::copy(Y[0], Y[nodes], &Y_out[LowerZ]);
}

template <typename DT>
void H2Matrix<DT>::matVecLeafHorizontalPass(DT* X_io) {
  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, 1>> VectorMap_dt;
  typedef Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> MatrixMap_dt;

  //long long ibegin = comm.oLocal();
  //long long nodes = comm.lenLocal();
  std::copy(&X_io[0], &X_io[lenX], X[0]);
  //comm.neighbor_bcast(X[0], NbXoffsets.data());

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i];
    long long K = DimsLr[i];
    VectorMap_dt y(Y[i], M);
    y.setZero();

    if (0 < K) {
      VectorMap_dt w(W[i], K);
      for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
        long long j = CCols[ij];
        long long N = DimsLr[j];

        VectorMap_dt z(Z[j], N);
        MatrixMap_dt c(C[ij], K, N);
        w.noalias() += c * z;
      }

      MatrixMap_dt q(Q[i], M, K);
      y.noalias() += q * w;
    }

    for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
      long long j = ACols[ij];
      long long N = Dims[j];

      VectorMap_dt x(X[j], N);
      MatrixMap_dt c(A[ij], M, N);
      y.noalias() += c * x;
    }
  }

  std::copy(Y[0], Y[nodes], X_io);
}

template <typename DT>
void H2Matrix<DT>::factorize() {
  //long long ibegin = comm.oLocal();
  //long long nodes = comm.lenLocal();
  //long long xlen = comm.lenNeighbors();

  long long dims_max = *std::max_element(Dims.begin(), Dims.end());
  typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;
  typedef Eigen::Map<Matrix_dt> MatrixMap_dt;

  std::vector<long long> Bsizes(nodes);
  std::fill(Bsizes.begin(), Bsizes.end(), dims_max * dims_max);
  MatrixDataContainer<DT> B;
  B.alloc(nodes, Bsizes.data());

  //if (nodes == 1)
    //comm.level_merge(A[0], A.size());

  for (long long i = 0; i < nodes; i++) {
    long long diag = lookupIJ(ARows, ACols, i, i);
    long long M = Dims[i];
    long long Ms = DimsLr[i];
    long long Mr = M - Ms;

    MatrixMap_dt Ui(Q[i], M, M);
    MatrixMap_dt V(R[i], M, M);
    MatrixMap_dt Aii(A[diag], M, M);
    MatrixMap_dt b(B[i], dims_max, M);

    b.topRows(M).noalias() = Ui.adjoint() * Aii.transpose();
    Aii.noalias() = Ui.adjoint() * b.topRows(M).transpose();
    V.topRows(Ms) = Ui.leftCols(Ms).adjoint();

    if (0 < Mr) {
      Eigen::HouseholderQR<Matrix_dt> fac(Aii.bottomRightCorner(Mr, Mr));
      V.bottomRows(Mr) = fac.solve(Ui.rightCols(Mr).adjoint());
      if (0 < Ms) {
        Aii.bottomLeftCorner(Mr, Ms).noalias() = V.bottomRows(Mr) * b.topRows(Ms).transpose();
        Aii.topLeftCorner(Ms, Ms).noalias() -= Aii.topRightCorner(Ms, Mr) * Aii.bottomLeftCorner(Mr, Ms);
      }
    }

    for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) 
      if (ij != diag) {
        long long j = ACols[ij];
        long long N = Dims[j];

        MatrixMap_dt Uj(Q[j], N, N);
        MatrixMap_dt Aij(A[ij], M, N);

        b.topRows(N) = Uj.adjoint() * Aij.transpose();
        Aij.noalias() = V * b.topRows(N).transpose();
      }
    
    b.topLeftCorner(Mr, Ms) = Aii.bottomLeftCorner(Mr, Ms);
    b.topRightCorner(Mr, Mr) = V.bottomRows(Mr) * Ui.rightCols(Mr);
  }
  //comm.dataSizesToNeighborOffsets(Bsizes.data());
  //comm.neighbor_bcast(B[0], Bsizes.data());

  for (long long i = 0; i < nodes; i++) {
    long long diag = lookupIJ(ARows, ACols, i, i);
    long long M = Dims[i];
    long long Ms = DimsLr[i];
    long long Mr = M - Ms;
    MatrixMap_dt Aii(A[diag], M, M);

    for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++)
      if (ij != diag) {
        long long j = ACols[ij];
        long long N = Dims[j];
        long long Ns = DimsLr[j];
        long long Nr = N - Ns;
        
        MatrixMap_dt Aij(A[ij], M, N);
        MatrixMap_dt Bj(B[j], dims_max, N);
        Aij.topLeftCorner(Ms, Ns) -= Aii.topRightCorner(Ms, Mr) * Aij.bottomLeftCorner(Mr, Ns) + Aij.topRightCorner(Ms, Nr) * Bj.topLeftCorner(Nr, Ns);
        Aii.topLeftCorner(Ms, Ms) -= Aij.topRightCorner(Ms, Nr) * Bj.topRightCorner(Nr, Nr) * Aij.topRightCorner(Ms, Nr).transpose();
      }
  }
}

template <typename DT>
void H2Matrix<DT>::factorizeCopyNext(const H2Matrix<DT>& lowerA) {
  //long long ibegin = lowerComm.oLocal();
  //long long nodes = lowerComm.lenLocal();
  typedef Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> MatrixMap_dt;

  for (long long i = 0; i < nodes; i++)
    for (long long ij = lowerA.ARows[i]; ij < lowerA.ARows[i + 1]; ij++) {
      long long j = lowerA.ACols[ij];
      long long M = lowerA.Dims[i];
      long long N = lowerA.Dims[j];
      long long Ms = lowerA.DimsLr[i];
      long long Ns = lowerA.DimsLr[j];

      MatrixMap_dt Aij(lowerA.A[ij], M, N);
      if (0 < Ms && 0 < Ns) {
        Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> An(A[0] + lowerA.NA[ij], Ms, Ns, Eigen::Stride<Eigen::Dynamic, 1>(lowerA.UpperStride[i], 1));
        An = Aij.topLeftCorner(Ms, Ns);
      }
    }
}

template <typename DT>
void H2Matrix<DT>::forwardSubstitute(const DT* X_in) {
  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, 1>> VectorMap_dt;
  typedef Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> MatrixMap_dt;

  //long long ibegin = comm.oLocal();
  //long long nodes = comm.lenLocal();
  std::copy(&X_in[LowerZ], &X_in[LowerZ + lenX], Y[0]);

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i];

    if (0 < M) {
      VectorMap_dt x(X[i], M);
      VectorMap_dt y(Y[i], M);
      MatrixMap_dt q(R[i], M, M);
      x.noalias() = q * y;
    }
  }

  //comm.neighbor_bcast(X[0], NbXoffsets.data());

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i];
    long long Ms = DimsLr[i];

    if (0 < Ms) {
      VectorMap_dt z(Z[i], Ms);
      z = VectorMap_dt(X[i], Ms);

      for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
        long long j = ACols[ij];
        long long N = Dims[j];
        long long Ns = DimsLr[j];
        long long Nr = N - Ns;

        if (0 < Nr) {
          VectorMap_dt xj(X[j], N);
          MatrixMap_dt Aij(A[ij], M, N);
          z.noalias() -= Aij.topRightCorner(Ms, Nr) * xj.bottomRows(Nr);
        }
      }
    }
  }

  //comm.neighbor_bcast(Z[0], NbZoffsets.data());
}

template <typename DT>
void H2Matrix<DT>::backwardSubstitute(DT* Y_out) {
  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, 1>> VectorMap_dt;
  typedef Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> MatrixMap_dt;

  //long long ibegin = comm.oLocal();
  //long long nodes = comm.lenLocal();
  //comm.neighbor_bcast(W[0], NbZoffsets.data());

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i];
    long long Ms = DimsLr[i];
    long long Mr = M - Ms;

    VectorMap_dt x(X[i], M);
    x.topRows(Ms) = VectorMap_dt(W[i], Ms);
      
    if (0 < Mr) {
      for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
        long long j = ACols[ij];
        long long N = Dims[j];
        long long Ns = DimsLr[j];

        if (0 < Ns) {
          VectorMap_dt wj(W[j], Ns);
          MatrixMap_dt Aij(A[ij], M, N);
          x.bottomRows(Mr).noalias() -= Aij.bottomLeftCorner(Mr, Ns) * wj;
        }
      }
    }
  }

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i];
    if (0 < M) {
      VectorMap_dt x(X[i], M);
      VectorMap_dt y(Y[i], M);
      MatrixMap_dt q(Q[i], M, M);
      y.noalias() = q.conjugate() * x;
    }
  }

  std::copy(Y[0], Y[nodes], &Y_out[LowerZ]);
}
