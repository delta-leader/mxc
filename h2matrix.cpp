#include <h2matrix.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

#include <Eigen/Dense>


long long int log2floor(const long long int &x){return 63 - __builtin_clzll(x);}

// complex double
template class H2Matrix<std::complex<double>>;


template<typename MDT, typename DT>
long long compute_basis(const Eigen::DenseBase<MDT>& mat, double epi, long long s[], DT q[], DT r[], bool orth) {
  long long M = mat.rows();
  long long N = mat.cols();
  long long K = std::min(M, N);
  long long rank = 0;

  typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;

  if (0 < K) {
    Matrix_dt RX = Matrix_dt::Zero(K, N);

    if (K < M) {
      Eigen::HouseholderQR<Matrix_dt> qr(mat);
      RX = qr.matrixQR().topRows(K).template triangularView<Eigen::Upper>();
    } else {
      RX = mat;
    }
    
    // fixed a bug where we directly used mat here
    Eigen::ColPivHouseholderQR<Matrix_dt> rrqr(RX);
    rank = std::min(K, (long long)std::floor(epi));
    if (epi < 1.) {
      rrqr.setThreshold(epi);
      rank = rrqr.rank();
    }
    
    Eigen::Map<Matrix_dt> Q(q, N, N), R(r, N, N);
    if (0 < rank && rank < N) {
      R.topRows(rank) = rrqr.matrixR().topRows(rank);
      R.topLeftCorner(rank, rank).template triangularView<Eigen::Upper>().solveInPlace(R.topRightCorner(rank, N - rank));
      R.topLeftCorner(rank, rank) = Matrix_dt::Identity(rank, rank);
      
      Eigen::Map<Eigen::RowVector<long long, Eigen::Dynamic>> indices(s, N);
      indices = indices * rrqr.colsPermutation();

      if (orth) {
        Matrix_dt RX = Q.template triangularView<Eigen::Upper>() * (rrqr.colsPermutation() * R.topRows(rank).transpose());
        Eigen::HouseholderQR<Eigen::Ref<Matrix_dt>> qr(RX);
        Q = qr.householderQ();
        R.setZero();
        R.topLeftCorner(rank, rank) = qr.matrixQR().topRows(rank).template triangularView<Eigen::Upper>();
      }
      else {
        Q.setZero();
        Q.leftCols(rank) = rrqr.colsPermutation() * R.topRows(rank).transpose();
        R.setZero();
        R.topLeftCorner(rank, rank) = Matrix_dt::Identity(rank, rank);
      }
    }
    else {
      R = Q.template triangularView<Eigen::Upper>();
      Q = Matrix_dt::Identity(N, N);
    }
  }
  return rank;
}

template <typename DT>
H2Matrix<DT>::H2Matrix(const H2Matrix& h2matrix) : UpperStride(h2matrix.UpperStride), S_ind(h2matrix.S_ind),
  CRows(h2matrix.CRows), CCols(h2matrix.CCols), NA(h2matrix.NA), NbXoffsets(h2matrix.NbXoffsets), NbZoffsets(h2matrix.NbZoffsets),
  lenX(h2matrix.lenX), LowerZ(h2matrix.LowerZ), n_mat(h2matrix.n_mat),
  Dims(h2matrix.Dims), DimsLr(h2matrix.DimsLr), ARows(h2matrix.ARows), ACols(h2matrix.ACols),
  Q(h2matrix.Q), R(h2matrix.R), A(h2matrix.A), C(h2matrix.C), U(h2matrix.U),
  X(h2matrix.X), Y(h2matrix.Y), Z(h2matrix.Z), W(h2matrix.W) {}

inline long long lookupIJ(const std::vector<long long>& RowIndex, const std::vector<long long>& ColIndex, long long i, long long j) {
  if (i < 0 || RowIndex.size() <= (1ull + i))
    return -1;
  long long k = std::distance(ColIndex.begin(), std::find(ColIndex.begin() + RowIndex[i], ColIndex.begin() + RowIndex[i + 1], j));
  return (k < RowIndex[i + 1]) ? k : -1;
}


// create entire far field during construction
template <typename DT>
void H2Matrix<DT>::construct(const MatrixGenerator& matgen, double epi, const Cell cells[], const CSR& Near, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm, const double omega, bool verbose) {
  // number of cells on this level (this process and neighbors)
  // note that the tree might have beens split into subtrees further up
  long long xlen = comm.lenNeighbors();
  // index of the first cell for this process on this level
  long long ibegin = comm.oLocal();
  // number of cells for this process on this level
  long long nodes = comm.lenLocal();
  // index of the first cell for this process in the global cell array
  long long ybegin = comm.oGlobal();

  // dimensions for each cell on this level
  Dims.resize(xlen, 0);
  // LR dimensions for each cell on this level
  DimsLr.resize(xlen, 0);
  UpperStride.resize(nodes, 0);

  // get the Nearfield indices CSR
  ARows.insert(ARows.begin(), comm.ARowOffsets.begin(), comm.ARowOffsets.end());
  ACols.insert(ACols.begin(), comm.AColumns.begin(), comm.AColumns.end());
  // get the far-field indices CSR
  CRows.insert(CRows.begin(), comm.CRowOffsets.begin(), comm.CRowOffsets.end());
  CCols.insert(CCols.begin(), comm.CColumns.begin(), comm.CColumns.end());
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
  }
  else {
    // only for leaf level
    // Dims stores the number of particels for each cell (multiplied by 3)
    std::transform(&cells[ybegin], &cells[ybegin + nodes], &Dims[ibegin], [](const Cell& c) { return (c.Body[1] - c.Body[0]) * 3; });
    // total number of particles on this process
    lenX = std::reduce(&Dims[ibegin], &Dims[ibegin + nodes]);
    LowerZ = 0;
    n_mat = matgen.get_num_elems() * 3;
  }
  double time_gen = 0, time_comp = 0, start;

  std::vector<long long> neighbor_ones(xlen, 1ll);
  comm.dataSizesToNeighborOffsets(neighbor_ones.data());
  comm.neighbor_bcast(Dims.data(), neighbor_ones.data());
  X.alloc(xlen, Dims.data());
  Y.alloc(xlen, Dims.data());
  // S_ind stores the indices of the elements
  std::vector<long long> Ssizes(Dims);
  S_ind.alloc(xlen, Ssizes.data());

  std::vector<long long> Qsizes(xlen, 0);
  std::transform(Dims.begin(), Dims.end(), Qsizes.begin(), [](const long long d) { return d * d; });
  Q.alloc(xlen, Qsizes.data());
  R.alloc(xlen, Qsizes.data());

  // near field blocks (i.e. dense matrices on the leaf level), not necessarily square
  std::vector<long long> Asizes(ARows[nodes]);
  for (long long i = 0; i < nodes; i++)
    std::transform(ACols.begin() + ARows[i], ACols.begin() + ARows[i + 1], Asizes.begin() + ARows[i],
      [&](long long col) { return Dims[i + ibegin] * Dims[col]; });
  A.alloc(ARows[nodes], Asizes.data());

  typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;
  typedef Eigen::Stride<Eigen::Dynamic, 1> Stride_t;
  typedef Eigen::Map<Matrix_dt, Eigen::Unaligned, Stride_t> MatrixMap_dt;

  // skip blocks that contain no points (because they are split further)
  if (std::reduce(Dims.begin(), Dims.end())) {
    // index of the first cell for this process on this level
    long long pbegin = lowerComm.oLocal();
    // number of cells for this process/level
    long long pend = pbegin + lowerComm.lenLocal();

    // loop over all nodes
     for (long long i = 0; i < nodes; i++) {
      // number of rows in that cell
      long long M = Dims[i + ibegin];
      long long childi = localChildOffsets[i];
      long long cendi = localChildOffsets[i + 1];
      // get the corresponding Q matrix (as a reference)
      Eigen::Map<Matrix_dt> Qi(Q[i + ibegin], M, M);

      // for all children (i.e. only on the intermediate levels)
      for (long long y = childi; y < cendi; y++) {
        long long offset_y = std::reduce(&lowerA.DimsLr[childi], &lowerA.DimsLr[y]);
        long long ny = lowerA.DimsLr[y];
        // S_ind already has been broadcast on the lower level, so this is fine
        std::copy(lowerA.S_ind[y], lowerA.S_ind[y] + ny, &(S_ind[i + ibegin])[offset_y]);

        MatrixMap_dt Ry(lowerA.R[y], ny, ny, Stride_t(lowerA.Dims[y], 1));
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
              DT* dp = A[ij] + offset_y + offset_x * M;
              if (0 <= lowN)
                lowerA.NA[lowN] = std::distance(A[0], dp);
              else if (0 <= lowC)
                MatrixMap_dt(dp, ny, nx, Stride_t(M, 1)) = Eigen::Map<Matrix_dt>(lowerA.C[lowC], ny, nx);
            }
          }
        }
      }
       
      // leaf level (i.e. no children)
      if (cendi <= childi) {
        long long ci = i + ybegin;
        // global numbering of indices
        std::iota(S_ind[i + ibegin], S_ind[i + ibegin + 1], cells[ci].Body[0] * 3);
        Qi = Matrix_dt::Identity(M, M);

        long long far_cols = n_mat;
        const long long M_elem = M / 3;
        // generate the near field aka dense matrices in A
        for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
          long long N = Dims[ACols[ij]];
          far_cols -= N;
          long long cj = Near.ColIndex[ij + Near.RowIndex[ybegin]];
          Eigen::Map<Matrix_dt> A_ij(A[ij], M, N);
          start = MPI_Wtime();
          matgen.gen_matrix_sorted_single_layer(A_ij.data(), cells[ci].Body[0], M_elem, cells[cj].Body[0], N / 3, omega);
          time_gen += MPI_Wtime() - start;
        }
        if (1. <= epi) {
          // build an HSS basis
          far_cols = n_mat - M;
          Matrix_dt far(M, far_cols);
          start = MPI_Wtime();
          matgen.gen_matrix_sorted_single_layer(far.data(), cells[ci].Body[0], M_elem, 0, cells[ci].Body[0], omega);
          time_gen += MPI_Wtime() - start;
          start = MPI_Wtime();
          matgen.gen_matrix_sorted_single_layer(far.data() + cells[ci].Body[0] * 3 * M, cells[ci].Body[0], M_elem, cells[ci].Body[1], n_mat / 3 - cells[ci].Body[1], omega);
          time_gen += MPI_Wtime() - start;
          start = MPI_Wtime();
          long long rank = compute_basis(far.transpose(), epi, S_ind[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
          time_comp += MPI_Wtime() - start;
          DimsLr[i + ibegin] = rank;
        } else {
          // build an H2 basis
          // generate the far field only if it exists
          if (far_cols > 0) {
            Matrix_dt far(M, far_cols);
            long long current_near = Near.ColIndex[ARows[i] + Near.RowIndex[ybegin]];
            start = MPI_Wtime();
            matgen.gen_matrix_sorted_single_layer(far.data(), cells[ci].Body[0], M_elem, 0, cells[current_near].Body[0], omega);
            time_gen += MPI_Wtime() - start;
            long long start_cols = cells[current_near].Body[0] * 3;
            for (long long ij = ARows[i] + 1; ij < ARows[i + 1]; ij++) {
              long long next_near = Near.ColIndex[ij + Near.RowIndex[ybegin]];
              long long add_cols = cells[next_near].Body[0] - cells[current_near].Body[1];
              start = MPI_Wtime();
              matgen.gen_matrix_sorted_single_layer(far.data() + start_cols * M, cells[ci].Body[0], M_elem, cells[current_near].Body[1], add_cols, omega);
              time_gen += MPI_Wtime() - start;
              current_near = next_near;
            }
            start = MPI_Wtime();
            matgen.gen_matrix_sorted_single_layer(far.data() + start_cols * M, cells[ci].Body[0], M_elem, cells[current_near].Body[1], n_mat / 3 - cells[current_near].Body[1], omega);
            time_gen += MPI_Wtime() - start;
            start = MPI_Wtime();
            long long rank = compute_basis(far.transpose(), epi, S_ind[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
            time_comp += MPI_Wtime() - start;
            DimsLr[i + ibegin] = rank;
          }
        }
      }
    }

    comm.dataSizesToNeighborOffsets(Ssizes.data());
    comm.neighbor_bcast(S_ind[0], Ssizes.data());

    for (long long i = 0; i < nodes; i++) {
      // Generate far field for the upper levels
      if (localChildOffsets[i+1] <= localChildOffsets[i]) {
        continue;
      }

      // we need to count the #elements in the far field
      // we do this bu subtracting the near field elements
      long long far_elems = matgen.get_num_elems();
      std::vector<long long> FS_ind;
      // either build a factorization aka HSS basis or a regular H2 basis
      if (1. <= epi) {
        // HSS basis only excludes the self interactions
        long long ci = i + ybegin;
        far_elems -= (cells[ci].Body[1] - cells[ci].Body[0]);
        if (far_elems) {
          FS_ind.resize(far_elems);
          long long num_elems = cells[ci].Body[0];
          long long far_start = 0;
          // add all elements until the the diagonal block
          std::iota(&FS_ind[far_start], &FS_ind[far_start + num_elems], 0);
          far_start += num_elems;    
          // add the elements after the diagonal block
          num_elems = matgen.get_num_elems() - cells[ci].Body[1];
          std::iota(&FS_ind[far_start], &FS_ind[far_start + num_elems], cells[ci].Body[1]);
          // now we have all the elements of the far field
          // and create the far field matrix F
          // compute F transpose directly
          long long M = Dims[i + ibegin];
          Matrix_dt F(FS_ind.size() * 3, M);
          start = MPI_Wtime();
          matgen.gen_matrix_idx_element_single_layer(F.data(), S_ind[i + ibegin], M, FS_ind.data(), FS_ind.size(), omega);
          time_gen += MPI_Wtime() - start;
          start = MPI_Wtime();
          long long rank = compute_basis(F, epi, S_ind[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
          time_comp += MPI_Wtime() - start;
          DimsLr[i + ibegin] = rank;
        }
      } else {
        // H2 excludes the entire near field
        for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
          // near field cell
          long long cj = Near.ColIndex[ij + Near.RowIndex[ybegin]];
          far_elems -= (cells[cj].Body[1] - cells[cj].Body[0]);
        }
        if (far_elems) {
          FS_ind.resize(far_elems);
          // first near field cell (there has to be at least one)
          long long current_near = Near.ColIndex[ARows[i] + Near.RowIndex[ybegin]];
          long long num_elems = cells[current_near].Body[0];
          long long far_start = 0;
          // add all elements until the first near field cell
          std::iota(&FS_ind[far_start], &FS_ind[far_start + num_elems], 0);
          far_start += num_elems;
          // loop through the near field
          for (long long ij = ARows[i] + 1; ij < ARows[i + 1]; ij++) {
            long long next_near = Near.ColIndex[ij + Near.RowIndex[ybegin]];
            // add the elements between the two near field cells
            num_elems = cells[next_near].Body[0] - cells[current_near].Body[1];
            std::iota(&FS_ind[far_start], &FS_ind[far_start + num_elems], cells[current_near].Body[1]);
            far_start += num_elems;
            current_near = next_near;
          }
          // add the elements between the last near field cell and the end of the far field
          num_elems = matgen.get_num_elems() - cells[current_near].Body[1];
          std::iota(&FS_ind[far_start], &FS_ind[far_start + num_elems], cells[current_near].Body[1]);
          // now we have all the elements of the far field
          // and create the far field matrix F
          // compute F transpose directly
          long long M = Dims[i + ibegin];
          Matrix_dt F(FS_ind.size() * 3, M);
          start = MPI_Wtime();
          matgen.gen_matrix_idx_element_single_layer(F.data(), S_ind[i + ibegin], M, FS_ind.data(), FS_ind.size(), omega);
          time_gen += MPI_Wtime() - start;
          start = MPI_Wtime();
          long long rank = compute_basis(F, epi, S_ind[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
          time_comp += MPI_Wtime() - start;
          DimsLr[i + ibegin] = rank;
        }  
      }
    }

    comm.dataSizesToNeighborOffsets(Qsizes.data());
    comm.neighbor_bcast(DimsLr.data(), neighbor_ones.data());
    // we need to communicate S again because the order has changed in the above
    // compute_basis() call
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

    for (long long i = 0; i < nodes; i++) {
      long long y = i + ibegin;
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
          start = MPI_Wtime();
          matgen.gen_matrix_element_single_layer(Ayx.data(), S_ind[y], M, S_ind[x], N, omega);
          time_gen += MPI_Wtime() - start;
          Cyx.noalias() = Ry.template triangularView<Eigen::Upper>() * Ayx * Rx.transpose().template triangularView<Eigen::Lower>();
        }
        else {
          start = MPI_Wtime();
          matgen.gen_matrix_element_single_layer(Cyx.data(), S_ind[y], M, S_ind[x], N, omega);
          time_gen += MPI_Wtime() - start;
        }
      }
    }
  }
  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (verbose && mpi_rank == 0) {
    std::cout<<"Matgen time (Rank 0): "<<time_gen<<std::endl;
    std::cout<<"Compress time (Rank 0): "<<time_comp<<std::endl;
  }

  NbXoffsets.insert(NbXoffsets.begin(), Dims.begin(), Dims.end());
  NbXoffsets.erase(NbXoffsets.begin() + comm.dataSizesToNeighborOffsets(NbXoffsets.data()), NbXoffsets.end());
  NbZoffsets.insert(NbZoffsets.begin(), DimsLr.begin(), DimsLr.end());
  NbZoffsets.erase(NbZoffsets.begin() + comm.dataSizesToNeighborOffsets(NbZoffsets.data()), NbZoffsets.end());
}

// create the far field using HiDR
template <typename DT>
void H2Matrix<DT>::construct_hidr(const MatrixGenerator& matgen, double epi, const Cell cells[], const CSR& Near, const HiDR& hidr, const ColCommMPI& comm, H2Matrix& lowerA, const ColCommMPI& lowerComm, const double omega, bool verbose) {
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
  UpperStride.resize(nodes, 0);

  // get the Nearfield indices CSR
  ARows.insert(ARows.begin(), comm.ARowOffsets.begin(), comm.ARowOffsets.end());
  ACols.insert(ACols.begin(), comm.AColumns.begin(), comm.AColumns.end());
  // get the far-field indices CSR
  CRows.insert(CRows.begin(), comm.CRowOffsets.begin(), comm.CRowOffsets.end());
  CCols.insert(CCols.begin(), comm.CColumns.begin(), comm.CColumns.end());
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
    // not actually needed with HiDR
    n_mat = std::reduce(&Dims[ibegin], &Dims[ibegin + nodes], 0ll);
  }
  else {
    // only for leaf level
    // Dims stores the number of particels for each cell (multiplied by 3)
    std::transform(&cells[ybegin], &cells[ybegin + nodes], &Dims[ibegin], [](const Cell& c) { return (c.Body[1] - c.Body[0]) * 3; });
    // total number of particles on this process
    lenX = std::reduce(&Dims[ibegin], &Dims[ibegin + nodes]);
    LowerZ = 0;
    // not actually needed with HiDR
    n_mat = matgen.get_num_elems() * 3;
  }
  double time_gen = 0, time_comp = 0, start;

  std::vector<long long> neighbor_ones(xlen, 1ll);
  comm.dataSizesToNeighborOffsets(neighbor_ones.data());
  comm.neighbor_bcast(Dims.data(), neighbor_ones.data());
  X.alloc(xlen, Dims.data());
  Y.alloc(xlen, Dims.data());
  // S_ind stores the indices of the elements
  std::vector<long long> Ssizes(Dims);
  S_ind.alloc(xlen, Ssizes.data());

  std::vector<long long> Qsizes(xlen, 0);
  std::transform(Dims.begin(), Dims.end(), Qsizes.begin(), [](const long long d) { return d * d; });
  Q.alloc(xlen, Qsizes.data());
  R.alloc(xlen, Qsizes.data());

  // near field blocks (i.e. dense matrices on the leaf level), not necessarily square
  std::vector<long long> Asizes(ARows[nodes]);
  for (long long i = 0; i < nodes; i++)
    std::transform(ACols.begin() + ARows[i], ACols.begin() + ARows[i + 1], Asizes.begin() + ARows[i],
      [&](long long col) { return Dims[i + ibegin] * Dims[col]; });
  A.alloc(ARows[nodes], Asizes.data());

  typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;
  typedef Eigen::Stride<Eigen::Dynamic, 1> Stride_t;
  typedef Eigen::Map<Matrix_dt, Eigen::Unaligned, Stride_t> MatrixMap_dt;


  // skip blocks that contain no points (because they are split further)
  if (std::reduce(Dims.begin(), Dims.end())) {
    // index of the first cell for this process on this level
    long long pbegin = lowerComm.oLocal();
    // number of cells for this process/level
    long long pend = pbegin + lowerComm.lenLocal();

    // loop over all nodes
    for (long long i = 0; i < nodes; i++) {
      // number of rows in that cell
      long long M = Dims[i + ibegin];
      long long childi = localChildOffsets[i];
      long long cendi = localChildOffsets[i + 1];
      // get the corresponding Q matrix (as a reference)
      Eigen::Map<Matrix_dt> Qi(Q[i + ibegin], M, M);

      // for all children (i.e. only on the intermediate levels)
      for (long long y = childi; y < cendi; y++) {
        long long offset_y = std::reduce(&lowerA.DimsLr[childi], &lowerA.DimsLr[y]);
        long long ny = lowerA.DimsLr[y];
        std::copy(lowerA.S_ind[y], lowerA.S_ind[y] + (ny), &(S_ind[i + ibegin])[offset_y]);

        MatrixMap_dt Ry(lowerA.R[y], ny, ny, Stride_t(lowerA.Dims[y], 1));
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
              DT* dp = A[ij] + offset_y + offset_x * M;
              if (0 <= lowN)
                lowerA.NA[lowN] = std::distance(A[0], dp);
              else if (0 <= lowC)
                MatrixMap_dt(dp, ny, nx, Stride_t(M, 1)) = Eigen::Map<Matrix_dt>(lowerA.C[lowC], ny, nx);
            }
          }
        }
      }
       
      // leaf level (i.e. no children)
      if (cendi <= childi) {
        long long ci = i + ybegin;
        std::iota(S_ind[i + ibegin], S_ind[i + ibegin + 1], cells[ci].Body[0] * 3);
        Qi = Matrix_dt::Identity(M, M);

        const long long M_elem = M / 3;
        // generate the near field aka dense matrices in A
        for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
          long long N = Dims[ACols[ij]];
             long long cj = Near.ColIndex[ij + Near.RowIndex[ybegin]];
          Eigen::Map<Matrix_dt> A_ij(A[ij], M, N);
          start = MPI_Wtime();
          matgen.gen_matrix_sorted_single_layer(A_ij.data(), cells[ci].Body[0], M_elem, cells[cj].Body[0], N / 3, omega);
          time_gen += MPI_Wtime() - start;
        }

        // HiDR is only used for the preconditioner and thus we always employ
        // a factorization basis here
        // we cannot use ibegin here, since the tree might have already been split further up
        const long long fbodies_begin = ybegin - (1 << log2floor(ybegin + 1)) + 1;
        const long long far_cols = hidr.fbodies_size_at_i(i + fbodies_begin);
            
        // this should always be true on the leaf level
        if (far_cols > 0) {
          Matrix_dt far(M, far_cols * 3);
          start = MPI_Wtime();
          matgen.gen_matrix_hidr_sorted_single_layer(far.data(), cells[ci].Body[0], M_elem, hidr.fbodies_at_i(i + fbodies_begin), far_cols, omega); 
          time_gen += MPI_Wtime() - start;
          start = MPI_Wtime();
          long long rank = compute_basis(far.transpose(), epi, S_ind[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
          time_gen += MPI_Wtime() - start;
          DimsLr[i + ibegin] = rank;
        }
      }
    }

    comm.dataSizesToNeighborOffsets(Ssizes.data());
    comm.neighbor_bcast(S_ind[0], Ssizes.data());

    for (long long i = 0; i < nodes; i++) {
      // Generate far field for the upper levels
      if (localChildOffsets[i+1] <= localChildOffsets[i]) {
        continue;
      }
      // same as above, always construct a factorization basis
      long long M = Dims[i + ibegin];
      const long long fbodies_begin = ybegin - (1 << log2floor(ybegin + 1)) + 1;
      long long far_cols = hidr.fbodies_size_at_i(i + fbodies_begin);
            
      if (far_cols > 0) {
        // compute F transpose directly
        Matrix_dt F(far_cols * 3, M);
        start = MPI_Wtime();
        matgen.gen_matrix_idx_element_single_layer(F.data(), S_ind[i + ibegin], M, hidr.fbodies_at_i(i + fbodies_begin), far_cols, omega);
        time_gen += MPI_Wtime() - start;
        start = MPI_Wtime();
        long long rank = compute_basis(F, epi, S_ind[i + ibegin], Q[i + ibegin], R[i + ibegin], 1. <= epi);
        time_comp += MPI_Wtime() - start;
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

    for (long long i = 0; i < nodes; i++) {
      long long y = i + ibegin;
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
          start = MPI_Wtime();
          matgen.gen_matrix_element_single_layer(Ayx.data(), S_ind[y], M, S_ind[x], N, omega);
          time_gen += MPI_Wtime() - start;
          Cyx.noalias() = Ry.template triangularView<Eigen::Upper>() * Ayx * Rx.transpose().template triangularView<Eigen::Lower>();
        }
        else {
          start = MPI_Wtime();
          matgen.gen_matrix_element_single_layer(Cyx.data(), S_ind[y], M, S_ind[x], N, omega);
          time_gen += MPI_Wtime() - start;
        }
      }
    }
  }
  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (verbose && mpi_rank == 0) {
    std::cout<<"Matgen time (Rank 0): "<<time_gen<<std::endl;
    std::cout<<"Compress time (Rank 0): "<<time_comp<<std::endl;
  }

  NbXoffsets.insert(NbXoffsets.begin(), Dims.begin(), Dims.end());
  NbXoffsets.erase(NbXoffsets.begin() + comm.dataSizesToNeighborOffsets(NbXoffsets.data()), NbXoffsets.end());
  NbZoffsets.insert(NbZoffsets.begin(), DimsLr.begin(), DimsLr.end());
  NbZoffsets.erase(NbZoffsets.begin() + comm.dataSizesToNeighborOffsets(NbZoffsets.data()), NbZoffsets.end());
}

template <typename DT>
void H2Matrix<DT>::matVecUpwardPass(const DT* X_in, const ColCommMPI& comm) {
  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, 1>> Vector_dt;
  typedef Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> Matrix_dt;

  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  std::copy(&X_in[LowerZ], &X_in[LowerZ + lenX], X[ibegin]);
 
  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    long long N = DimsLr[i + ibegin];
    Vector_dt x(X[i + ibegin], M);
    if (0 < N) {
      Vector_dt z(Z[i + ibegin], N);
      Matrix_dt q(Q[i + ibegin], M, N);
      z = q.transpose() * x;
         }
  }

  comm.neighbor_bcast(Z[0], NbZoffsets.data());
}

template <typename DT>
void H2Matrix<DT>::matVecHorizontalandDownwardPass(DT* Y_out, const ColCommMPI& comm) {
  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, 1>> Vector_dt;
  typedef Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> Matrix_dt;

  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    long long K = DimsLr[i + ibegin];
    if (0 < K) {
      Vector_dt w(W[i + ibegin], K);
      for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
        long long j = CCols[ij];
        long long N = DimsLr[j];

        Vector_dt z(Z[j], N);
        Matrix_dt c(C[ij], K, N);
        w.noalias() += c * z;
      }

      Matrix_dt q(Q[i + ibegin], M, K);
      Vector_dt y(Y[i + ibegin], M);
      y.noalias() = q * w;
    }
  }

  std::copy(Y[ibegin], Y[ibegin + nodes], &Y_out[LowerZ]);
}

template <typename DT>
void H2Matrix<DT>::matVecLeafHorizontalPass(DT* X_io, const ColCommMPI& comm) {
  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, 1>> Vector_dt;
  typedef Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> Matrix_dt;

  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  std::copy(&X_io[0], &X_io[lenX], X[ibegin]);
  comm.neighbor_bcast(X[0], NbXoffsets.data());

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    long long K = DimsLr[i + ibegin];
    Vector_dt y(Y[i + ibegin], M);
    y.setZero();

    if (0 < K) {
      Vector_dt w(W[i + ibegin], K);
      for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
        long long j = CCols[ij];
        long long N = DimsLr[j];

        Vector_dt z(Z[j], N);
        Matrix_dt c(C[ij], K, N);
        w.noalias() += c * z;
      }

      Matrix_dt q(Q[i + ibegin], M, K);
      y.noalias() += q * w;
    }

    for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
      long long j = ACols[ij];
      long long N = Dims[j];

      Vector_dt x(X[j], N);
      Matrix_dt c(A[ij], M, N);
      y.noalias() += c * x;
    }
  }

  std::copy(Y[ibegin], Y[ibegin + nodes], X_io);
}

template <typename DT>
void H2Matrix<DT>::factorize(const ColCommMPI& comm) {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long xlen = comm.lenNeighbors();
  long long dims_max = *std::max_element(Dims.begin(), Dims.end());
  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> Matrix_dt;

  std::vector<long long> Bsizes(xlen);
  std::fill(Bsizes.begin(), Bsizes.end(), dims_max * dims_max);
  MatrixDataContainer<DT> B;
  B.alloc(xlen, Bsizes.data());

  if (nodes == 1)
    comm.level_merge(A[0], A.size());
  for (long long i = 0; i < nodes; i++) {
    long long diag = lookupIJ(ARows, ACols, i, i + ibegin);
    long long M = Dims[i + ibegin];
    long long Ms = DimsLr[i + ibegin];
    long long Mr = M - Ms;

    Matrix_dt Ui(Q[i + ibegin], M, M);
    Matrix_dt V(R[i + ibegin], M, M);
    Matrix_dt Aii(A[diag], M, M);
    Matrix_dt b(B[i + ibegin], dims_max, M);

    b.topRows(M).noalias() = Ui.adjoint() * Aii.transpose();
    Aii.noalias() = Ui.adjoint() * b.topRows(M).transpose();
    V.topRows(Ms) = Ui.leftCols(Ms).adjoint();

    if (0 < Mr) {
      Eigen::PartialPivLU<Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> fac(Aii.bottomRightCorner(Mr, Mr));
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

        Matrix_dt Uj(Q[j], N, N);
        Matrix_dt Aij(A[ij], M, N);

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
    Matrix_dt Aii(A[diag], M, M);

    for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++)
      if (ij != diag) {
        long long j = ACols[ij];
        long long N = Dims[j];
        long long Ns = DimsLr[j];
        long long Nr = N - Ns;
        
        Matrix_dt Aij(A[ij], M, N);
        Matrix_dt Bj(B[j], dims_max, N);
        Aij.topLeftCorner(Ms, Ns) -= Aii.topRightCorner(Ms, Mr) * Aij.bottomLeftCorner(Mr, Ns) + Aij.topRightCorner(Ms, Nr) * Bj.topLeftCorner(Nr, Ns);
        Aii.topLeftCorner(Ms, Ms) -= Aij.topRightCorner(Ms, Nr) * Bj.topRightCorner(Nr, Nr) * Aij.topRightCorner(Ms, Nr).transpose();
      }
  }
}

template <typename DT>
void H2Matrix<DT>::factorizeCopyNext(const H2Matrix& lowerA, const ColCommMPI& lowerComm) {
  long long ibegin = lowerComm.oLocal();
  long long nodes = lowerComm.lenLocal();
  typedef Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> Matrix_dt;

  for (long long i = 0; i < nodes; i++)
    for (long long ij = lowerA.ARows[i]; ij < lowerA.ARows[i + 1]; ij++) {
      long long j = lowerA.ACols[ij];
      long long M = lowerA.Dims[i + ibegin];
      long long N = lowerA.Dims[j];
      long long Ms = lowerA.DimsLr[i + ibegin];
      long long Ns = lowerA.DimsLr[j];

      Matrix_dt Aij(lowerA.A[ij], M, N);
      if (0 < Ms && 0 < Ns) {
        Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> An(A[0] + lowerA.NA[ij], Ms, Ns, Eigen::Stride<Eigen::Dynamic, 1>(lowerA.UpperStride[i], 1));
        An = Aij.topLeftCorner(Ms, Ns);
      }
    }
}

template <typename DT>
void H2Matrix<DT>::forwardSubstitute(const DT* X_in, const ColCommMPI& comm) {
  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, 1>> Vector_dt;
  typedef Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> Matrix_dt;

  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  std::copy(&X_in[LowerZ], &X_in[LowerZ + lenX], Y[ibegin]);

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];

    if (0 < M) {
      Vector_dt x(X[i + ibegin], M);
      Vector_dt y(Y[i + ibegin], M);
      Matrix_dt q(R[i + ibegin], M, M);
      x.noalias() = q * y;
    }
  }

  comm.neighbor_bcast(X[0], NbXoffsets.data());

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    long long Ms = DimsLr[i + ibegin];

    if (0 < Ms) {
      Vector_dt z(Z[i + ibegin], Ms);
      z = Vector_dt(X[i + ibegin], Ms);

      for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
        long long j = ACols[ij];
        long long N = Dims[j];
        long long Ns = DimsLr[j];
        long long Nr = N - Ns;

        if (0 < Nr) {
          Vector_dt xj(X[j], N);
          Matrix_dt Aij(A[ij], M, N);
          z.noalias() -= Aij.topRightCorner(Ms, Nr) * xj.bottomRows(Nr);
        }
      }
    }
  }

  comm.neighbor_bcast(Z[0], NbZoffsets.data());
}

template <typename DT>
void H2Matrix<DT>::backwardSubstitute(DT* Y_out, const ColCommMPI& comm) {
  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, 1>> Vector_dt;
  typedef Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> Matrix_dt;

  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  comm.neighbor_bcast(W[0], NbZoffsets.data());

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    long long Ms = DimsLr[i + ibegin];
    long long Mr = M - Ms;

    Vector_dt x(X[i + ibegin], M);
    x.topRows(Ms) = Vector_dt(W[i + ibegin], Ms);
      
    if (0 < Mr) {
      for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
        long long j = ACols[ij];
        long long N = Dims[j];
        long long Ns = DimsLr[j];

        if (0 < Ns) {
          Vector_dt wj(W[j], Ns);
          Matrix_dt Aij(A[ij], M, N);
          x.bottomRows(Mr).noalias() -= Aij.bottomLeftCorner(Mr, Ns) * wj;
        }
      }
    }
  }

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    if (0 < M) {
      Vector_dt x(X[i + ibegin], M);
      Vector_dt y(Y[i + ibegin], M);
      Matrix_dt q(Q[i + ibegin], M, M);
      y.noalias() = q.conjugate() * x;
    }
  }

  std::copy(Y[ibegin], Y[ibegin + nodes], &Y_out[LowerZ]);
}

void check_mpi_status(int error) {
  if (error != MPI_SUCCESS)
    std::cerr<<"Something went wrong in MPI/IO "<<error<<std::endl;
}

template <typename DT>
void H2Matrix<DT>::write(long long level, std::string& basename) const {
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_File fh;

  std::string filename = basename + std::to_string(level) + "_" + std::to_string(mpi_rank) + ".dat";
  MPI_File_open(MPI_COMM_SELF, filename.c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_Offset offset = 0;
  MPI_Status status;

  MPI_File_write_at(fh, offset, &lenX, 1, MPI_LONG_LONG_INT, &status);
  offset += sizeof(long long);
  MPI_File_write_at(fh, offset, &LowerZ, 1, MPI_LONG_LONG_INT, &status);
  offset += sizeof(long long);
  MPI_File_write_at(fh, offset, &n_mat, 1, MPI_LONG_LONG_INT, &status);
  offset += sizeof(long long);

  long long size = Dims.size();
  MPI_File_write_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status);
  offset += sizeof(long long);
  MPI_File_write_at(fh, offset, Dims.data(), size, MPI_LONG_LONG_INT, &status);
  offset += size * sizeof(long long);
  size = DimsLr.size();
  MPI_File_write_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status);
  offset += sizeof(long long);
  MPI_File_write_at(fh, offset, DimsLr.data(), size, MPI_LONG_LONG_INT, &status);
  offset += size * sizeof(long long);

  size = ARows.size();
  MPI_File_write_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status);
  offset += sizeof(long long);
  MPI_File_write_at(fh, offset, ARows.data(), size, MPI_LONG_LONG_INT, &status);
  offset += size * sizeof(long long);
  size = ACols.size();
  MPI_File_write_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status);
  offset += sizeof(long long);
  MPI_File_write_at(fh, offset, ACols.data(), size, MPI_LONG_LONG_INT, &status);
  offset += size * sizeof(long long);
  size = CRows.size();
  MPI_File_write_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status);
  offset += sizeof(long long);
  MPI_File_write_at(fh, offset, CRows.data(), size, MPI_LONG_LONG_INT, &status);
  offset += size * sizeof(long long);
  size = CCols.size();
  MPI_File_write_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status);
  offset += sizeof(long long);
  MPI_File_write_at(fh, offset, CCols.data(), size, MPI_LONG_LONG_INT, &status);
  offset += size * sizeof(long long);

  size = NA.size();
  MPI_File_write_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status);
  offset += sizeof(long long);
  MPI_File_write_at(fh, offset, NA.data(), size, MPI_LONG_LONG_INT, &status);
  offset += size * sizeof(long long);
  size = NbXoffsets.size();
  MPI_File_write_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status);
  offset += sizeof(long long);
  MPI_File_write_at(fh, offset, NbXoffsets.data(), size, MPI_LONG_LONG_INT, &status);
  offset += size * sizeof(long long);
  size = NbZoffsets.size();
  MPI_File_write_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status);
  offset += sizeof(long long);
  MPI_File_write_at(fh, offset, NbZoffsets.data(), size, MPI_LONG_LONG_INT, &status);
  offset += size * sizeof(long long);
  size = UpperStride.size();
  MPI_File_write_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status);
  offset += sizeof(long long);
  MPI_File_write_at(fh, offset, UpperStride.data(), size, MPI_LONG_LONG_INT, &status);
  offset += size * sizeof(long long);

  S_ind.write(fh, offset, status);
  Q.write(fh, offset, status);
  R.write(fh, offset, status);
  A.write(fh, offset, status);
  C.write(fh, offset, status);
  U.write(fh, offset, status);

  X.write(fh, offset, status);
  Y.write(fh, offset, status);
  Z.write(fh, offset, status);
  W.write(fh, offset, status);

  MPI_File_close(&fh);
}

template <typename DT>
bool H2Matrix<DT>::read(long long level, std::string& basename) {
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_File fh;
  std::string filename = basename + std::to_string(level) + "_" + std::to_string(mpi_rank) + ".dat";
  if (MPI_File_open(MPI_COMM_SELF, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
    return false;
  }
  MPI_Offset offset = 0;
  MPI_Status status;

  check_mpi_status(MPI_File_read_at(fh, offset, &lenX, 1, MPI_LONG_LONG_INT, &status));
  offset += sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, &LowerZ, 1, MPI_LONG_LONG_INT, &status));
  offset += sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, &n_mat, 1, MPI_LONG_LONG_INT, &status));
  offset += sizeof(long long);

  long long size;
  check_mpi_status(MPI_File_read_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status));
  Dims.resize(size);
  offset += sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, Dims.data(), size, MPI_LONG_LONG_INT, &status));
  offset += size * sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status));
  DimsLr.resize(size);
  offset += sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, DimsLr.data(), size, MPI_LONG_LONG_INT, &status));
  offset += size * sizeof(long long);

  check_mpi_status(MPI_File_read_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status));
  ARows.resize(size);
  offset += sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, ARows.data(), size, MPI_LONG_LONG_INT, &status));
  offset += size * sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status));
  ACols.resize(size);
  offset += sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, ACols.data(), size, MPI_LONG_LONG_INT, &status));
  offset += size * sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status));
  CRows.resize(size);
  offset += sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, CRows.data(), size, MPI_LONG_LONG_INT, &status));
  offset += size * sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status));
  CCols.resize(size);
  offset += sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, CCols.data(), size, MPI_LONG_LONG_INT, &status));
  offset += size * sizeof(long long);

  check_mpi_status(MPI_File_read_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status));
  NA.resize(size);
  offset += sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, NA.data(), size, MPI_LONG_LONG_INT, &status));
  offset += size * sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status));
  NbXoffsets.resize(size);
  offset += sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, NbXoffsets.data(), size, MPI_LONG_LONG_INT, &status));
  offset += size * sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status));
  NbZoffsets.resize(size);
  offset += sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, NbZoffsets.data(), size, MPI_LONG_LONG_INT, &status));
  offset += size * sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status));
  UpperStride.resize(size);
  offset += sizeof(long long);
  check_mpi_status(MPI_File_read_at(fh, offset, UpperStride.data(), size, MPI_LONG_LONG_INT, &status));
  offset += size * sizeof(long long);

  S_ind.read(fh, offset, status);
  Q.read(fh, offset, status);
  R.read(fh, offset, status);
  A.read(fh, offset, status);
  C.read(fh, offset, status);
  U.read(fh, offset, status);

  X.read(fh, offset, status);
  Y.read(fh, offset, status);
  Z.read(fh, offset, status);
  W.read(fh, offset, status);
 
  MPI_File_close(&fh);
  return true;
}
