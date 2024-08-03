#include <h2matrix.hpp>
#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <kernel.hpp>

#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>

#include <iostream>

WellSeparatedApproximation::WellSeparatedApproximation(const MatrixAccessor& kernel, double epsilon, long long rank, long long cell_begin, long long ncells, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation& upper_level) :
  lbegin(cell_begin), lend(cell_begin + ncells), M(ncells) {
  // loop over the cells in the upper level
  for (long long i = upper_level.lbegin; i < upper_level.lend; i++)
    // collect the far field points from the upper level
    for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; c++)
      if (lbegin <= c && c < lend){
        // c - lbegin converts the global cell index to a local one
        M[c - lbegin] = std::vector<double>(upper_level.M[i - upper_level.lbegin].begin(), upper_level.M[i - upper_level.lbegin].end());
      }
  // loop over all cells on the current level
  for (long long c = lbegin; c < lend; c++) {
    // for each cell in the far field
    for (long long i = Far.RowIndex[c]; i < Far.RowIndex[c + 1]; i++) {
      long long j = Far.ColIndex[i];
      long long nrows = cells[c].Body[1] - cells[c].Body[0];
      const double* row_bodies = &bodies[3 * cells[c].Body[0]];
      long long ncols = cells[j].Body[1] - cells[j].Body[0];
      const double* col_bodies = &bodies[3 * cells[j].Body[0]];
      
      long long max_rank = std::min(rank, std::min(nrows, ncols));
      // sample the bodies, note that we only sample the columns
      std::vector<long long> piv(max_rank);
      long long iters = adaptive_cross_approximation(kernel, epsilon, max_rank, nrows, ncols, row_bodies, col_bodies, nullptr, &piv[0]);
      // resize to the actual rank (could be less than max_rank)
      piv.resize(iters);

      Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic>> col_bodies_map(col_bodies, 3, ncols);
      Eigen::VectorXd sampled_bodies(3 * iters);

      sampled_bodies = col_bodies_map(Eigen::all, piv).reshaped();
      // add the sampled bodies to the far field of the current cell
      M[c - lbegin].insert(M[c - lbegin].end(), sampled_bodies.begin(), sampled_bodies.end());
    }
  }
}

long long WellSeparatedApproximation::fbodies_size_at_i(const long long i) const {
  return 0 <= i && i < (long long)M.size() ? M[i].size() / 3 : 0;
}

const double* WellSeparatedApproximation::fbodies_at_i(const long long i) const {
  return 0 <= i && i < (long long)M.size() ? M[i].data() : nullptr;
}

/*
In:
  kernel: kernel function
  epsilon: accuracy threshold for rank revealing QR
  nrows: number of rows (i.e. the current cell), also a and c are square matrices of nrows x nrows
  ncols: number of columns (in the far field)
  col_bodies: the far field points (column points)
Inout:
  row_bodies: the points in this cell (row points - S in the H2matrix class)
  Q: input: identity for leaf nodes, R matrix from the lower level otherwise
     output: Q matrix for the the current level in the H2matrix class
Out:
  R: the R matrix in the H2matrix class
Returns:
  rank: The rank corresponding to epsilon from the column pivoted QR
*/
long long compute_basis(const MatrixAccessor& kernel, const double epsilon, const long long nrows, const long long ncols, double row_bodies[], const double col_bodies[], std::complex<double> Q[], std::complex<double> R[]) {
  const long long MIN_D = std::min(nrows, ncols);
  long long rank = 0;
  // safeguard against empty matrix
  if (0 < MIN_D) {
    // RX is the transpose of the far field
    // We need the transpose because we compute a row ID
    Eigen::MatrixXcd RX = Eigen::MatrixXcd::Zero(MIN_D, nrows);

    if (MIN_D < ncols) {
      // N is the larger dimension, so RX is M x M
      // The QR assures we actually have those dimensions
      // Note that in this case, the following rank revealing QR
      // wouldn't actually do anything if it were not for the column pivoting
      Eigen::MatrixXcd XF(ncols, nrows);
      gen_matrix(kernel, ncols, nrows, col_bodies, row_bodies, XF.data());
      Eigen::HouseholderQR<Eigen::MatrixXcd> qr(XF);
      RX = qr.matrixQR().topRows(MIN_D).triangularView<Eigen::Upper>();
    }
    else {
      // M is the largest dimension, so RX is N X M
      gen_matrix(kernel, ncols, nrows, col_bodies, row_bodies, RX.data());
    }
    
    // Rank revealing QR of the far field
    Eigen::ColPivHouseholderQR<Eigen::MatrixXcd> rrqr(RX);
    // TODO not sure what this should accomplish
    // maybe for debugging?
    rank = std::min(MIN_D, (long long)std::floor(epsilon));
    // set the accuracy threshold and
    // get the corresponding rank
    if (epsilon < 1.) {
      rrqr.setThreshold(epsilon);
      rank = rrqr.rank();
    }

    // on the leaf level a contains the identity matrix
    // R matrix from the lower level otherwise
    Eigen::Map<Eigen::MatrixXcd> Q_ref(Q, nrows, nrows), R_ref(R, nrows, nrows);
    if (0 < rank && rank < nrows) {
      // QR successful
      // We use R_ref to store the intermediate results
      // compute the row ID from the RRQR
      // see https://users.oden.utexas.edu/~pgm/Teaching/APPM5720_2016s/scribe_week07_wed.pdf
      R_ref.topRows(rank) = rrqr.matrixR().topRows(rank);
      R_ref.topLeftCorner(rank, rank).triangularView<Eigen::Upper>().solveInPlace(R_ref.topRightCorner(rank, nrows - rank));
      R_ref.topLeftCorner(rank, rank) = Eigen::MatrixXcd::Identity(rank, rank);

      // We reorder the row points stored in S accordingly
      // Supposedly because the upper levels reuse them
      // TODO confirm this
      Eigen::Map<Eigen::MatrixXd> body(row_bodies, 3, nrows);
      body = body * rrqr.colsPermutation();

      // transpose back and reorder
      // TODO why do I need to reorder?
      // We now have the row ID A = X * A(rows) and calculate the QR of X
      // because we want an orthogonal basis
      Eigen::HouseholderQR<Eigen::MatrixXcd> qr = (Q_ref.triangularView<Eigen::Upper>() * (rrqr.colsPermutation() * R_ref.topRows(rank).transpose())).householderQr();
      // A stores the Q matrix
      Q_ref = qr.householderQ();
      // C stores the R matrix, since the memory was already allocated before
      // we knew the real rank we just zero out the rest (delete intermediate results)
      R_ref = Eigen::MatrixXcd::Zero(nrows, nrows);
      R_ref.topLeftCorner(rank, rank) = qr.matrixQR().topRows(rank).triangularView<Eigen::Upper>();
    }
    else {
      // QR failed for some reason
      Q_ref = Eigen::MatrixXcd::Identity(nrows, nrows);
      R_ref = Eigen::MatrixXcd::Identity(nrows, nrows);
    }
  }
  return rank;
}

/*
Translates 2D to 1D index for example for finding the corresponding
locatiion in C from CRows and Cols
In:
  RowIndex: vector or row indices in CSR format
  ColIndex: vector of column indices in CSR format
  i: column index
  j: row index
Returns:
  desired 1D index
*/
inline long long lookupIJ(const std::vector<long long>& RowIndex, const std::vector<long long>& ColIndex, const long long i, const long long j) {
  if (i < 0 || RowIndex.size() <= (1ull + i))
    return -1;
  long long k = std::distance(ColIndex.begin(), std::find(ColIndex.begin() + RowIndex[i], ColIndex.begin() + RowIndex[i + 1], j));
  return (k < RowIndex[i + 1]) ? k : -1;
}

H2Matrix::H2Matrix(const MatrixAccessor& kernel, const double epsilon, const Cell cells[], const CSR& Near, const CSR& Far, const double bodies[], const WellSeparatedApproximation& wsa, const ColCommMPI& comm, H2Matrix& h2_lower, const ColCommMPI& lowerComm, const bool use_near_bodies) {
  // number of cells on the same level
  long long xlen = comm.lenNeighbors();
  // index of the first cell for this process on this level (always 0 for a single process)
  long long ibegin = comm.oLocal();
  // number of cells for this process on this level (same as xlen for a single process)
  long long nodes = comm.lenLocal();
  // index of the first cell for this process on this level in the global cell array (xlen-1 for a single process)
  long long ybegin = comm.oGlobal();

  // first child of first cell
  long long ychild = cells[ybegin].Child[0];
  // convert to local index
  long long localChildIndex = lowerComm.iLocal(ychild);

  // stores the offsets to the first child of each cell on this level
  // local means it starts from 0 (i.e. it does not include the level offset)
  // Usually a sequence of 0, 2, 4, etc. (all 0s for the leaf level)
  std::vector<long long> localChildOffsets(nodes + 1);
  localChildOffsets[0] = 0;
  std::transform(&cells[ybegin], &cells[ybegin + nodes], localChildOffsets.begin() + 1, [=](const Cell& c) { return c.Child[1] - ychild; });

  // initalize vectors to 0
  Dims = std::vector<long long>(xlen, 0);
  DimsLr = std::vector<long long>(xlen, 0);
  UpperStride = std::vector<long long>(nodes, 0);

  // extract the indices of the near field for this level
  ARows = std::vector<long long>(Near.RowIndex.begin() + ybegin, Near.RowIndex.begin() + ybegin + nodes + 1);
  ACols = std::vector<long long>(Near.ColIndex.begin() + ARows[0], Near.ColIndex.begin() + ARows[nodes]);
  // transform to entries local for this level
  long long offset = ARows[0];
  std::for_each(ARows.begin(), ARows.end(), [=](long long& i) { i = i - offset; });
  std::for_each(ACols.begin(), ACols.end(), [&](long long& col) { col = comm.iLocal(col); });
  NA = std::vector<std::complex<double>*>(ARows[nodes], nullptr);

  // extract the indices of the far field (sampled bodies) for this level
  CRows = std::vector<long long>(Far.RowIndex.begin() + ybegin, Far.RowIndex.begin() + ybegin + nodes + 1);
  CCols = std::vector<long long>(Far.ColIndex.begin() + CRows[0], Far.ColIndex.begin() + CRows[nodes]);
  // transform to entries local for this level
  offset = CRows[0];
  std::for_each(CRows.begin(), CRows.end(), [=](long long& i) { i = i - offset; });
  std::for_each(CCols.begin(), CCols.end(), [&](long long& col) { col = comm.iLocal(col); });
  C = std::vector<std::complex<double>*>(CRows[nodes], nullptr);

  // get the number of points for each cell
  if (localChildOffsets.back() == 0){
    // leaf level case (i.e. there is no lower level)
    // number of points contained in each cell
    std::transform(&cells[ybegin], &cells[ybegin + nodes], &Dims[ibegin], [](const Cell& c) { return c.Body[1] - c.Body[0]; });
  } else {
    // get the low-rank dimensions from the lower level,
    // i.e. add the ranks of the (two) children.
    std::vector<long long>::const_iterator iter = h2_lower.DimsLr.begin() + localChildIndex;
    std::transform(localChildOffsets.begin(), localChildOffsets.begin() + nodes, localChildOffsets.begin() + 1, &Dims[ibegin],
      [&](long long start, long long end) { return std::reduce(iter + start, iter + end); });
  } 

  // cast the dimensions to all processes (on the same level)
  comm.neighbor_bcast(Dims.data());
  // allocates storage for the total number of points on this level
  // i.e. sum of points per cell
  X = MatrixDataContainer<std::complex<double>>(xlen, Dims.data());
  Y = MatrixDataContainer<std::complex<double>>(xlen, Dims.data());
  // pointers to X/Y on the parent cell
  // i.e. if a parent has two children with 128 elements each
  // the corresponding pointers for the first block would be set 
  // to NX[0] -> X[0], NX[1] -> X[128]
  NX = std::vector<std::complex<double>*>(xlen, nullptr);
  NY = std::vector<std::complex<double>*>(xlen, nullptr);

  // basically sets the NX, NY pointers of the children (i.e. lower level)
  // for every node on this level
  for (long long i = 0; i < xlen; i++) {
    long long ci = comm.iGlobal(i);
    long long child = lowerComm.iLocal(cells[ci].Child[0]);
    long long cend = 0 <= child ? (child + cells[ci].Child[1] - cells[ci].Child[0]) : -1;
    // for all children of that node
    for (long long y = child; y < cend; y++) {
      long long offset_y = std::reduce(&h2_lower.DimsLr[child], &h2_lower.DimsLr[y]);
      h2_lower.NX[y] = X[i] + offset_y;
      h2_lower.NY[y] = Y[i] + offset_y;
    }
  }

  // Q stores the basis (as a matrix) for each cell
  // Each Q is a square matrix of size Dims x Dims
  // note that the storage is 0 initialized
  std::vector<long long> Qsizes(xlen, 0);
  std::transform(Dims.begin(), Dims.end(), Qsizes.begin(), [](const long long d) { return d * d; });
  Q = MatrixDataContainer<std::complex<double>>(xlen, Qsizes.data());
  // stores the corresponding R from the QR factorization of the basis
  R = MatrixDataContainer<std::complex<double>>(xlen, Qsizes.data());

  // S stores the (rank) points contained within each cell
  // the dimensions are given by 3 * Dims
  std::vector<long long> Ssizes(xlen);
  std::transform(Dims.begin(), Dims.end(), Ssizes.begin(), [](const long long d) { return 3 * d; });
  S = MatrixDataContainer<double>(xlen, Ssizes.data());

  // on the leaf level A stores the near field matrices
  // on the intermediate level, A stores the skeleton matrices
  // from the lower level
  // Create a storage container for the near field (dense matrices?)
  // TODO cannot be the near field (dense matrices only exist on the lowest level)
  // for the lowest level, it seems to be the near field (dense matrices) indeed
  // but I don't know what is stored on the upper levels
  // this forms the cross product of cells, i.e. block cluster tree I x J
  std::vector<long long> Asizes(ARows[nodes]);
  for (long long i = 0; i < nodes; i++)
    std::transform(ACols.begin() + ARows[i], ACols.begin() + ARows[i + 1], Asizes.begin() + ARows[i],
      [&](long long col) { return Dims[i + ibegin] * Dims[col]; });
  A = MatrixDataContainer<std::complex<double>>(ARows[nodes], Asizes.data());
  // the lenght of this vector is the total number of points stores for all S on this level
  // within the constructor this is only allocated, but not used
  Ipivots = std::vector<int>(std::reduce(Dims.begin() + ibegin, Dims.begin() + (ibegin + nodes)));

  // skips levels that contain no points
  // e.g. if there are no low-rank blocks on this level (because they are split further)
  if (std::reduce(Dims.begin(), Dims.end())) {
    typedef Eigen::Stride<Eigen::Dynamic, 1> Stride_t;
    typedef Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Stride_t> Matrix_t; 
    // index of the first cell for this process on this level (always 0 for a single process)
    // same as ibegin
    long long pbegin = lowerComm.oLocal();
    // number of cells for this process on this level (same as xlen for a single process)
    // same as nodes
    long long pend = pbegin + lowerComm.lenLocal();

    // for every node/cell on the current level
    for (long long i = 0; i < nodes; i++) {
      // get the number of points in that cell
      const long long nrows = Dims[i + ibegin];
      // get the child cell indices
      long long childi_start = localChildIndex + localChildOffsets[i];
      long long childi_end = localChildIndex + localChildOffsets[i + 1];
      // get the corresponding Q matrix for this cell
      // note that this is a reference
      Eigen::Map<Eigen::MatrixXcd> Qi_ref(Q[i + ibegin], nrows, nrows);

      // for all children (if there are any, so only for intermediate levels)
      for (long long ci = childi_start; ci < childi_end; ci++) { 
        // get the number of points from the child
        long long offset_i = std::reduce(&h2_lower.DimsLr[childi_start], &h2_lower.DimsLr[ci]);
        long long ni = h2_lower.DimsLr[ci];
        // combine the points from the children into S
        std::copy(h2_lower.S[ci], h2_lower.S[ci] + (ni * 3), &(S[i + ibegin])[offset_i * 3]);

        // get the R matrix from the child
        Matrix_t Ri(h2_lower.R[ci], ni, ni, Stride_t(h2_lower.Dims[ci], 1));
        // assemble the R matrices into Q (used only during the basis computation)
        // Basically         |R1 0 |
        //           Qi_ref =|0  R2|
        // so Qi is a block diagonal matrix
        Qi_ref.block(offset_i, offset_i, ni, ni) = Ri;

        // Check if the child is actually on this process
        // irreleveant for a single process
        if (pbegin <= ci && ci < pend && 0 < nrows) {
          // on a single process pbegin will always be 0
          long long pi = ci - pbegin;
          // set the upper stride for each child
          h2_lower.UpperStride[pi] = nrows;

          // for all cells in the near field
          // remember, this is at the intermediate level, so the near field is hierarhical
          // (all cells are split until the leaf level)
          for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
            // get the indices of children of the near field
            const long long j_global = Near.ColIndex[ij + Near.RowIndex[ybegin]];
            const long long childj_start = lowerComm.iLocal(cells[j_global].Child[0]);
            const long long childj_end = (0 <= childj_start) ? (childj_start + cells[j_global].Child[1] - cells[j_global].Child[0]) : -1;

            // for all child nodes
            for (long long childj = childj_start; childj < childj_end; childj++) {
              // the the number of points from the child
              const long long offset_j = std::reduce(&h2_lower.DimsLr[childj_start], &h2_lower.DimsLr[childj]);
              const long long nj = h2_lower.DimsLr[childj];
              const long long low_near_idx = lookupIJ(h2_lower.ARows, h2_lower.ACols, pi, childj);
              const long long low_far_idx = lookupIJ(h2_lower.CRows, h2_lower.CCols, pi, childj);
              // this assembles the skeleton matrix stored in A from the children
              // i.e. it is a 2 x 2 block matrix (since we are looping over I x J now)
              // get the pointer to the corresponding block in A
              std::complex<double>* const A_ptr = A[ij] + offset_i + offset_j * nrows;
              // set the pointer for the children
              if (0 <= low_near_idx)
                // TODO is this ever used? It just points to zeroes I think
                h2_lower.NA[low_near_idx] = A_ptr;
              if (0 <= low_far_idx) {
                h2_lower.C[low_far_idx] = A_ptr;
                // construct R from the child
                Matrix_t Rj(h2_lower.R[childj], nj, nj, Stride_t(h2_lower.Dims[childj], 1));
                // this is a reference to the corresponding block in A
                Matrix_t A_ref(A_ptr, ni, nj, Stride_t(nrows, 1));
                // generate the rank1 x rank2 matrix corresponding to i x y
                Eigen::MatrixXcd Aij(ni, nj);
                gen_matrix(kernel, ni, nj, h2_lower.S[ci], h2_lower.S[childj], Aij.data());
                // compute the skeleton matrix R A(rows, cols) R^T
                A_ref.noalias() = Ri.triangularView<Eigen::Upper>() * Aij * Rj.transpose().triangularView<Eigen::Lower>();
              }
            }
          }
        }
      } // loop over children finished

      if (childi_end <= childi_start) { 
        // Leaf level (i.e. no children)
        // global cell array index
        const long long ci = i + ybegin;
        // copy all the particles in that cell to S
        std::copy(&bodies[3 * cells[ci].Body[0]], &bodies[3 * cells[ci].Body[1]], S[i + ibegin]);
        // set the corresponding Q matrix to the identity
        Qi_ref = Eigen::MatrixXcd::Identity(nrows, nrows);
        // ARows and AColumns contain the near field
        // so this creates the dense matrices and stores them in A
        for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
          // M and ncols are the box dimensions
          //.at the leaf level this is always leaf_size x leaf_size
          //.TODO maybe this accounts for overall non-square dimensions
          const long long ncols = Dims[ACols[ij]];
          const long long cj = Near.ColIndex[ij + Near.RowIndex[ybegin]];
          gen_matrix(kernel, nrows, ncols, &bodies[3 * cells[ci].Body[0]], &bodies[3 * cells[cj].Body[0]], A[ij]);
        }
      }
    } // loop over nodes finished
    
    // We broadcast all the particles
    // not entirely sure why, were they not created on every node?
    comm.neighbor_bcast(S);
    std::vector<std::vector<double>> cbodies(nodes);
    // loop over all nodes
    for (long long i = 0; i < nodes; i++) {
      // get the sampled  far field bodies
      const long long fsize = wsa.fbodies_size_at_i(i);
      const double* fbodies = wsa.fbodies_at_i(i);

      // TODO look into that later, for now assume it is not used (at least for the initial H2)
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
        // copies the far field points into cbodies
        cbodies[i] = std::vector<double>(3 * fsize);
        std::copy(fbodies, &fbodies[3 * fsize], cbodies[i].begin());
      }
    }

    // loop over all the nodes again
    for (long long i = 0; i < nodes; i++) {
      // we could use fbodies directly, this extra step is probably to
      // accomodate use_near_bodies
      const long long fsize = cbodies[i].size() / 3;
      const double* fbodies = cbodies[i].data();
      // compute the far field basis for each cell
      const long long rank = compute_basis(kernel, epsilon, Dims[i + ibegin], fsize, S[i + ibegin], fbodies, Q[i + ibegin], R[i + ibegin]);
      // set the low-rank dimensions
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
