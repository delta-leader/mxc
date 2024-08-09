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

// explicit template instantiation
template class WellSeparatedApproximation<std::complex<double>>;
template class H2Matrix<std::complex<double>>;
template class WellSeparatedApproximation<double>;
template class H2Matrix<double>;

template class WellSeparatedApproximation<float>;
template class H2Matrix<float>;
template H2Matrix<float>::H2Matrix(const H2Matrix<double>&);

template class WellSeparatedApproximation<std::complex<float>>;
template class H2Matrix<std::complex<float>>;
template H2Matrix<std::complex<float>>::H2Matrix(const H2Matrix<std::complex<double>>&);
template H2Matrix<std::complex<double>>::H2Matrix(const H2Matrix<std::complex<float>>&);

template <typename DT>
WellSeparatedApproximation<DT>::WellSeparatedApproximation(const MatrixAccessor<DT>& kernel, double epsilon, long long max_rank, long long cell_begin, long long ncells, const Cell cells[], const CSR& Far, const double bodies[], const WellSeparatedApproximation<DT>& upper_level, const bool fix_rank) :
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

      max_rank = fix_rank ? std::min(max_rank, std::min(nrows, ncols)) : std::min(nrows, ncols);
      
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

template <typename DT>
long long WellSeparatedApproximation<DT>::fbodies_size_at_i(const long long i) const {
  return 0 <= i && i < (long long)M.size() ? M[i].size() / 3 : 0;
}

template <typename DT>
const double* WellSeparatedApproximation<DT>::fbodies_at_i(const long long i) const {
  return 0 <= i && i < (long long)M.size() ? M[i].data() : nullptr;
}

/*
In:
  kernel: kernel function
  epsilon: accuracy threshold for rank revealing QR, or the maximum fank (fixed rank)
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
template <typename DT>
long long compute_basis(const MatrixAccessor<DT>& kernel, const double epsilon, const long long nrows, const long long ncols, double row_bodies[], const double col_bodies[], DT Q[], DT R[]) {
  typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;
  const long long MIN_D = std::min(nrows, ncols);
  long long rank = 0;
  // safeguard against empty matrix
  if (0 < MIN_D) {
    // RX is the transpose of the far field
    // We need the transpose because we compute a row ID
    Matrix_dt RX = Matrix_dt::Zero(MIN_D, nrows);

    if (MIN_D < ncols) {
      // N is the larger dimension, so RX is M x M
      // The QR assures we actually have those dimensions
      // Note that in this case, the following rank revealing QR
      // wouldn't actually do anything if it were not for the column pivoting
      Matrix_dt XF(ncols, nrows);
      gen_matrix(kernel, ncols, nrows, col_bodies, row_bodies, XF.data());
      Eigen::HouseholderQR<Matrix_dt> qr(XF);
      RX = qr.matrixQR().topRows(MIN_D).template triangularView<Eigen::Upper>();
    }
    else {
      // M is the largest dimension, so RX is N X M
      gen_matrix(kernel, ncols, nrows, col_bodies, row_bodies, RX.data());
    }
    
    // Rank revealing QR of the far field
    Eigen::ColPivHouseholderQR<Matrix_dt> rrqr(RX);
    // if epsilon contains the maximum rank, extract it here
    rank = std::min(MIN_D, (long long)std::floor(epsilon));
    // if epsilon does not contain the maximum rank
    // use it as intended
    if (epsilon < 1.) {
      // set the accuracy threshold and
      // get the corresponding rank
      rrqr.setThreshold(epsilon);
      rank = rrqr.rank();
    }

    // on the leaf level a contains the identity matrix
    // R matrix from the lower level otherwise
    Eigen::Map<Matrix_dt> Q_ref(Q, nrows, nrows), R_ref(R, nrows, nrows);
    if (0 < rank && rank < nrows) {
      // QR successful
      // We use R_ref to store the intermediate results
      // compute the row ID from the RRQR
      // see https://users.oden.utexas.edu/~pgm/Teaching/APPM5720_2016s/scribe_week07_wed.pdf
      R_ref.topRows(rank) = rrqr.matrixR().topRows(rank);
      R_ref.topLeftCorner(rank, rank).template triangularView<Eigen::Upper>().solveInPlace(R_ref.topRightCorner(rank, nrows - rank));
      R_ref.topLeftCorner(rank, rank) = Matrix_dt::Identity(rank, rank);

      // We reorder the row points stored in S accordingly
      // Supposedly because the upper levels reuse them
      // TODO confirm this
      Eigen::Map<Eigen::MatrixXd> body(row_bodies, 3, nrows);
      body = body * rrqr.colsPermutation();

      // transpose back and reorder
      // TODO why do I need to reorder?
      // We now have the row ID A = X * A(rows) and calculate the QR of X
      // because we want an orthogonal basis
      Eigen::HouseholderQR<Matrix_dt> qr = (Q_ref.template triangularView<Eigen::Upper>() * (rrqr.colsPermutation() * R_ref.topRows(rank).transpose())).householderQr();
      // A stores the Q matrix
      Q_ref = qr.householderQ();
      // C stores the R matrix, since the memory was already allocated before
      // we knew the real rank we just zero out the rest (delete intermediate results)
      R_ref = Matrix_dt::Zero(nrows, nrows);
      R_ref.topLeftCorner(rank, rank) = qr.matrixQR().topRows(rank).template triangularView<Eigen::Upper>();
    }
    else {
      // QR failed for some reason
      Q_ref = Matrix_dt::Identity(nrows, nrows);
      R_ref = Matrix_dt::Identity(nrows, nrows);
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

template <typename DT>
H2Matrix<DT>::H2Matrix(const MatrixAccessor<DT>& kernel, const double epsilon, const Cell cells[], const CSR& Near, const CSR& Far, const double bodies[], const WellSeparatedApproximation<DT>& wsa, const ColCommMPI& comm, H2Matrix& h2_lower, const ColCommMPI& lowerComm, const bool use_near_bodies) {
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
  NA = std::vector<DT*>(ARows[nodes], nullptr);

  // extract the indices of the far field (sampled bodies) for this level
  CRows = std::vector<long long>(Far.RowIndex.begin() + ybegin, Far.RowIndex.begin() + ybegin + nodes + 1);
  CCols = std::vector<long long>(Far.ColIndex.begin() + CRows[0], Far.ColIndex.begin() + CRows[nodes]);
  // transform to entries local for this level
  offset = CRows[0];
  std::for_each(CRows.begin(), CRows.end(), [=](long long& i) { i = i - offset; });
  std::for_each(CCols.begin(), CCols.end(), [&](long long& col) { col = comm.iLocal(col); });
  C = std::vector<DT*>(CRows[nodes], nullptr);

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
  X = MatrixDataContainer<DT>(xlen, Dims.data());
  Y = MatrixDataContainer<DT>(xlen, Dims.data());
  // pointers to X/Y on the parent cell
  // i.e. if a parent has two children with 128 elements each
  // the corresponding pointers for the first block would be set 
  // to NX[0] -> X[0], NX[1] -> X[128]
  NX = std::vector<DT*>(xlen, nullptr);
  NY = std::vector<DT*>(xlen, nullptr);

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
  Q = MatrixDataContainer<DT>(xlen, Qsizes.data());
  // stores the corresponding R from the QR factorization of the basis
  R = MatrixDataContainer<DT>(xlen, Qsizes.data());

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
  A = MatrixDataContainer<DT>(ARows[nodes], Asizes.data());
  // the lenght of this vector is the total number of points stores for all S on this level
  // within the constructor this is only allocated, but not used
  Ipivots = std::vector<int>(std::reduce(Dims.begin() + ibegin, Dims.begin() + (ibegin + nodes)));

  // skips levels that contain no points
  // e.g. if there are no low-rank blocks on this level (because they are split further)
  if (std::reduce(Dims.begin(), Dims.end())) {
    typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;
    typedef Eigen::Stride<Eigen::Dynamic, 1> Stride_t;
    typedef Eigen::Map<Matrix_dt, Eigen::Unaligned, Stride_t> MatrixMap_dt; 
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
      Eigen::Map<Matrix_dt> Qi_ref(Q[i + ibegin], nrows, nrows);

      // for all children (if there are any, so only for intermediate levels)
      for (long long ci = childi_start; ci < childi_end; ci++) { 
        // get the number of points from the child
        long long offset_i = std::reduce(&h2_lower.DimsLr[childi_start], &h2_lower.DimsLr[ci]);
        long long ni = h2_lower.DimsLr[ci];
        // combine the points from the children into S
        std::copy(h2_lower.S[ci], h2_lower.S[ci] + (ni * 3), &(S[i + ibegin])[offset_i * 3]);

        // get the R matrix from the child
        MatrixMap_dt Ri(h2_lower.R[ci], ni, ni, Stride_t(h2_lower.Dims[ci], 1));
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
              DT* const A_ptr = A[ij] + offset_i + offset_j * nrows;
              // set the pointer for the children
              if (0 <= low_near_idx)
                // TODO is this ever used? It just points to zeroes I think
                h2_lower.NA[low_near_idx] = A_ptr;
              if (0 <= low_far_idx) {
                h2_lower.C[low_far_idx] = A_ptr;
                // construct R from the child
                MatrixMap_dt Rj(h2_lower.R[childj], nj, nj, Stride_t(h2_lower.Dims[childj], 1));
                // this is a reference to the corresponding block in A
                MatrixMap_dt A_ref(A_ptr, ni, nj, Stride_t(nrows, 1));
                // generate the rank1 x rank2 matrix corresponding to i x y
                Matrix_dt Aij(ni, nj);
                gen_matrix(kernel, ni, nj, h2_lower.S[ci], h2_lower.S[childj], Aij.data());
                // compute the skeleton matrix R A(rows, cols) R^T
                A_ref.noalias() = Ri.template triangularView<Eigen::Upper>() * Aij * Rj.transpose().template triangularView<Eigen::Lower>();
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
        Qi_ref = Matrix_dt::Identity(nrows, nrows);
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
      // If I understand it correctly, this adds the near field points to the far field points
      // in order to build a factorization basis that incorporates the fill-ins
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

template <typename DT> template <typename OT>
H2Matrix<DT>::H2Matrix(const H2Matrix<OT>& h2matrix) : DimsLr(h2matrix.DimsLr), 
  UpperStride(h2matrix.UpperStride), Q(h2matrix.Q), R(h2matrix.R), S(h2matrix.S),
  CRows(h2matrix.CRows), CCols(h2matrix.CCols), ARows(h2matrix.ARows), ACols(h2matrix.ACols),
  A(h2matrix.A), Ipivots(h2matrix.Ipivots), Dims(h2matrix.Dims), X(h2matrix.X), Y(h2matrix.Y) {

    // those pointer all point to the upper level, so we need to set them
    // from outside
    NX = std::vector<DT*>(Dims.size(), nullptr);
    NY = std::vector<DT*>(Dims.size(), nullptr);
    long long nodes = UpperStride.size();
    NA = std::vector<DT*>(ARows[nodes], nullptr);
    C = std::vector<DT*>(CRows[nodes], nullptr);
}

template <typename DT>
void H2Matrix<DT>::matVecUpwardPass(const ColCommMPI& comm) {
  // from the lowest level to the highest level
  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, 1>> VectorMap_dt;
  typedef Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> MatrixMap_dt;

  // starting index for the current level (on this process)
  const long long ibegin = comm.oLocal();
  // number of cells on this process
  const long long nodes = comm.lenLocal();
  // TODO confirm
  // reduce and broadcast X to all processes?
  comm.level_merge(X[0], X.size());
  comm.neighbor_bcast(X);

  // for all cells on this process
  for (long long i = 0; i < nodes; i++) {
    const long long nrows = Dims[i + ibegin];
    const long long rank = DimsLr[i + ibegin];
    // skip levels without data
    if (0 < rank) {
      // multiply the vector with the row basis
      // and store it to X of the upper level
      VectorMap_dt x(X[i + ibegin], nrows);
      VectorMap_dt x_parent(NX[i + ibegin], rank);
      MatrixMap_dt q(Q[i + ibegin], nrows, rank);
      x_parent.noalias() = q.transpose() * x;
    }
  }
}

template <typename DT>
void H2Matrix<DT>::matVecHorizontalandDownwardPass(const ColCommMPI& comm) {
  // from the highest level to the lowest level
  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, 1>> VectorMap_dt;
  typedef Eigen::Stride<Eigen::Dynamic, 1> Stride_t;
  typedef Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Unaligned, Stride_t> MatrixMap_dt;

  // the first cell for this process on this level
  const long long ibegin = comm.oLocal();
  // the number of cells for this process on this level
  const long long nodes = comm.lenLocal();

  // for every cell on this process
  for (long long i = 0; i < nodes; i++) {
    const long long nrows = Dims[i + ibegin];
    const long long rank_i = DimsLr[i + ibegin];
    // skip upper levels
    // e.g. where NY is not set
    if (0 < rank_i) {
      VectorMap_dt y(Y[i + ibegin], nrows);
      VectorMap_dt y_parent(NY[i + ibegin], rank_i);

      // for all cells in the far field (can be multiple per row)
      for (long long ij = CRows[i]; ij < CRows[i + 1]; ij++) {
        const long long j = CCols[ij];
        const long long rank_j = DimsLr[j];

        VectorMap_dt x_parent(NX[j], rank_j);
        // skeleton matrix
        MatrixMap_dt c(C[ij], rank_i, rank_j, Stride_t(UpperStride[i], 1));
        // multiply with the skeleton matrix
        y_parent.noalias() += c * x_parent;
      }
      // multiply the vector with the column basis
      // and store in Y
      MatrixMap_dt q(Q[i + ibegin], nrows, rank_i, Stride_t(nrows, 1));
      y.noalias() = q * y_parent;
    }
  }
}

template <typename DT>
void H2Matrix<DT>::matVecLeafHorizontalPass(const ColCommMPI& comm) {
  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, 1>> VectorMap_dt;
  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> MatrixMap_dt;

  // the first cell for this process on this level
  const long long ibegin = comm.oLocal();
  // the number of cells for this process on this level
  const long long nodes = comm.lenLocal();

  // for all cells
  for (long long i = 0; i < nodes; i++) {
    const long long nrows = Dims[i + ibegin];
    VectorMap_dt y(Y[i + ibegin], nrows);
    
    // TODO does this ever not trigger? (doesn't seem like it)
    if (0 < nrows)
      // multiplies the near field (i.e. dense) matrices
      for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
        const long long j = ACols[ij];
        const long long ncols = Dims[j];

        VectorMap_dt x(X[j], ncols);
        MatrixMap_dt c(A[ij], nrows, ncols);
        y.noalias() += c * x;
      }
  }
}

template <typename DT>
void H2Matrix<DT>::resetX() {
  std::fill(X[0], X[0] + X.size(), DT{});
  std::fill(Y[0], Y[0] + Y.size(), DT{});
}

template <typename DT>
void H2Matrix<DT>::factorize(const ColCommMPI& comm) {
  // the first cell for this process on this level
  long long ibegin = comm.oLocal();
  // the number of cells for this process on this level
  long long nodes = comm.lenLocal();
  // the maximum dimension on this level
  long long dims_max = *std::max_element(Dims.begin(), Dims.end());

  typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;
  typedef Eigen::Map<Matrix_dt> MatrixMap_dt;
  std::vector<long long> ipiv_offsets(nodes);
  // prefix sum over the dimensions on this process
  std::exclusive_scan(Dims.begin() + ibegin, Dims.begin() + (ibegin + nodes), ipiv_offsets.begin(), 0ll);
 
  // supposedly, this merges the A matrices from all processes?
  comm.level_merge(A[0], A.size());
  // for all cells (we should be able to process each cell in parallel)
  for (long long i = 0; i < nodes; i++) {
    // index of the diagonal block
    long long diag = lookupIJ(ARows, ACols, i, i + ibegin);
    // nrows
    long long M = Dims[i + ibegin];
    // rank
    long long Ms = DimsLr[i + ibegin];
    // leftover
    long long Mr = M - Ms;

    // the basis (Q matrix) for this cell
    MatrixMap_dt Ui(Q[i + ibegin], M, M);
    // the R matrix for this cell
    MatrixMap_dt V(R[i + ibegin], M, M);
    // the diagonal block for this cell 
    MatrixMap_dt Aii(A[diag], M, M);
    // V = Q^-1 A^T (note that this overwrites R)
    V.noalias() = Ui.adjoint() * Aii.transpose();
    // Aii = Q^-1 Aii (Q^-1)^T, so we baically decompose into the skeleton matrix
    //       | Sss Ssr|
    // Aii = | Srs Srr|
    Aii.noalias() = Ui.adjoint() * V.transpose();

    // factorize the redundant part, i.e. Srr
    Eigen::PartialPivLU<Matrix_dt> plu = Aii.bottomRightCorner(Mr, Mr).lu();
    Matrix_dt b(dims_max, M);
    // this is a reference
    Eigen::Map<Eigen::VectorXi> ipiv(Ipivots.data() + ipiv_offsets[i], Mr);

    // write the factorization back into A
    Aii.bottomRightCorner(Mr, Mr) = plu.matrixLU();
    // write back the pivots
    ipiv = plu.permutationP().indices();
    // Srr^-1 Qr^-1 
    // note that V is completely overwritten (see below)
    V.bottomRows(Mr) = plu.solve(Ui.rightCols(Mr).adjoint());

    // if there is a skeleton part
    // TODO should always trigger, unless all blocks are dense
    if (0 < Ms) {
      // update the skeleton part
      Eigen::Map<Matrix_dt, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> An(NA[diag], Ms, Ms, Eigen::Stride<Eigen::Dynamic, 1>(UpperStride[i], 1));
      // Srs = Srr^-1 Srs = Urr^-1 Lrr^-1 Srs 
      Aii.bottomLeftCorner(Mr, Ms) = plu.solve(Aii.bottomLeftCorner(Mr, Ms));
      // Schur complement update (only on the upper level)
      // Srr = Srr - Ssr Urr^-1 Lrr^-1 Srs
      An.noalias() = Aii.topLeftCorner(Ms, Ms) - Aii.topRightCorner(Ms, Mr) * Aii.bottomLeftCorner(Mr, Ms);
      // write the rest of Qr^-1 into V to finish the overwrite
      V.topRows(Ms) = Ui.leftCols(Ms).adjoint();
    }

    // for all cells in the near field
    for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
      if (ij != diag) {
        long long j = ACols[ij];
        long long N = Dims[j];
        long long Ns = DimsLr[j];

        MatrixMap_dt Uj(Q[j], N, N);
        MatrixMap_dt Aij(A[ij], M, N);
        
        // Decompse Aij as Qi^-1 Aij (Qj^-1)^T
        // note that V (i.e. Qi) already contains Arr^-1 (from Aii)
        // TODO not sure, confirm this
        b.topRows(N) = Uj.adjoint() * Aij.transpose();
        Aij.noalias() = V * b.topRows(N).transpose();

        if (0 < Ms && 0 < Ns) {
          // update the skeleton matrix on the upper level
          Eigen::Map<Matrix_dt, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> An(NA[ij], Ms, Ns, Eigen::Stride<Eigen::Dynamic, 1>(UpperStride[i], 1));
          An = Aij.topLeftCorner(Ms, Ns);
        }
      }
    }
  }
}

// This part is not the same as the paper, only the basic ideas
template <typename DT>
void H2Matrix<DT>::forwardSubstitute(const ColCommMPI& comm) {
  // the first cell for this process on this level
  long long ibegin = comm.oLocal();
  // the number of cells for this process on this level
  long long nodes = comm.lenLocal();
  std::vector<long long> ipiv_offsets(nodes);
  // prefix sum over the dimensions on this process
  // stored in ipiv_offsets
  std::exclusive_scan(Dims.begin() + ibegin, Dims.begin() + (ibegin + nodes), ipiv_offsets.begin(), 0ll);

  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, 1>> VectorMap_dt;
  typedef Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> MatrixMap_dt;

  comm.level_merge(X[0], X.size());
  // The error occurs here
  std::cout<<"First Loop"<<std::endl;
  // for all cells on this level
  for (long long i = 0; i < nodes; i++) {
    // index of the diagonal block
    long long diag = lookupIJ(ARows, ACols, i, i + ibegin);
    // nrows
    long long M = Dims[i + ibegin];
    // skeleton dimension
    long long Ms = DimsLr[i + ibegin];
    // redundant dimension
    long long Mr = M - Ms;

    VectorMap_dt x(X[i + ibegin], M);
    VectorMap_dt y(Y[i + ibegin], M);
    MatrixMap_dt q(Q[i + ibegin], M, M);
    // the diagonal block
    MatrixMap_dt Aii(A[diag], M, M);
    // get the pivots for the redudandant part
    Eigen::PermutationMatrix<Eigen::Dynamic> p(Eigen::Map<Eigen::VectorXi>(Ipivots.data() + ipiv_offsets[i], Mr));
    
    // The error must be in the GeMM here
    // Y = Q^-1 X
    std::cout<<"GEMM "<<M<<std::endl;
    y.noalias() = q.adjoint() * x;
    std::cout<<"GEMM end"<<std::endl;
    // privot Yr
    y.bottomRows(Mr).applyOnTheLeft(p);
    // solve Lrr y = Yr
    Aii.bottomRightCorner(Mr, Mr).template triangularView<Eigen::UnitLower>().solveInPlace(y.bottomRows(Mr));
    // solve Urr y = y
    Aii.bottomRightCorner(Mr, Mr).template triangularView<Eigen::Upper>().solveInPlace(y.bottomRows(Mr));
    // store result in x
    x = y;
  }

  // broadcast X
  comm.neighbor_bcast(X);
  // for all cells
  std::cout<<"Second Loop"<<std::endl;
  for (long long i = 0; i < nodes; i++) {
    long long diag = lookupIJ(ARows, ACols, i, i + ibegin);
    long long M = Dims[i + ibegin];
    long long Ms = DimsLr[i + ibegin];
    long long Mr = M - Ms;

    VectorMap_dt x(X[i + ibegin], M);
    if (0 < Ms) {
      std::cout<<"PArt1"<<std::endl;
      for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
        long long j = ACols[ij];
        long long N = Dims[j];
        long long Ns = DimsLr[j];
        long long Nr = N - Ns;

        VectorMap_dt xj(X[j], N);
        MatrixMap_dt Aij(A[ij], M, N);
        x.topRows(Ms).noalias() -= Aij.topRightCorner(Ms, Nr) * xj.bottomRows(Nr);
      }
      VectorMap_dt xo(NX[i + ibegin], Ms);
      xo = x.topRows(Ms);
    }
    VectorMap_dt y(Y[i + ibegin], M);
    y = x;
    std::cout<<"PArt2"<<std::endl;
    // and here
    for (long long ij = ARows[i]; ij < diag; ij++) {
      long long j = ACols[ij];
      long long N = Dims[j];
      long long Ns = DimsLr[j];
      long long Nr = N - Ns;

      VectorMap_dt xj(X[j], N);
      MatrixMap_dt Aij(A[ij], M, N);
      y.bottomRows(Mr).noalias() -= Aij.bottomRightCorner(Mr, Nr) * xj.bottomRows(Nr);
    }
  }

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    VectorMap_dt x(X[i + ibegin], M);
    VectorMap_dt y(Y[i + ibegin], M);
    x = y;
  }
}

template <typename DT>
void H2Matrix<DT>::backwardSubstitute(const ColCommMPI& comm) {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();

  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, 1>> VectorMap_dt;
  typedef Eigen::Map<const Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>> MatrixMap_dt;

  comm.neighbor_bcast(X);
  for (long long i = 0; i < nodes; i++) {
    long long diag = lookupIJ(ARows, ACols, i, i + ibegin);
    long long M = Dims[i + ibegin];
    long long Ms = DimsLr[i + ibegin];
    long long Mr = M - Ms;

    VectorMap_dt x(X[i + ibegin], M);
    VectorMap_dt y(Y[i + ibegin], M);

    y.bottomRows(Mr) = x.bottomRows(Mr);
    if (0 < Ms) {
      VectorMap_dt xo(NX[i + ibegin], Ms);
      y.topRows(Ms) = xo;
    }

    for (long long ij = diag + 1; ij < ARows[i + 1]; ij++) {
      long long j = ACols[ij];
      long long N = Dims[j];
      long long Ns = DimsLr[j];
      long long Nr = N - Ns;

      VectorMap_dt xj(X[j], N);
      MatrixMap_dt Aij(A[ij], M, N);
      y.bottomRows(Mr).noalias() -= Aij.bottomRightCorner(Mr, Nr) * xj.bottomRows(Nr);
    }

    for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
      long long j = ACols[ij];
      long long N = Dims[j];
      long long Ns = DimsLr[j];

      if (0 < Ns) {
        VectorMap_dt xo(NX[j], Ns);
        MatrixMap_dt Aij(A[ij], M, N);
        y.bottomRows(Mr).noalias() -= Aij.bottomLeftCorner(Mr, Ns) * xo;
      }
    }
  }

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    VectorMap_dt x(X[i + ibegin], M);
    VectorMap_dt y(Y[i + ibegin], M);
    MatrixMap_dt q(Q[i + ibegin], M, M);
    x.noalias() = q.conjugate() * y;
  }

  comm.neighbor_bcast(X);
}
