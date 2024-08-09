#include <solver.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>

// explicit template instantiation
template class H2MatrixSolver<std::complex<double>>;
template class H2MatrixSolver<std::complex<float>>;
template class H2MatrixSolver<double>;
template class H2MatrixSolver<float>;
template double computeRelErr<double>(const long long, const std::complex<double> X[], const std::complex<double> ref[], MPI_Comm);
template double computeRelErr<double>(const long long, const double X[], const double ref[], MPI_Comm);

template H2MatrixSolver<float>::H2MatrixSolver(const H2MatrixSolver<double>&);
template double computeRelErr<float>(const long long, const float X[], const float ref[], MPI_Comm);
template H2MatrixSolver<std::complex<float>>::H2MatrixSolver(const H2MatrixSolver<std::complex<double>>&);
template double computeRelErr<float>(const long long, const std::complex<float> X[], const std::complex<float> ref[], MPI_Comm);

// for converting the LU factors back (GMRESIR)
template H2MatrixSolver<std::complex<double>>::H2MatrixSolver(const H2MatrixSolver<std::complex<float>>&);

template <typename DT>
H2MatrixSolver<DT>::H2MatrixSolver() : levels(-1), A(), comm(), allocedComm(), local_bodies(0, 0) {
}

template <typename DT>
H2MatrixSolver<DT>::H2MatrixSolver(const MatrixAccessor<DT>& kernel, double epsilon, const long long max_rank, const std::vector<Cell>& cells, const double theta, const double bodies[], const long long max_level, const bool fix_rank, const bool factorization_basis, MPI_Comm world) : 
  levels(max_level), A(max_level + 1), comm(max_level + 1), allocedComm(), local_bodies(0, 0) {
  
  // stores the indices of the cells in the near field for each cell
  CSR Near('N', cells, theta);
  // stores the indices of the cell in the far field for each cell
  CSR Far('F', cells, theta);
  // stores the indices of all cells on the same level (i.e. near and far field)
  CSR Neighbor(Near, Far);
  int mpi_size = 1;
  MPI_Comm_size(world, &mpi_size);

  // TODO skip this for now, as I will focus on a single process
  // is this the mapping of mpi processes to cells?
  std::vector<std::pair<long long, long long>> mapping(mpi_size, std::make_pair(0, 1));
  // this seems to be the cluster tree again, but each node contains the two children of that cell
  std::vector<std::pair<long long, long long>> tree(cells.size());
  std::transform(cells.begin(), cells.end(), tree.begin(), [](const Cell& c) { return std::make_pair(c.Child[0], c.Child[1]); });
  
  // create a communicator for each level
  for (long long i = 0; i <= levels; i++)
    comm[i] = ColCommMPI(&tree[0], &mapping[0], Neighbor.RowIndex.data(), Neighbor.ColIndex.data(), allocedComm, world);

  // sample the far field for all cells in each level
  std::vector<WellSeparatedApproximation<DT>> wsa(levels + 1);
  for (long long l = 1; l <= levels; l++) {
    // edited so that WSA will always try to match epsilon, unless fix_rank is specified
    wsa[l] = WellSeparatedApproximation(kernel, epsilon, max_rank, comm[l].oGlobal(), comm[l].lenLocal(), cells.data(), Far, bodies, wsa[l - 1], fix_rank);
  }
  // this is an ugly fix to pack the max_rank into epsilon
  epsilon = fix_rank ? (double)max_rank : epsilon;
  // Create an H2 matrix for each level
  A[levels] = H2Matrix(kernel, epsilon, cells.data(), Near, Far, bodies, wsa[levels], comm[levels], A[levels], comm[levels], factorization_basis);
  for (long long l = levels - 1; l >= 0; l--){
    A[l] = H2Matrix(kernel, epsilon, cells.data(), Near, Far, bodies, wsa[l], comm[l], A[l + 1], comm[l + 1], fix_rank);
  }
  // the bodies local to each process
  // TODO confirm if this only references the S or the bodies array
  long long llen = comm[levels].lenLocal();
  long long gbegin = comm[levels].oGlobal();
  local_bodies = std::make_pair(cells[gbegin].Body[0], cells[gbegin + llen - 1].Body[1]);
}

template <typename DT> template <typename OT>
H2MatrixSolver<DT>::H2MatrixSolver(const H2MatrixSolver<OT>& solver) :
  levels(solver.levels), A(solver.levels + 1), local_bodies(solver.local_bodies) {
  // could have a weaker admissibility
  // could have a larger epsilon
  // could have a different data type
  // should use a fixed rank

  // this should duplicate all the allocated communicators
  for (size_t i = 0; i < solver.allocedComm.size(); ++i) {
    MPI_Comm mpi_comm = MPI_COMM_NULL;
    MPI_Comm_dup(solver.allocedComm[i], &mpi_comm);
    allocedComm.emplace_back(mpi_comm);
  }
  for (size_t i = 0; i < solver.comm.size(); ++i) {
    comm.emplace_back(ColCommMPI(solver.comm[i], allocedComm));
  }
  for (size_t i = 0; i < A.size(); ++i) {
    A[i] = H2Matrix<DT>(solver.A[i]);
  }
  // set the pointers to the parent level
  long long offset;
  for (size_t level = 1; level < A.size(); ++level) {
    // A matrix can be empty
    if (solver.A[level - 1].A.size() > 0) {
      auto parent_begin = solver.A[level - 1].A[0];
      auto child_begin = A[level - 1].A[0];
      for (size_t i = 0; i < A[level].C.size(); ++i) {
        // need to make the ptr constant 
        const OT* ptr = solver.A[level].C[i];
        offset = std::distance(parent_begin, ptr);
        A[level].C[i] = child_begin + offset;
      }
      for (size_t i = 0; i < A[level].NA.size(); ++i) {
        const OT* ptr = solver.A[level].NA[i];
        offset = std::distance(parent_begin, ptr);
        A[level].NA[i] = child_begin + offset;
      }
    }
    for (size_t i = 0; i < A[level].NX.size(); ++i) {
      const OT* ptrx = solver.A[level].NX[i];
      offset = std::distance(solver.A[level - 1].X[0], ptrx);
      A[level].NX [i]= A[level - 1].X[0] + offset;
      const OT* ptry = solver.A[level].NY[i];
      offset = std::distance(solver.A[level - 1].Y[0], ptry);
      A[level].NY[i] = A[level - 1].Y[0] + offset;
    }
  }
}

template <typename DT>
void H2MatrixSolver<DT>::matVecMul(DT X[]) {
  // if the solver has not been initialized, return
  if (levels < 0)
    return;
  
  // on the lowest level
  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, 1>> VectorMap_dt;
  // starting index for the lowest level on this process (single process: 0)
  long long lbegin = comm[levels].oLocal();
  // number of cells for the lowest level on this process (single process: all cells)
  long long llen = comm[levels].lenLocal();
  // number of points the lowest level for this process (single process: all)
  long long lenX = std::reduce(A[levels].Dims.begin() + lbegin, A[levels].Dims.begin() + (lbegin + llen));

  // reference to X
  VectorMap_dt X_in(X, lenX);
  // reference to the lowest level X and Y
  VectorMap_dt X_leaf(A[levels].X[lbegin], lenX);
  VectorMap_dt Y_leaf(A[levels].Y[lbegin], lenX);

  // set the X and Y of all levels to 0
  for (long long l = levels; l >= 0; l--)
    A[l].resetX();
  // X_leaf initially stores the input vector
  X_leaf = X_in;
  
  // Multiply all the row bases with the vector
  for (long long l = levels; l >= 0; l--)
    A[l].matVecUpwardPass(comm[l]);
  // TODO is there any point in calling this for level 0?
  // because NX and NY would just be nullptr
  // multiply all the skeleton matrices and column bases
  for (long long l = 0; l <= levels; l++) {
    A[l].matVecHorizontalandDownwardPass(comm[l]);
  }
  // multiply the leaf level dense matrices
  A[levels].matVecLeafHorizontalPass(comm[levels]);
  // write back the result (accumulated in Y_leaf)
  X_in = Y_leaf;
}

template <typename DT>
void H2MatrixSolver<DT>::factorizeM() {
  // factorize all levels, bottom  up
  // TODO how does this precompute the fill-ins?
  // I thought that would require 2 loops?
  for (long long l = levels; l >= 0; l--)
    A[l].factorize(comm[l]);
}

template <typename DT>
void H2MatrixSolver<DT>::solvePrecondition(DT X[]) {
  if (levels < 0)
    return;
  
  typedef Eigen::Map<Eigen::Matrix<DT, Eigen::Dynamic, 1>> VectorMap_dt;
  long long lbegin = comm[levels].oLocal();
  long long llen = comm[levels].lenLocal();
  long long lenX = std::reduce(A[levels].Dims.begin() + lbegin, A[levels].Dims.begin() + (lbegin + llen));
  
  VectorMap_dt X_in(X, lenX);
  VectorMap_dt X_leaf(A[levels].X[lbegin], lenX);

  // reset X to 0 for all levels
  for (long long l = levels; l >= 0; l--)
    A[l].resetX();
  // store X_in in X of the leaf levels
  X_leaf = X_in;

  // forward substitution for all levels
  std::cout<<"Forward"<<std::endl;
  for (long long l = levels; l >= 0; l--) {
    std::cout<<"Level "<<l<<std::endl;
    A[l].forwardSubstitute(comm[l]);
  }
  // backward substitution for all levels
  std::cout<<"Backward"<<std::endl;
  for (long long l = 0; l <= levels; l++)
    A[l].backwardSubstitute(comm[l]);
  // get the result from the leaf level and stor in X_in
  X_in = X_leaf;
}

template <typename DT>
void H2MatrixSolver<DT>::solveGMRES(double tol, H2MatrixSolver& M, DT x[], const DT b[], long long inner_iters, long long outer_iters) {

  long long lbegin = comm[levels].oLocal();
  long long llen = comm[levels].lenLocal();
  long long N = std::reduce(A[levels].Dims.begin() + lbegin, A[levels].Dims.begin() + (lbegin + llen));
  long long ld = inner_iters + 1;

  typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> Vector_dt;
  typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;
  Eigen::Map<const Vector_dt> B(b, N);
  Eigen::Map<Vector_dt> X(x, N);
  Vector_dt R = B;
  M.solvePrecondition(R.data());

  DT normr = R.adjoint() * R;
  comm[levels].level_sum(&normr, 1);
  double normb = std::sqrt(std::real(normr));
  //double normb = std::sqrt(normr.real());
  if (normb == 0.)
    normb = 1.;
  resid = std::vector<double>(outer_iters + 1, 0.);

  for (iters = 0; iters < outer_iters; iters++) {
    R = -X;
    matVecMul(R.data());
    R += B;
    M.solvePrecondition(R.data());

    normr = R.adjoint() * R;
    comm[levels].level_sum(&normr, 1);
    double beta = std::sqrt(std::real(normr));
    resid[iters] = beta / normb;
    if (resid[iters] < tol)
      return;

    Matrix_dt H = Matrix_dt::Zero(ld, inner_iters);
    Matrix_dt v = Matrix_dt::Zero(N, ld);
    v.col(0) = R * (1. / beta);
    
    for (long long i = 0; i < inner_iters; i++) {
      Vector_dt w = v.col(i);
      matVecMul(w.data());
      M.solvePrecondition(w.data());

      for (long long k = 0; k <= i; k++)
        H(k, i) = v.col(k).adjoint() * w;
      comm[levels].level_sum(H.col(i).data(), i + 1);

      for (long long k = 0; k <= i; k++)
        w -= H(k, i) * v.col(k);

      DT normw = w.adjoint() * w;
      comm[levels].level_sum(&normw, 1);
      H(i + 1, i) = std::sqrt(std::real(normw));
      v.col(i + 1) = w * ((DT)1. / H(i + 1, i));
    }

    Vector_dt s = Vector_dt::Zero(ld);
    s(0) = beta;

    Vector_dt y = H.colPivHouseholderQr().solve(s);
    X += v.leftCols(inner_iters) * y;
  }

  R = -X;
  matVecMul(R.data());
  R += B;
  M.solvePrecondition(R.data());

  normr = R.adjoint() * R;
  comm[levels].level_sum(&normr, 1);
  double beta = std::sqrt(std::real(normr));
  resid[outer_iters] = beta / normb;
}

template <typename DT>
long long H2MatrixSolver<DT>::solveMyGMRES(double tol, H2MatrixSolver& M, DT x[], const DT b[], long long inner_iters, long long outer_iters) {

  long long lbegin = comm[levels].oLocal();
  long long llen = comm[levels].lenLocal();
  long long N = std::reduce(A[levels].Dims.begin() + lbegin, A[levels].Dims.begin() + (lbegin + llen));
  long long ld = inner_iters + 1;
 
  typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> Vector_dt;
  typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;
  Eigen::Map<const Vector_dt> B(b, N);
  Eigen::Map<Vector_dt> X(x, N);
  
  // We assume we do not have an initial solution x0 yet (i.e. zero initialized)
  // compute x0
  X = B;
  M.solvePrecondition(X.data());

  // compute the residual
  Vector_dt R = -X;
  matVecMul(R.data());
  R += B;

  // compute the initial solution A d0 = res0 (stored in R)
  M.solvePrecondition(R.data());

  DT normr = R.adjoint() * R;
  comm[levels].level_sum(&normr, 1);
  // ||d0||
  double normd = std::sqrt(std::real(normr));
  double beta = normd;
  // we could use this as a switch beteen LU and GMRES IR
  // but not tested yet
  //if (normd < tol) {
    // use LU-IR whenever possible
  //  X += R;
  //  return;
  //}

  for (iters = 0; iters < outer_iters; iters++) {
    // restart GMRES with new d0
    Matrix_dt H = Matrix_dt::Zero(ld, inner_iters);
    Matrix_dt v = Matrix_dt::Zero(N, ld);
    v.col(0) = R * (1. / beta);
    
    for (long long i = 0; i < inner_iters; i++) {
      Vector_dt w = v.col(i);
      matVecMul(w.data());
      M.solvePrecondition(w.data());

      for (long long k = 0; k <= i; k++)
        H(k, i) = v.col(k).adjoint() * w;
      comm[levels].level_sum(H.col(i).data(), i + 1);

      for (long long k = 0; k <= i; k++)
        w -= H(k, i) * v.col(k);

      DT normw = w.adjoint() * w;
      comm[levels].level_sum(&normw, 1);
      H(i + 1, i) = std::sqrt(std::real(normw));
      v.col(i + 1) = w * ((DT)1. / H(i + 1, i));
    }

    Vector_dt s = Vector_dt::Zero(ld);
    s(0) = beta;

    Vector_dt y = H.colPivHouseholderQr().solve(s);
    //  d0 + di
    X += v.leftCols(inner_iters) * y;
    // beta is equal to ||di||
    // we could use this to check for early finish
    /*normr = y.squaredNorm();
    comm[levels].level_sum(&normr, 1);
    beta = std::sqrt(std::real(normr));
    if (beta / normd < tol) {
      // in this case we have ||di|| / ||d0|| < tol and we are finished
      std::cout<<"FINISHED"<<std::endl;
      return;
    }*/
    // but instead we calculate the next residual
    R = -X;
    matVecMul(R.data());
    R += B;

    // compute the new initial solution A d0 = resi (stored in R)
    M.solvePrecondition(R.data());
    //X += R;

    DT normr = R.adjoint() * R;
    comm[levels].level_sum(&normr, 1);
    // ||d0||
    beta = std::sqrt(std::real(normr));
    if (beta / normd < tol) {
      // in this case we have || new d0|| / ||d0|| < tol and we are finished
      // X += R;
      return iters * inner_iters;
    }
  }
  return outer_iters * inner_iters;
}

template <typename DT>
void H2MatrixSolver<DT>::free_all_comms() {
  for (MPI_Comm& c : allocedComm)
    MPI_Comm_free(&c);
  allocedComm.clear();
}

template <typename DT>
double computeRelErr(const long long lenX, const DT X[], const DT ref[], MPI_Comm world) {
  double err[2] = { 0., 0. };
  for (long long i = 0; i < lenX; i++) {
    DT diff = X[i] - ref[i];
    // sum of squares of diff
    err[0] = err[0] + diff * diff;
    // sum of squared of the reference
    err[1] = err[1] + ref[i] * ref[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, world);
  // the relative error
  return std::sqrt(err[0] / err[1]);
}
template <typename DT>
double computeRelErr(const long long lenX, const std::complex<DT> X[], const std::complex<DT> ref[], MPI_Comm world) {
  double err[2] = { 0., 0. };
  for (long long i = 0; i < lenX; i++) {
    std::complex<DT> diff = X[i] - ref[i];
    // sum of squares of diff
    err[0] = err[0] + (diff.real() * diff.real());
    // sum of squared of the reference
    err[1] = err[1] + (ref[i].real() * ref[i].real());
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, world);
  // the relative error
  return std::sqrt(err[0] / err[1]);
}