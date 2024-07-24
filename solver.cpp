
#include <solver.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>

H2MatrixSolver::H2MatrixSolver() : levels(-1), A(), comm(), allocedComm(), local_bodies(0, 0) {
}

H2MatrixSolver::H2MatrixSolver(const MatrixAccessor& eval, double epi, long long rank, const std::vector<Cell>& cells, double theta, const double bodies[], long long levels, bool fix_rank, MPI_Comm world) : 
  levels(levels), A(levels + 1), comm(levels + 1), allocedComm(), local_bodies(0, 0) {
  
  // stores the indices of the cells in the near field for each cell
  CSR Near('N', cells, theta);
  // stores the indices of the cell in the far field for each cell
  CSR Far('F', cells, theta);
  // stores the indices of all cells on the same level (i.e. near and far field)
  CSR Neighbor(Near, Far);
  int mpi_size = 1;
  MPI_Comm_size(world, &mpi_size);

  // is this the mapping of mpi processes to cells?
  std::vector<std::pair<long long, long long>> mapping(mpi_size, std::make_pair(0, 1));
  // this seems to be the cluster tree again, but each node contains the two children of that cell
  std::vector<std::pair<long long, long long>> tree(cells.size());
  std::transform(cells.begin(), cells.end(), tree.begin(), [](const Cell& c) { return std::make_pair(c.Child[0], c.Child[1]); });
  
  // create a communicator for each level
  for (long long i = 0; i <= levels; i++)
    comm[i] = ColCommMPI(&tree[0], &mapping[0], Neighbor.RowIndex.data(), Neighbor.ColIndex.data(), allocedComm, world);

  std::vector<WellSeparatedApproximation> wsa(levels + 1);
  for (long long l = 1; l <= levels; l++) {
    wsa[l] = WellSeparatedApproximation(eval, epi, rank, comm[l].oGlobal(), comm[l].lenLocal(), cells.data(), Far, bodies, wsa[l - 1]);
  }
  epi = fix_rank ? (double)rank : epi;
  A[levels] = H2Matrix(eval, epi, cells.data(), Near, Far, bodies, wsa[levels], comm[levels], A[levels], comm[levels], fix_rank);
  for (long long l = levels - 1; l >= 0; l--)
    A[l] = H2Matrix(eval, epi, cells.data(), Near, Far, bodies, wsa[l], comm[l], A[l + 1], comm[l + 1], fix_rank);

  long long llen = comm[levels].lenLocal();
  long long gbegin = comm[levels].oGlobal();
  local_bodies = std::make_pair(cells[gbegin].Body[0], cells[gbegin + llen - 1].Body[1]);
}

void H2MatrixSolver::matVecMul(std::complex<double> X[]) {
  if (levels < 0)
    return;
  
  typedef Eigen::Map<Eigen::VectorXcd> Vector_t;
  long long lbegin = comm[levels].oLocal();
  long long llen = comm[levels].lenLocal();
  long long lenX = std::reduce(A[levels].Dims.begin() + lbegin, A[levels].Dims.begin() + (lbegin + llen));
  
  Vector_t X_in(X, lenX);
  Vector_t X_leaf(A[levels].X[lbegin], lenX);
  Vector_t Y_leaf(A[levels].Y[lbegin], lenX);

  for (long long l = levels; l >= 0; l--)
    A[l].resetX();
  X_leaf = X_in;

  for (long long l = levels; l >= 0; l--)
    A[l].matVecUpwardPass(comm[l]);
  for (long long l = 0; l <= levels; l++)
    A[l].matVecHorizontalandDownwardPass(comm[l]);
  A[levels].matVecLeafHorizontalPass(comm[levels]);
  X_in = Y_leaf;
}

void H2MatrixSolver::factorizeM() {
  for (long long l = levels; l >= 0; l--)
    A[l].factorize(comm[l]);
}

void H2MatrixSolver::solvePrecondition(std::complex<double> X[]) {
  if (levels < 0)
    return;
  
  typedef Eigen::Map<Eigen::VectorXcd> Vector_t;
  long long lbegin = comm[levels].oLocal();
  long long llen = comm[levels].lenLocal();
  long long lenX = std::reduce(A[levels].Dims.begin() + lbegin, A[levels].Dims.begin() + (lbegin + llen));
  
  Vector_t X_in(X, lenX);
  Vector_t X_leaf(A[levels].X[lbegin], lenX);

  for (long long l = levels; l >= 0; l--)
    A[l].resetX();
  X_leaf = X_in;

  for (long long l = levels; l >= 0; l--)
    A[l].forwardSubstitute(comm[l]);
  for (long long l = 0; l <= levels; l++)
    A[l].backwardSubstitute(comm[l]);
  X_in = X_leaf;
}

void H2MatrixSolver::solveGMRES(double tol, H2MatrixSolver& M, std::complex<double> x[], const std::complex<double> b[], long long inner_iters, long long outer_iters) {
  using Eigen::VectorXcd, Eigen::MatrixXcd;

  long long lbegin = comm[levels].oLocal();
  long long llen = comm[levels].lenLocal();
  long long N = std::reduce(A[levels].Dims.begin() + lbegin, A[levels].Dims.begin() + (lbegin + llen));
  long long ld = inner_iters + 1;

  Eigen::Map<const Eigen::VectorXcd> B(b, N);
  Eigen::Map<Eigen::VectorXcd> X(x, N);
  VectorXcd R = B;
  M.solvePrecondition(R.data());

  std::complex<double> normr = R.adjoint() * R;
  comm[levels].level_sum(&normr, 1);
  double normb = std::sqrt(normr.real());
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
    double beta = std::sqrt(normr.real());
    resid[iters] = beta / normb;
    if (resid[iters] < tol)
      return;

    MatrixXcd H = MatrixXcd::Zero(ld, inner_iters);
    MatrixXcd v = MatrixXcd::Zero(N, ld);
    v.col(0) = R * (1. / beta);
    
    for (long long i = 0; i < inner_iters; i++) {
      VectorXcd w = v.col(i);
      matVecMul(w.data());
      M.solvePrecondition(w.data());

      for (long long k = 0; k <= i; k++)
        H(k, i) = v.col(k).adjoint() * w;
      comm[levels].level_sum(H.col(i).data(), i + 1);

      for (long long k = 0; k <= i; k++)
        w -= H(k, i) * v.col(k);

      std::complex<double> normw = w.adjoint() * w;
      comm[levels].level_sum(&normw, 1);
      H(i + 1, i) = std::sqrt(normw.real());
      v.col(i + 1) = w * (1. / H(i + 1, i));
    }

    VectorXcd s = VectorXcd::Zero(ld);
    s(0) = beta;

    VectorXcd y = H.colPivHouseholderQr().solve(s);
    X += v.leftCols(inner_iters) * y;
  }

  R = -X;
  matVecMul(R.data());
  R += B;
  M.solvePrecondition(R.data());

  normr = R.adjoint() * R;
  comm[levels].level_sum(&normr, 1);
  double beta = std::sqrt(normr.real());
  resid[outer_iters] = beta / normb;
}

void H2MatrixSolver::free_all_comms() {
  for (MPI_Comm& c : allocedComm)
    MPI_Comm_free(&c);
  allocedComm.clear();
}

double H2MatrixSolver::solveRelErr(long long lenX, const std::complex<double> X[], const std::complex<double> ref[], MPI_Comm world) {
  double err[2] = { 0., 0. };
  for (long long i = 0; i < lenX; i++) {
    std::complex<double> diff = X[i] - ref[i];
    err[0] = err[0] + (diff.real() * diff.real());
    err[1] = err[1] + (ref[i].real() * ref[i].real());
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, world);
  return std::sqrt(err[0] / err[1]);
}
