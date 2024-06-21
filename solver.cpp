
#include <solver.hpp>

#include <mkl.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

H2MatrixSolver::H2MatrixSolver(const MatrixAccessor& eval, double epi, long long rank, const Cell cells[], long long ncells, const CSR& Near, const CSR& Far, const double bodies[], long long levels, MPI_Comm world) : 
  levels(levels), A(levels + 1), M(levels + 1), comm(levels + 1), allocedComm(), timer(0., 0.), local_bodies(0, 0) {
  CSR cellNeighbor(Near, Far);
  int mpi_size = 1;
  MPI_Comm_size(world, &mpi_size);

  std::vector<std::pair<long long, long long>> mapping(mpi_size, std::make_pair(0, 1));
  std::vector<std::pair<long long, long long>> tree(ncells);
  std::transform(cells, &cells[ncells], tree.begin(), [](const Cell& c) { return std::make_pair(c.Child[0], c.Child[1]); });
  
  for (long long i = 0; i <= levels; i++) {
    comm[i] = ColCommMPI(&tree[0], &mapping[0], cellNeighbor.RowIndex.data(), cellNeighbor.ColIndex.data(), allocedComm, world);
    comm[i].timer = &timer;
  }

  std::vector<WellSeparatedApproximation> wsa(levels + 1);
  for (long long l = 1; l <= levels; l++)
    wsa[l] = WellSeparatedApproximation(eval, epi, rank, comm[l].oGlobal(), comm[l].lenLocal(), cells, Far, bodies, wsa[l - 1]);

  A[levels] = H2Matrix(eval, epi, cells, Near, Far, bodies, wsa[levels], comm[levels], A[levels], comm[levels], false);
  for (long long l = levels - 1; l >= 0; l--)
    A[l] = H2Matrix(eval, epi, cells, Near, Far, bodies, wsa[l], comm[l], A[l + 1], comm[l + 1], false);

  M[levels] = H2Matrix(eval, rank, cells, Near, Far, bodies, wsa[levels], comm[levels], M[levels], comm[levels], true);
  for (long long l = levels - 1; l >= 0; l--)
    M[l] = H2Matrix(eval, rank, cells, Near, Far, bodies, wsa[l], comm[l], M[l + 1], comm[l + 1], true);

  long long llen = comm[levels].lenLocal();
  long long gbegin = comm[levels].oGlobal();
  local_bodies = std::make_pair(cells[gbegin].Body[0], cells[gbegin + llen - 1].Body[1]);
}

void H2MatrixSolver::matVecMul(std::complex<double> X[]) {
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
    M[l].factorize(comm[l]);
}

void H2MatrixSolver::solvePrecondition(std::complex<double> X[]) {
  typedef Eigen::Map<Eigen::VectorXcd> Vector_t;
  long long lbegin = comm[levels].oLocal();
  long long llen = comm[levels].lenLocal();
  long long lenX = std::reduce(M[levels].Dims.begin() + lbegin, M[levels].Dims.begin() + (lbegin + llen));
  
  Vector_t X_in(X, lenX);
  Vector_t X_leaf(M[levels].X[lbegin], lenX);

  for (long long l = levels; l >= 0; l--)
    M[l].resetX();
  X_leaf = X_in;

  for (long long l = levels; l >= 0; l--)
    M[l].forwardSubstitute(comm[l]);
  for (long long l = 0; l <= levels; l++)
    M[l].backwardSubstitute(comm[l]);
  X_in = X_leaf;
}

std::pair<double, long long> H2MatrixSolver::solveGMRES(double tol, std::complex<double> x[], const std::complex<double> b[], long long inner_iters, long long outer_iters) {
  using Eigen::VectorXcd, Eigen::MatrixXcd;

  long long lbegin = comm[levels].oLocal();
  long long llen = comm[levels].lenLocal();
  long long N = std::reduce(A[levels].Dims.begin() + lbegin, A[levels].Dims.begin() + (lbegin + llen));
  long long ld = inner_iters + 1;

  Eigen::Map<const Eigen::VectorXcd> B(b, N);
  Eigen::Map<Eigen::VectorXcd> X(x, N);
  VectorXcd R = B;
  solvePrecondition(R.data());

  std::complex<double> normr = R.adjoint() * R;
  comm[levels].level_sum(&normr, 1);
  double normb = std::sqrt(normr.real());
  if (normb == 0.)
    normb = 1.;

  for (long long j = 0; j < outer_iters; j++) {
    R = -X;
    matVecMul(R.data());
    R += B;
    solvePrecondition(R.data());

    normr = R.adjoint() * R;
    comm[levels].level_sum(&normr, 1);
    double beta = std::sqrt(normr.real());
    double resid = beta / normb;
    if (resid < tol)
      return std::make_pair(resid, j);

    MatrixXcd H = MatrixXcd::Zero(ld, inner_iters);
    MatrixXcd v = MatrixXcd::Zero(N, ld);
    v.col(0) = R * (1. / beta);
    
    for (long long i = 0; i < inner_iters; i++) {
      VectorXcd w = v.col(i);
      matVecMul(w.data());
      solvePrecondition(w.data());

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
  solvePrecondition(R.data());

  normr = R.adjoint() * R;
  comm[levels].level_sum(&normr, 1);
  double beta = std::sqrt(normr.real());
  double resid = beta / normb;
  return std::make_pair(resid, outer_iters);
}

void H2MatrixSolver::free_all_comms() {
  for (MPI_Comm& c : allocedComm)
    MPI_Comm_free(&c);
  allocedComm.clear();
}
