
#include <solver.hpp>

#include <hidr.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>

// complex double
template class H2MatrixSolver<std::complex<double>>;
template void H2MatrixSolver<std::complex<double>>::solveGMRES<std::complex<float>>(double, H2MatrixSolver<std::complex<float>>&, std::complex<double>[], const std::complex<double>[], long long, long long);
template double solveRelErr(long long, const std::complex<double> X[], const std::complex<double> ref[], MPI_Comm);

template <typename DT>
H2MatrixSolver<DT>::H2MatrixSolver() : levels(-1), A(), comm(), allocedComm(), local_bodies(0, 0) {
}

template <typename DT>
H2MatrixSolver<DT>::H2MatrixSolver(const MatrixGenerator& matgen, double epi, long long rank, long long leveled_rank, const std::vector<Cell>& cells, double theta, long long levels, double omega, std::string filename, bool verbose, MPI_Comm world) : 
  levels(levels), A(levels + 1), local_bodies(0, 0) {
  
  CSR Near('N', cells, cells, theta);
  CSR Far('F', cells, cells, theta);
  int mpi_size = 1;
  MPI_Comm_size(world, &mpi_size);
  std::vector<std::pair<long long, long long>> mapping(mpi_size, std::make_pair(0, 1));
  std::vector<std::pair<long long, long long>> tree(cells.size());
  std::transform(cells.begin(), cells.end(), tree.begin(), [](const Cell& c) { return std::make_pair(c.Child[0], c.Child[1]); });
  for (long long i = 0; i <= levels; i++) {
    comm.emplace_back(&tree[0], &mapping[0], Near.RowIndex.data(), Near.ColIndex.data(), Far.RowIndex.data(), Far.ColIndex.data(), allocedComm, world);
  }
  bool fix_rank = (epi == 0.);
  auto rank_func = [=](long long l) { return (levels - l) * leveled_rank + rank; };
  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if (!filename.empty()) {
    if (verbose && mpi_rank == 0) {
      std::cout<<"Start reading from file."<<std::endl;
    }
    // read from the files if they exist, otherwise construct and write
    if (!A[levels].read(levels, filename)) {
      if (mpi_rank == 0) {
        std::cerr<<"Reading from file failed."<<std::endl;
      }
      A[levels].construct(matgen, fix_rank ? (double)rank_func(levels) : epi, cells.data(), Near, comm[levels], A[levels], comm[levels], omega);
      A[levels].write(levels, filename);
    }
  } else {
    A[levels].construct(matgen, fix_rank ? (double)rank_func(levels) : epi, cells.data(), Near, comm[levels], A[levels], comm[levels], omega);
  }
  A[levels].lowest = true;
  for (long long l = levels - 1; l >= 0; l--) {
    if (verbose && mpi_rank == 0) {
      std::cout<<"Construct level "<<l<<std::endl;
    }
    if (!filename.empty()) {
      if (!A[l].read(l, filename)) {
        if (mpi_rank == 0) {
          std::cerr<<"Reading from file failed."<<std::endl;
        }
        A[l].construct(matgen, fix_rank ? (double)rank_func(l) : epi, cells.data(), Near, comm[l], A[l + 1], comm[l + 1], omega);
        A[l].write(l, filename);
      }
    } else {
      A[l].construct(matgen, fix_rank ? (double)rank_func(l) : epi, cells.data(), Near, comm[l], A[l + 1], comm[l + 1], omega);
    }
  }
  long long llen = comm[levels].lenLocal();
  long long gbegin = comm[levels].oGlobal();
  local_bodies = std::make_pair(cells[gbegin].Body[0], cells[gbegin + llen - 1].Body[1]);
}

std::vector<double> extract_coords(const std::vector<elastWave3d::element>& elems) {
  std::vector<double> pts(elems.size() * 3);
  for (size_t i = 0; i < elems.size(); ++i) {
    pts[i * 3] = elems[i].xc[0];
    pts[i * 3 + 1] = elems[i].xc[1];
    pts[i * 3 + 2] = elems[i].xc[3];
  }
  return pts;
}

template <typename DT>
H2MatrixSolver<DT>::H2MatrixSolver(const MatrixGenerator& matgen, double epi, long long rank, long long leveled_rank, const std::vector<Cell>& cells, double theta, long long levels, double omega, const std::vector<elastWave3d::element>& elems, long long s1, long long s2, bool verbose, MPI_Comm world) : 
  levels(levels), A(levels + 1), local_bodies(0, 0) {
  
  CSR Near('N', cells, cells, theta);
  CSR Far('F', cells, cells, theta);
  
  int mpi_size = 1;
  MPI_Comm_size(world, &mpi_size);
  std::vector<std::pair<long long, long long>> mapping(mpi_size, std::make_pair(0, 1));
  std::vector<std::pair<long long, long long>> tree(cells.size());
  std::transform(cells.begin(), cells.end(), tree.begin(), [](const Cell& c) { return std::make_pair(c.Child[0], c.Child[1]); });
  for (long long i = 0; i <= levels; i++) {
    comm.emplace_back(&tree[0], &mapping[0], Near.RowIndex.data(), Near.ColIndex.data(), Far.RowIndex.data(), Far.ColIndex.data(), allocedComm, world);
  }

  bool fix_rank = (epi == 0.);
  auto rank_func = [=](long long l) { return (levels - l) * leveled_rank + rank; };
  // create a new far field for the factorization basis
  if (fix_rank)
   Far = CSR('F', cells, cells, 0);
  

  std::vector<HiDR> hidr(levels + 1);
  auto pts = extract_coords(elems);
  // the sampling is fast, so we just re-create it on each process
  long long Nleaf = (long long)1 << levels;
  hidr[levels].initialize_f(s1, Nleaf - 1, Nleaf, cells.data(), pts);
  // root does not need to be sampled, since it will always be split
  for (long long l = levels - 1; l > 0; l--) {
    Nleaf >>= 1;
    hidr[l].bottom_up_sweep_f(s1, Nleaf - 1, Nleaf, cells.data(), hidr[l + 1]);
  }
  for (long long l = 1; l <= levels; l++) {
    hidr[l].top_down_sweep_f(s2, cells.data(), Far, hidr[l - 1]);
  }

  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  A[levels].construct_hidr(matgen, fix_rank ? (double)rank_func(levels) : epi, cells.data(), Near, hidr[levels], comm[levels], A[levels], comm[levels], omega);
  for (long long l = levels - 1; l >= 0; l--) {
    if (verbose && mpi_rank == 0) {
      std::cout<<"Construct level "<<l<<std::endl;
    }
    A[l].construct_hidr(matgen, fix_rank ? (double)rank_func(l) : epi, cells.data(), Near, hidr[l], comm[l], A[l + 1], comm[l + 1], omega);
  }
  long long llen = comm[levels].lenLocal();
  long long gbegin = comm[levels].oGlobal();
  local_bodies = std::make_pair(cells[gbegin].Body[0], cells[gbegin + llen - 1].Body[1]);
}

template <typename DT>
H2MatrixSolver<DT>::H2MatrixSolver(const H2MatrixSolver& solver) :
  levels(solver.levels), local_bodies(solver.local_bodies) {
  // this should duplicate all the allocated communicators
  for (size_t i = 0; i < solver.allocedComm.size(); ++i) {
    MPI_Comm mpi_comm = MPI_COMM_NULL;
    MPI_Comm_dup(solver.allocedComm[i], &mpi_comm);
    allocedComm.emplace_back(mpi_comm);
  }
  for (size_t i = 0; i < solver.comm.size(); ++i) {
    comm.emplace_back(ColCommMPI(solver.comm[i], allocedComm));
  }
  A.reserve(solver.A.size());
  for (size_t i = 0; i < solver.A.size(); ++i) {
    A.emplace_back(H2Matrix(solver.A[i]));
  }
}

template <typename DT>
void H2MatrixSolver<DT>::matVecMul(DT X[]) {
  if (levels < 0)
    return;

  A[levels].matVecUpwardPass(X, comm[levels]);
  for (long long l = levels - 1; l >= 0; l--){
    A[l].matVecUpwardPass(A[l + 1].Z[0], comm[l]);
  }

  for (long long l = 0; l < levels; l++)
    A[l].matVecHorizontalandDownwardPass(A[l + 1].W[0], comm[l]);

  A[levels].matVecLeafHorizontalPass(X, comm[levels]);
}

template <typename DT>
void H2MatrixSolver<DT>::factorizeM() {
  for (long long l = levels; l >= 0; l--) {
    A[l].factorize(comm[l]);
    if (0 < l)
      A[l - 1].factorizeCopyNext(A[l], comm[l]);
  }

  for (long long l = levels; l >= 0; l--)
    if (A[l].info)
      printf("singularity detected at level %lld.\n", l);
}

template <typename DT>
void H2MatrixSolver<DT>::solvePrecondition(DT X[]) {
  if (levels < 0)
    return;

  A[levels].forwardSubstitute(X, comm[levels]);
  for (long long l = levels - 1; l >= 0; l--)
    A[l].forwardSubstitute(A[l + 1].Z[0], comm[l]);

  for (long long l = 0; l < levels; l++)
    A[l].backwardSubstitute(A[l + 1].W[0], comm[l]);
  A[levels].backwardSubstitute(X, comm[levels]);
}

template <typename DT>
void H2MatrixSolver<DT>::solveGMRES(double tol, H2MatrixSolver& M, DT x[], const DT b[], long long inner_iters, long long outer_iters) {
  typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> Vector_dt;
  typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;

  long long N = A[levels].lenX;
  long long ld = inner_iters + 1;

  Eigen::Map<const Vector_dt> B(b, N);
  Eigen::Map<Vector_dt> X(x, N);

  DT nsum = B.adjoint() * B;
  comm[levels].level_sum(&nsum, 1);
  double normb = std::sqrt(nsum.real());
  if (normb == 0.)
    normb = 1.;

  Vector_dt R = B;
  resid.resize(outer_iters + 1);
  resid[0] = 1.;
  iters = 0;

  while (iters < outer_iters && tol <= resid[iters]) {
    M.solvePrecondition(R.data());
    nsum = R.adjoint() * R;
    comm[levels].level_sum(&nsum, 1);

    double beta = std::sqrt(nsum.real());
    Matrix_dt H = Matrix_dt::Zero(ld, inner_iters);
    Matrix_dt v = Matrix_dt::Zero(N, ld);
    v.col(0) = R * (1. / beta);
    
    for (long long i = 0; i < inner_iters; i++) {
      R = v.col(i);
      matVecMul(R.data());
      M.solvePrecondition(R.data());

      H.block(0, i, i + 1, 1).noalias() = v.leftCols(i + 1).adjoint() * R;
      comm[levels].level_sum(H.col(i).data(), i + 1);
      R.noalias() -= v.leftCols(i + 1) * H.block(0, i, i + 1, 1);

      nsum = R.adjoint() * R;
      comm[levels].level_sum(&nsum, 1);
      H(i + 1, i) = std::sqrt(nsum.real());
      v.col(i + 1) = R * ((DT)1. / H(i + 1, i));
    }

    Vector_dt s = Vector_dt::Zero(ld);
    s(0) = beta;

    R = H.householderQr().solve(s);
    X.noalias() += v.leftCols(inner_iters) * R;

    R = -X;
    matVecMul(R.data());
    R += B;

    nsum = R.adjoint() * R;
    comm[levels].level_sum(&nsum, 1);
    resid[++iters] = std::sqrt(nsum.real()) / normb;
  }
}

template <typename DT>
void H2MatrixSolver<DT>::solveGMRES(double tol, DT x[], const DT b[], long long inner_iters, long long outer_iters) {
  typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> Vector_dt;
  typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;

  long long N = A[levels].lenX;
  long long ld = inner_iters + 1;

  Eigen::Map<const Vector_dt> B(b, N);
  Eigen::Map<Vector_dt> X(x, N);

  DT nsum = B.adjoint() * B;
  comm[levels].level_sum(&nsum, 1);
  double normb = std::sqrt(nsum.real());
  if (normb == 0.)
    normb = 1.;

  Vector_dt R = B;
  resid.resize(outer_iters + 1);
  resid[0] = 1.;
  iters = 0;

  while (iters < outer_iters && tol <= resid[iters]) {
    nsum = R.adjoint() * R;
    comm[levels].level_sum(&nsum, 1);

    double beta = std::sqrt(nsum.real());
    Matrix_dt H = Matrix_dt::Zero(ld, inner_iters);
    Matrix_dt v = Matrix_dt::Zero(N, ld);
    v.col(0) = R * (1. / beta);
    
    for (long long i = 0; i < inner_iters; i++) {
      R = v.col(i);
      matVecMul(R.data());

      H.block(0, i, i + 1, 1).noalias() = v.leftCols(i + 1).adjoint() * R;
      comm[levels].level_sum(H.col(i).data(), i + 1);
      R.noalias() -= v.leftCols(i + 1) * H.block(0, i, i + 1, 1);

      nsum = R.adjoint() * R;
      comm[levels].level_sum(&nsum, 1);
      H(i + 1, i) = std::sqrt(nsum.real());
      v.col(i + 1) = R * ((DT)1. / H(i + 1, i));
    }

    Vector_dt s = Vector_dt::Zero(ld);
    s(0) = beta;

    R = H.householderQr().solve(s);
    X.noalias() += v.leftCols(inner_iters) * R;

    R = -X;
    matVecMul(R.data());
    R += B;

    nsum = R.adjoint() * R;
    comm[levels].level_sum(&nsum, 1);
    resid[++iters] = std::sqrt(nsum.real()) / normb;
  }
}

template <typename DT> template <typename OT>
void H2MatrixSolver<DT>::solveGMRES(double tol, H2MatrixSolver<OT>& M, DT x[], const DT b[], long long inner_iters, long long outer_iters) {
  typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> Vector_dt;
  typedef Eigen::Matrix<OT, Eigen::Dynamic, 1> Vector_ot;
  typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;

  long long N = A[levels].lenX;
  long long ld = inner_iters + 1;

  Eigen::Map<const Vector_dt> B(b, N);
  Eigen::Map<Vector_dt> X(x, N);

  double nsum = B.squaredNorm();
  comm[levels].level_sum(&nsum, 1);
  double normb = std::sqrt(nsum);
  if (normb == 0.)
    normb = 1.;

  Vector_dt R = B;
  Vector_ot R_low;
  resid.resize(outer_iters + 1);
  resid[0] = 1.;
  iters = 0;

  while (iters < outer_iters && tol <= resid[iters]) {
    R_low = R.template cast<OT>();
    M.solvePrecondition(R_low.data());
    R = R_low.template cast<DT>();
    nsum = R.squaredNorm();
    comm[levels].level_sum(&nsum, 1);

    double beta = std::sqrt(nsum);
    Matrix_dt H = Matrix_dt::Zero(ld, inner_iters);
    Matrix_dt v = Matrix_dt::Zero(N, ld);
    v.col(0) = R * (1. / beta);
    
    for (long long i = 0; i < inner_iters; i++) {
      R = v.col(i);
      matVecMul(R.data());
      R_low = R.template cast<OT>();
      M.solvePrecondition(R_low.data());
      R = R_low.template cast<DT>();

      H.block(0, i, i + 1, 1).noalias() = v.leftCols(i + 1).adjoint() * R;
      comm[levels].level_sum(H.col(i).data(), i + 1);
      R.noalias() -= v.leftCols(i + 1) * H.block(0, i, i + 1, 1);

      nsum = R.squaredNorm();
      comm[levels].level_sum(&nsum, 1);
      H(i + 1, i) = std::sqrt(nsum);
      v.col(i + 1) = R * ((DT)1. / H(i + 1, i));
    }

    Vector_dt s = Vector_dt::Zero(ld);
    s(0) = beta;

    R = H.householderQr().solve(s);
    X.noalias() += v.leftCols(inner_iters) * R;

    R = -X;
    matVecMul(R.data());
    R += B;

    nsum = R.squaredNorm();
    comm[levels].level_sum(&nsum, 1);
    resid[++iters] = std::sqrt(nsum) / normb;
  }
}

template <typename DT>
void H2MatrixSolver<DT>::free_all_comms() {
  for (MPI_Comm& c : allocedComm)
    MPI_Comm_free(&c);
  allocedComm.clear();
}

template <typename DT>
double solveRelErr(long long lenX, const DT X[], const DT ref[], MPI_Comm world) {
  double err[2] = { 0., 0. };
  for (long long i = 0; i < lenX; i++) {
    DT diff = X[i] - ref[i];
    err[0] = err[0] + (diff.real() * diff.real());
    err[1] = err[1] + (ref[i].real() * ref[i].real());
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, world);
  return std::sqrt(err[0] / err[1]);
}
