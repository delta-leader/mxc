
#include <solver.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>

/* explicit template instantiation */
// complex double
template class H2MatrixSolver<std::complex<double>>;
template void H2MatrixSolver<std::complex<double>>::solveGMRES<std::complex<float>>(double, H2MatrixSolver<std::complex<float>>&, std::complex<double>[], const std::complex<double>[], long long, long long);
template double solveRelErr<std::complex<double>>(long long, const std::complex<double> X[], const std::complex<double> ref[], MPI_Comm);
// complex float
template class H2MatrixSolver<std::complex<float>>;
template double solveRelErr<std::complex<float>>(long long, const std::complex<float> X[], const std::complex<float> ref[], MPI_Comm);
// double
template class H2MatrixSolver<double>;
template void H2MatrixSolver<double>::solveGMRES<float>(double, H2MatrixSolver<float>&, double[], const double[], long long, long long);
template double solveRelErr<double>(long long, const double X[], const double ref[], MPI_Comm);
// float
template class H2MatrixSolver<float>;
template double solveRelErr<float>(long long, const float X[], const float ref[], MPI_Comm);

/* supported type conversions */
// (complex) double to float
template H2MatrixSolver<std::complex<float>>::H2MatrixSolver(const H2MatrixSolver<std::complex<double>>&);
template H2MatrixSolver<float>::H2MatrixSolver(const H2MatrixSolver<double>&);
// (complex) float to double
template H2MatrixSolver<std::complex<double>>::H2MatrixSolver(const H2MatrixSolver<std::complex<float>>&);
template H2MatrixSolver<double>::H2MatrixSolver(const H2MatrixSolver<float>&);

template <typename DT>
H2MatrixSolver<DT>::H2MatrixSolver() : levels(-1), A(), comm(), allocedComm(), local_bodies(0, 0) {
}

template <typename DT>
H2MatrixSolver<DT>::H2MatrixSolver(const MatrixAccessor<DT>& eval, double epi, long long rank, long long leveled_rank, const std::vector<Cell>& cells, double theta, const double bodies[], long long levels, MPI_Comm world) : 
  levels(levels), A(levels + 1), local_bodies(0, 0) {
  
  CSR Near('N', cells, cells, theta);
  CSR Far('F', cells, cells, theta);
  int mpi_size = 1;
  MPI_Comm_size(world, &mpi_size);

  std::vector<std::pair<long long, long long>> mapping(mpi_size, std::make_pair(0, 1));
  std::vector<std::pair<long long, long long>> tree(cells.size());
  std::transform(cells.begin(), cells.end(), tree.begin(), [](const Cell& c) { return std::make_pair(c.Child[0], c.Child[1]); });
  
  for (long long i = 0; i <= levels; i++)
    comm.emplace_back(&tree[0], &mapping[0], Near.RowIndex.data(), Near.ColIndex.data(), Far.RowIndex.data(), Far.ColIndex.data(), allocedComm, world);

  auto rank_func = [=](long long l) { return (levels - l) * leveled_rank + rank; };
  std::vector<WellSeparatedApproximation<DT>> wsa(levels + 1);
  for (long long l = 1; l <= levels; l++)
    wsa[l].construct(eval, epi, rank_func(l), comm[l].oGlobal(), comm[l].lenLocal(), cells.data(), Far, bodies, wsa[l - 1]);

  bool fix_rank = (epi == 0.);
  A[levels].construct(eval, fix_rank ? (double)rank_func(levels) : epi, cells.data(), Near, Far, bodies, wsa[levels], comm[levels], A[levels], comm[levels]);
  for (long long l = levels - 1; l >= 0; l--)
    A[l].construct(eval, fix_rank ? (double)rank_func(l) : epi, cells.data(), Near, Far, bodies, wsa[l], comm[l], A[l + 1], comm[l + 1]);

  long long llen = comm[levels].lenLocal();
  long long gbegin = comm[levels].oGlobal();
  local_bodies = std::make_pair(cells[gbegin].Body[0], cells[gbegin + llen - 1].Body[1]);
}

template <typename DT> template <typename OT>
H2MatrixSolver<DT>::H2MatrixSolver(const H2MatrixSolver<OT>& solver) :
  levels(solver.levels), local_bodies(solver.local_bodies) {
  
  for (size_t i = 0; i < solver.allocedComm.size(); ++i) {
    MPI_Comm mpi_comm = MPI_COMM_NULL;
    MPI_Comm_dup(solver.allocedComm[i], &mpi_comm);
    allocedComm.emplace_back(mpi_comm);
  }
  for (size_t i = 0; i < solver.comm.size(); ++i) {
    comm.emplace_back(ColCommMPI(solver.comm[i], allocedComm));
  }
  A.reserve(solver.A.size());
  for (size_t i = 0; i < A.size(); ++i) {
    A.push_back(H2Matrix<DT>(solver.A[i]));
  }
}

template <typename DT>
void H2MatrixSolver<DT>::init_gpu_handles(const ncclComms nccl_comms) {
  desc.resize(levels + 1);
  long long bdim = *std::max_element(A[levels].Dims.begin(), A[levels].Dims.end());
  long long rank = *std::max_element(A[levels].DimsLr.begin(), A[levels].DimsLr.end());
  createMatrixDesc(&desc[levels], bdim, rank, deviceMatrixDesc_t<DT>(), comm[levels], nccl_comms);

  for (long long l = levels - 1; l >= 0; l--) {
    long long bdim = *std::max_element(A[l].Dims.begin(), A[l].Dims.end());
    long long rank = *std::max_element(A[l].DimsLr.begin(), A[l].DimsLr.end());
    createMatrixDesc(&desc[l], bdim, rank, desc[l + 1], comm[l], nccl_comms);
  }

  long long lenX = bdim * comm[levels].lenLocal();
  cudaMalloc(reinterpret_cast<void**>(&X_dev), lenX * sizeof(DT));
}

template <typename DT>
void H2MatrixSolver<DT>::allocSparseMV(deviceHandle_t handle, const ncclComms nccl_comms) {
  A_mv.resize(levels + 1);
  for (long long l = 0; l <= levels; l++) {
    createSpMatrixDesc(handle, &A_mv[l], l == levels, A[l].LowerZ, A[l].Dims.data(), A[l].DimsLr.data(), A[l].U[0], A[l].C[0], A[l].A[0], comm[l], nccl_comms);
  }
}

template <typename DT>
void H2MatrixSolver<DT>::matVecMulSp(deviceHandle_t handle, DT X[]) {
  if (levels < 0)
    return;

  long long lenX = A[levels].lenX;
  cudaMemcpy(X_dev, X, lenX * sizeof(DT), cudaMemcpyHostToDevice);
  matVecDeviceH2(handle, levels, A_mv.data(), X_dev);
  cudaMemcpy(X, X_dev, lenX * sizeof(DT), cudaMemcpyDeviceToHost);
}

template <typename DT>
void H2MatrixSolver<DT>::matVecMul(DT X[]) {
  if (levels < 0)
    return;

  A[levels].matVecUpwardPass(X, comm[levels]);
  for (long long l = levels - 1; l >= 0; l--)
    A[l].matVecUpwardPass(A[l + 1].Z[0], comm[l]);

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
}

template <typename DT>
void H2MatrixSolver<DT>::factorizeDeviceM(deviceHandle_t handle) {
  copyDataInMatrixDesc(desc[levels], A[levels].A[0], A[levels].Q[0], handle->compute_stream);
  compute_factorize(handle, desc[levels], deviceMatrixDesc_t<DT>());

  for (long long l = levels - 1; l >= 0; l--) {
    copyDataInMatrixDesc(desc[l], A[l].A[0], A[l].Q[0], handle->memory_stream);
    cudaDeviceSynchronize();
    copyDataOutMatrixDesc(desc[l + 1], A[l + 1].A[0], A[l + 1].R[desc[l + 1].diag_offset], handle->memory_stream);
    compute_factorize(handle, desc[l], desc[l + 1]);
  }

  copyDataOutMatrixDesc(desc[0], A[0].A[0], A[0].R[0], handle->compute_stream);
  cudaDeviceSynchronize();

  for (long long l = levels; l >= 0; l--)
    if (check_info(desc[l], comm[l]))
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
void H2MatrixSolver<DT>::solvePreconditionDevice(deviceHandle_t handle, DT X[]) {
  if (levels < 0)
    return;

  long long lenX = A[levels].lenX;
  cudaMemcpy(X_dev, X, lenX * sizeof(DT), cudaMemcpyHostToDevice);
  matSolvePreconditionDeviceH2(handle, levels, desc.data(), X_dev);
  cudaMemcpy(X, X_dev, lenX * sizeof(DT), cudaMemcpyDeviceToHost);
}

// TODO switch to squared norm
template <typename DT>
void H2MatrixSolver<DT>::solveGMRES(double tol, H2MatrixSolver<DT>& M, DT x[], const DT b[], long long inner_iters, long long outer_iters) {
  typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> Vector_dt;
  typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;

  long long N = A[levels].lenX;
  long long ld = inner_iters + 1;

  Eigen::Map<const Vector_dt> B(b, N);
  Eigen::Map<Vector_dt> X(x, N);

  DT nsum = B.adjoint() * B;
  comm[levels].level_sum(&nsum, 1);
  double normb = std::sqrt(std::real(nsum));
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

    double beta = std::sqrt(std::real(nsum));
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
      H(i + 1, i) = std::sqrt(std::real(nsum));
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
    resid[++iters] = std::sqrt(std::real(nsum)) / normb;
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
  Vector_ot R_low = R.template cast<OT>();
  resid.resize(outer_iters + 1);
  resid[0] = 1.;
  iters = 0;

  while (iters < outer_iters && tol <= resid[iters]) {
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
void H2MatrixSolver<DT>::solveGMRESDevice(deviceHandle_t handle, double tol, H2MatrixSolver<DT>& M, DT X[], const DT B[], long long inner_iters, long long outer_iters, const ncclComms nccl_comms) {
  resid.resize(outer_iters + 1);
  iters = solveDeviceGMRES(handle, levels, A_mv.data(), M.levels, M.desc.data(), tol, X, B, inner_iters, outer_iters, resid.data(), comm[levels], nccl_comms);
}

template <typename DT>
void H2MatrixSolver<DT>::free_all_comms() {
  for (MPI_Comm& c : allocedComm)
    MPI_Comm_free(&c);
  allocedComm.clear();
}

template <typename DT>
void H2MatrixSolver<DT>::freeSparseMV() {
  for (long long l = 0; l <= levels; l++) {
    destroySpMatrixDesc(A_mv[l]);
  }
  A_mv.clear();
}

template <typename DT>
void H2MatrixSolver<DT>::free_gpu_handles() {
  for (long long l = levels; l >= 0; l--) {
    destroyMatrixDesc(desc[l]);
  }
  desc.clear();
  cudaFree(X_dev);
}

// Ma's version
/*template <typename DT>
double solveRelErr(long long lenX, const DT X[], const DT ref[], MPI_Comm world) {
  double err[2] = { 0., 0. };
  for (long long i = 0; i < lenX; i++) {
    DT diff = X[i] - ref[i];
    err[0] = err[0] + (std::real(diff) * std::real(diff));
    err[1] = err[1] + (std::real(ref[i]) * std::real(ref[i]));
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, world);
  return std::sqrt(err[0] / err[1]);
}*/

// Eigen version
template <typename DT>
double solveRelErr(long long lenX, const DT X[], const DT ref[], MPI_Comm world) {
  double err[2] = { 0., 0. };
  typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> Vector_dt;
  Eigen::Map<const Vector_dt> x1(X, lenX);
  Eigen::Map<const Vector_dt> x2(ref, lenX);
  Eigen::Matrix<DT, Eigen::Dynamic, 1> diff = x1 - x2;
  err[0] = diff.squaredNorm();
  err[1] = x2.squaredNorm();

  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, world);
  return std::sqrt(err[0] / err[1]);
}
