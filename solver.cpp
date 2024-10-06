
#include <solver.hpp>

#include <mkl.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

H2MatrixSolver::H2MatrixSolver() : levels(-1), A(), comm(), allocedComm(), local_bodies(0, 0) {
}

H2MatrixSolver::H2MatrixSolver(const MatrixAccessor& eval, double epi, long long rank, long long leveled_rank, const std::vector<Cell>& cells, double theta, const double bodies[], long long levels, MPI_Comm world) : 
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
  std::vector<WellSeparatedApproximation> wsa(levels + 1);
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

void H2MatrixSolver::init_gpu_handles(MPI_Comm world) {
  initGpuEnvs(&memory_stream, &compute_stream, &cublasH, nccl_comms, allocedComm, world);
  
  desc.resize(levels + 1);
  long long bdim = *std::max_element(A[levels].Dims.begin(), A[levels].Dims.end());
  long long rank = *std::max_element(A[levels].DimsLr.begin(), A[levels].DimsLr.end());
  createMatrixDesc(&desc[levels], bdim, rank, deviceMatrixDesc_t(), comm[levels]);

  for (long long l = levels - 1; l >= 0; l--) {
    long long bdim = *std::max_element(A[l].Dims.begin(), A[l].Dims.end());
    long long rank = *std::max_element(A[l].DimsLr.begin(), A[l].DimsLr.end());
    createMatrixDesc(&desc[l], bdim, rank, desc[l + 1], comm[l]);
  }

  long long lenX = bdim * comm[levels].lenLocal();
  cudaMalloc(reinterpret_cast<void**>(&X_dev), lenX * sizeof(CUDA_CTYPE));
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
  { A[l].X.reset(); A[l].Y.reset(); }
  X_leaf = X_in;

  for (long long l = levels; l >= 0; l--) {
    A[l].matVecUpwardPass(comm[l]);
    if (0 < l)
      A[l - 1].upwardCopyNext('X', 'X', comm[l - 1], A[l]);
  }
  for (long long l = 1; l <= levels; l++) {
    A[l].downwardCopyNext('Y', 'Y', A[l - 1], comm[l - 1]);
    A[l].matVecHorizontalandDownwardPass(A[l - 1], comm[l]);
  }

  X_leaf = X_in;
  A[levels].matVecLeafHorizontalPass(comm[levels]);

  X_in = Y_leaf;
}

void H2MatrixSolver::factorizeM() {
  for (long long l = levels; l >= 0; l--) {
    A[l].factorize(comm[l]);
    if (0 < l)
      A[l - 1].factorizeCopyNext(A[l], comm[l]);
  }
}

void H2MatrixSolver::factorizeDeviceM() {
  copyDataInMatrixDesc(desc[levels], comm[levels].ARowOffsets.back(), A[levels].A[0], comm[levels].lenNeighbors(), A[levels].Q[0], compute_stream);
  compute_factorize(desc[levels], deviceMatrixDesc_t(), compute_stream, cublasH, comm[levels], nccl_comms);

  for (long long l = levels - 1; l >= 0; l--) {
    copyDataInMatrixDesc(desc[l], comm[l].ARowOffsets.back(), A[l].A[0], comm[l].lenNeighbors(), A[l].Q[0], memory_stream);
    cudaDeviceSynchronize();
    copyDataOutMatrixDesc(desc[l + 1], comm[l + 1].ARowOffsets.back(), A[l + 1].A[0], comm[l + 1].lenLocal(), A[l + 1].R[desc[l + 1].diag_offset], memory_stream);
    compute_factorize(desc[l], desc[l + 1], compute_stream, cublasH, comm[l], nccl_comms);
  }

  copyDataOutMatrixDesc(desc[0], 1, A[0].A[0], 1, A[0].R[0], compute_stream);
  cudaDeviceSynchronize();
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
  Vector_t Y_leaf(A[levels].Y[lbegin], lenX);

  X_leaf = X_in;
  A[levels].forwardSubstitute(comm[levels]);
  for (long long l = levels - 1; l >= 0; l--) {
    A[l].upwardCopyNext('X', 'X', comm[l], A[l + 1]);
    A[l].forwardSubstitute(comm[l]);
  }

  for (long long l = 0; l < levels; l++) {
    A[l].backwardSubstitute(comm[l]);
    A[l + 1].downwardCopyNext('Y', 'Y', A[l], comm[l]);
  }
  A[levels].backwardSubstitute(comm[levels]);
  X_in = Y_leaf;
}

void H2MatrixSolver::solvePreconditionDevice(std::complex<double> X[]) {
  if (levels < 0)
    return;
  
  long long lbegin = comm[levels].oLocal();
  long long llen = comm[levels].lenLocal();
  long long lenX = std::reduce(A[levels].Dims.begin() + lbegin, A[levels].Dims.begin() + (lbegin + llen));
  
  cudaMemcpy(X_dev, X, lenX * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

  compute_forward_substitution(desc[levels], X_dev, compute_stream, cublasH, comm[levels], nccl_comms);
  for (long long l = levels - 1; l >= 0; l--)
    compute_forward_substitution(desc[l], desc[l + 1].Xdata, compute_stream, cublasH, comm[l], nccl_comms);

  for (long long l = 0; l < levels; l++)
    compute_backward_substitution(desc[l], desc[l + 1].Ydata, compute_stream, cublasH, comm[l], nccl_comms);
  compute_backward_substitution(desc[levels], X_dev, compute_stream, cublasH, comm[levels], nccl_comms);
  
  cudaMemcpy(X, X_dev, lenX * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
}

void H2MatrixSolver::solveGMRES(double tol, H2MatrixSolver& M, std::complex<double> x[], const std::complex<double> b[], long long inner_iters, long long outer_iters) {
  using Eigen::VectorXcd, Eigen::MatrixXcd;

  long long lbegin = comm[levels].oLocal();
  long long llen = comm[levels].lenLocal();
  long long N = std::reduce(A[levels].Dims.begin() + lbegin, A[levels].Dims.begin() + (lbegin + llen));
  long long ld = inner_iters + 1;

  Eigen::Map<const Eigen::VectorXcd> B(b, N);
  Eigen::Map<Eigen::VectorXcd> X(x, N);

  std::complex<double> normb_sum = B.adjoint() * B;
  comm[levels].level_sum(&normb_sum, 1);
  double normb = std::sqrt(normb_sum.real());
  if (normb == 0.)
    normb = 1.;

  VectorXcd R(N);
  resid = std::vector<double>(outer_iters + 1, 0.);

  for (iters = 0; iters < outer_iters; iters++) {
    std::pair<std::complex<double>, std::complex<double>> normr_sum;
    R = -X;
    matVecMul(R.data());
    R += B;
    normr_sum.first = R.adjoint() * R;

    M.solvePrecondition(R.data());
    normr_sum.second = R.adjoint() * R;
    comm[levels].level_sum(reinterpret_cast<std::complex<double>*>(&normr_sum), 2);

    resid[iters] = std::sqrt(normr_sum.first.real()) / normb;
    if (resid[iters] < tol)
      return;

    double beta = std::sqrt(normr_sum.second.real());
    MatrixXcd H = MatrixXcd::Zero(ld, inner_iters);
    MatrixXcd v = MatrixXcd::Zero(N, ld);
    v.col(0) = R * (1. / beta);
    
    for (long long i = 0; i < inner_iters; i++) {
      VectorXcd w = v.col(i);
      matVecMul(w.data());
      M.solvePrecondition(w.data());

      H.block(0, i, i + 1, 1).noalias() = v.leftCols(i + 1).adjoint() * w;
      comm[levels].level_sum(H.col(i).data(), i + 1);
      w.noalias() -= v.leftCols(i + 1) * H.block(0, i, i + 1, 1);

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

  normb_sum = R.adjoint() * R;
  comm[levels].level_sum(&normb_sum, 1);
  resid[outer_iters] = std::sqrt(normb_sum.real()) / normb;
}

void H2MatrixSolver::free_all_comms() {
  for (MPI_Comm& c : allocedComm)
    MPI_Comm_free(&c);
  allocedComm.clear();
}

void H2MatrixSolver::free_gpu_handles() {
  finalizeGpuEnvs(memory_stream, compute_stream, cublasH, nccl_comms);
  for (long long l = levels; l >= 0; l--) {
    destroyMatrixDesc(desc[l]);
  }
  desc.clear();
  cudaFree(X_dev);
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
