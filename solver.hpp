#pragma once

#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <h2matrix.hpp>
#include <kernel.hpp>
#include <device_factorize.cuh>
#include <device_csr_matrix.cuh>

template <typename DT = std::complex<double>>
class H2MatrixSolver {
public:
  long long levels;
  std::vector<H2Matrix<DT>> A;
  std::vector<ColCommMPI> comm;
  std::vector<MPI_Comm> allocedComm;

  std::vector<CsrMatVecDesc_t> A_mv;

  std::vector<deviceMatrixDesc_t<DT>> desc;
  DT* X_dev;
  //CUDA_CTYPE* X_dev;

  std::pair<long long, long long> local_bodies;
  std::vector<double> resid;
  long long iters;
  
  H2MatrixSolver();
  H2MatrixSolver(const MatrixAccessor<DT>& eval, double epi, long long rank, long long leveled_rank, const std::vector<Cell>& cells, double theta, const double bodies[], long long levels, MPI_Comm world = MPI_COMM_WORLD);
  void init_gpu_handles(const ncclComms nccl_comms);
  void move_data_gpu();

  void allocSparseMV(deviceHandle_t handle, const ncclComms nccl_comms);
  void matVecMulSp(deviceHandle_t handle, DT X[]);

  void matVecMul(DT X[]);
  void factorizeM();
  void factorizeDeviceM(deviceHandle_t handle);
  void solvePrecondition(DT X[]);
  void solvePreconditionDevice(deviceHandle_t handle, DT X[]);
  void solveGMRES(double tol, H2MatrixSolver<DT>& M, DT X[], const DT B[], long long inner_iters, long long outer_iters);
  void solveGMRESDevice(deviceHandle_t handle, double tol, H2MatrixSolver<DT>& M, DT X[], const DT B[], long long inner_iters, long long outer_iters, const ncclComms nccl_comms);

  void free_all_comms();
  void freeSparseMV();
  void free_gpu_handles();
};

template <typename DT>
  double solveRelErr(long long lenX, const DT X[], const DT ref[], MPI_Comm world = MPI_COMM_WORLD);
