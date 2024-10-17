
#include <factorize.cuh>
#include <comm-mpi.hpp>

void compute_forward_substitution(deviceHandle_t handle, deviceMatrixDesc_t A, const CUDA_CTYPE* X, const ColCommMPI& comm, const ncclComms nccl_comms) {
  long long bdim = A.bdim;
  long long rank = A.rank;
  long long rdim = bdim - rank;
  long long block = bdim * bdim;

  long long D = A.diag_offset;
  long long M = comm.lenLocal();
  long long N = comm.lenNeighbors();
  long long lenA = comm.ARowOffsets[M];
  long long reduc_len = A.reducLen;

  cudaStream_t stream = handle->compute_stream;
  cublasHandle_t cublasH = handle->cublasH;

  STD_CTYPE constants[3] = { 1., 0., -1. };
  CUDA_CTYPE& one = reinterpret_cast<CUDA_CTYPE&>(constants[0]);
  CUDA_CTYPE& zero = reinterpret_cast<CUDA_CTYPE&>(constants[1]); 
  CUDA_CTYPE& minus_one = reinterpret_cast<CUDA_CTYPE&>(constants[2]); 

  cublasZgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, bdim, 1, bdim, &one, A.Vdata, bdim, block, &X[A.lower_offset], bdim, bdim, &zero, &(A.Ydata)[D * bdim], bdim, bdim, M);

  if (1 < N) {
    ncclGroupStart();
    for (long long p = 0; p < (long long)comm.NeighborComm.size(); p++) {
      long long start = comm.BoxOffsets[p] * bdim;
      long long len = comm.BoxOffsets[p + 1] * bdim - start;
      auto neighbor = nccl_comms->find(comm.NeighborComm[p].second);
      ncclBroadcast(const_cast<const CUDA_CTYPE*>(&(A.Ydata)[start]), &(A.Ydata)[start], len * 2, ncclDouble, comm.NeighborComm[p].first, (*neighbor).second, stream);
    }

    auto dup = nccl_comms->find(comm.DupComm);
    if (comm.DupComm != MPI_COMM_NULL)
      ncclBroadcast(const_cast<const CUDA_CTYPE*>(A.Ydata), A.Ydata, bdim * N * 2, ncclDouble, 0, (*dup).second, stream);
    ncclGroupEnd();
  }

  size_t sizeX = rank * sizeof(CUDA_CTYPE);
  cudaMemcpy2DAsync(&(A.Xdata)[D * rank], sizeX, &(A.Ydata)[D * bdim], bdim * sizeof(CUDA_CTYPE), sizeX, M, cudaMemcpyDeviceToDevice, stream);
  if (0 < rank && 0 < rdim) {
    cudaMemsetAsync(A.ACdata, 0, reduc_len * M * rank * sizeof(CUDA_CTYPE), stream);
    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, 1, rdim, &minus_one, A.A_sr, bdim, A.Y_R_cols, bdim, &zero, A.AC_X, rank, lenA);
    cublasZgemv(cublasH, CUBLAS_OP_N, M * rank, reduc_len, &one, A.ACdata, M * rank, A.ONEdata, 1, &one, &(A.Xdata)[D * rank], 1);
  }

  if (1 < N) {
    ncclGroupStart();
    for (long long p = 0; p < (long long)comm.NeighborComm.size(); p++) {
      long long start = comm.BoxOffsets[p] * rank;
      long long len = comm.BoxOffsets[p + 1] * rank - start;
      auto neighbor = nccl_comms->find(comm.NeighborComm[p].second);
      ncclBroadcast(const_cast<const CUDA_CTYPE*>(&(A.Xdata)[start]), &(A.Xdata)[start], len * 2, ncclDouble, comm.NeighborComm[p].first, (*neighbor).second, stream);
    }

    auto dup = nccl_comms->find(comm.DupComm);
    if (comm.DupComm != MPI_COMM_NULL)
      ncclBroadcast(const_cast<const CUDA_CTYPE*>(A.Xdata), A.Xdata, rank * N * 2, ncclDouble, 0, (*dup).second, stream);
    ncclGroupEnd();
  }
}

void compute_backward_substitution(deviceHandle_t handle, deviceMatrixDesc_t A, CUDA_CTYPE* X, const ColCommMPI& comm, const ncclComms nccl_comms) {
  long long bdim = A.bdim;
  long long rank = A.rank;
  long long rdim = bdim - rank;
  long long block = bdim * bdim;

  long long D = A.diag_offset;
  long long M = comm.lenLocal();
  long long N = comm.lenNeighbors();
  long long lenA = comm.ARowOffsets[M];
  long long reduc_len = A.reducLen;

  cudaStream_t stream = handle->compute_stream;
  cublasHandle_t cublasH = handle->cublasH;

  STD_CTYPE constants[3] = { 1., 0., -1. };
  CUDA_CTYPE& one = reinterpret_cast<CUDA_CTYPE&>(constants[0]);
  CUDA_CTYPE& zero = reinterpret_cast<CUDA_CTYPE&>(constants[1]); 
  CUDA_CTYPE& minus_one = reinterpret_cast<CUDA_CTYPE&>(constants[2]); 

  if (1 < N) {
    ncclGroupStart();
    for (long long p = 0; p < (long long)comm.NeighborComm.size(); p++) {
      long long start = comm.BoxOffsets[p] * rank;
      long long len = comm.BoxOffsets[p + 1] * rank - start;
      auto neighbor = nccl_comms->find(comm.NeighborComm[p].second);
      ncclBroadcast(const_cast<const CUDA_CTYPE*>(&(A.Xdata)[start]), &(A.Xdata)[start], len * 2, ncclDouble, comm.NeighborComm[p].first, (*neighbor).second, stream);
    }

    auto dup = nccl_comms->find(comm.DupComm);
    if (comm.DupComm != MPI_COMM_NULL)
      ncclBroadcast(const_cast<const CUDA_CTYPE*>(A.Xdata), A.Xdata, rank * N * 2, ncclDouble, 0, (*dup).second, stream);
    ncclGroupEnd();
  }

  size_t sizeX = rank * sizeof(CUDA_CTYPE);
  cudaMemcpy2DAsync(&(A.Ydata)[D * bdim], bdim * sizeof(CUDA_CTYPE), &(A.Xdata)[D * rank], sizeX, sizeX, M, cudaMemcpyDeviceToDevice, stream);
  if (0 < rank && 0 < rdim) {
    cudaMemsetAsync(A.ACdata, 0, reduc_len * M * bdim * sizeof(CUDA_CTYPE), stream);
    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rdim, 1, rank, &minus_one, A.A_rs, bdim, A.X_cols, bdim, &zero, A.AC_X_R, bdim, lenA);
    cublasZgemv(cublasH, CUBLAS_OP_N, M * bdim, reduc_len, &one, A.ACdata, M * bdim, A.ONEdata, 1, &one, &(A.Ydata)[D * bdim], 1);
  }

  cublasZgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_C, 1, bdim, bdim, &one, &(A.Ydata)[D * bdim], 1, bdim, &(A.Udata)[D * block], bdim, block, &zero, &X[A.lower_offset], 1, bdim, M);
}
