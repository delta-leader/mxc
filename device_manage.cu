
#include <factorize.cuh>

void initGpuEnvs(cudaStream_t* memory_stream, cudaStream_t* compute_stream, cublasHandle_t* cublasH, cusparseHandle_t* cusparseH, cusolverDnHandle_t* cusolverH, std::map<const MPI_Comm, ncclComm_t>& nccl_comms, const std::vector<MPI_Comm>& comms, MPI_Comm world) {
  int mpi_rank, num_device;
  if (cudaGetDeviceCount(&num_device) != cudaSuccess)
    return;

  MPI_Comm_rank(world, &mpi_rank);
  cudaSetDevice(mpi_rank % num_device);
  cudaStreamCreate(memory_stream);
  cudaStreamCreate(compute_stream);
  cublasCreate(cublasH);
  cublasSetStream(*cublasH, *compute_stream);
  cusparseCreate(cusparseH);
  cusparseSetStream(*cusparseH, *compute_stream);
  cusolverDnCreate(cusolverH);
  cusolverDnSetStream(*cusolverH, *compute_stream);

  long long len = comms.size();
  std::vector<ncclUniqueId> ids(len);
  std::vector<ncclComm_t> nccl_alloc(len);

  ncclGroupStart();
  for (long long i = 0; i < len; i++) {
    int rank, size;
    MPI_Comm_rank(comms[i], &rank);
    MPI_Comm_size(comms[i], &size);
    if (rank == 0)
      ncclGetUniqueId(&ids[i]);
    MPI_Bcast(reinterpret_cast<void*>(&ids[i]), sizeof(ncclUniqueId), MPI_BYTE, 0, comms[i]);
    ncclCommInitRank(&nccl_alloc[i], size, ids[i], rank);
  }
  ncclGroupEnd();

  for (long long i = 0; i < len; i++)
    nccl_comms.insert(std::make_pair(comms[i], nccl_alloc[i]));
}

void finalizeGpuEnvs(cudaStream_t memory_stream, cudaStream_t compute_stream, cublasHandle_t cublasH, cusparseHandle_t cusparseH, cusolverDnHandle_t cusolverH, std::map<const MPI_Comm, ncclComm_t>& nccl_comms) {
  cudaDeviceSynchronize();
  cudaStreamDestroy(memory_stream);
  cudaStreamDestroy(compute_stream);
  cublasDestroy(cublasH);
  cusparseDestroy(cusparseH);
  cusolverDnDestroy(cusolverH);
  for (auto& c : nccl_comms)
    ncclCommDestroy(c.second);
  nccl_comms.clear();
  cudaDeviceReset();
}
