
#include <gpu_handles.cuh>
#include <algorithm>

void cudaSetDevice(MPI_Comm world) {
  int mpi_rank, num_device;
  if (cudaGetDeviceCount(&num_device) != cudaSuccess)
    return;
  MPI_Comm_rank(world, &mpi_rank);
  cudaSetDevice(mpi_rank % num_device);
  cudaDeviceReset();
}

void initNcclComms(ncclComms* nccl_comms, const std::vector<MPI_Comm>& mpi_comms) {
  if (*nccl_comms == nullptr)
    *nccl_comms = new std::map<MPI_Comm, ncclComm_t>();

  std::vector<MPI_Comm> allocated((*nccl_comms)->size()), comms(mpi_comms.size());
  std::transform((*nccl_comms)->begin(), (*nccl_comms)->end(), allocated.begin(), [](std::pair<MPI_Comm, ncclComm_t> x) { return x.first; });
  
  auto comms_end = std::copy_if(mpi_comms.begin(), mpi_comms.end(), comms.begin(), [&](MPI_Comm comm) { 
    return allocated.end() == std::find_if(allocated.begin(), allocated.end(), [comm](MPI_Comm c) { 
      int result; MPI_Comm_compare(comm, c, &result); return result == MPI_IDENT || result == MPI_CONGRUENT; }); });

  long long len = std::distance(comms.begin(), comms_end);
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
    (*nccl_comms)->insert(std::make_pair(comms[i], nccl_alloc[i]));
}

void initGpuEnvs(deviceHandle_t* handle) {
  *handle = new deviceHandle();
  cudaStreamCreate(&((*handle)->memory_stream));
  cudaStreamCreate(&((*handle)->compute_stream));
  cublasCreate(&((*handle)->cublasH));
  cublasSetStream((*handle)->cublasH, (*handle)->compute_stream);
  cusparseCreate(&((*handle)->cusparseH));
  cusparseSetStream((*handle)->cusparseH, (*handle)->compute_stream);
  cusolverDnCreate(&((*handle)->cusolverH));
  cusolverDnSetStream((*handle)->cusolverH, (*handle)->compute_stream);
}

void finalizeGpuEnvs(deviceHandle_t handle) {
  cudaDeviceSynchronize();
  cudaStreamDestroy(handle->memory_stream);
  cudaStreamDestroy(handle->compute_stream);
  cublasDestroy(handle->cublasH);
  cusparseDestroy(handle->cusparseH);
  cusolverDnDestroy(handle->cusolverH);
  delete handle;
}

void finalizeNcclComms(ncclComms nccl_comms) {
  for (auto& c : *nccl_comms)
    ncclCommDestroy(c.second);
  nccl_comms->clear();
  delete nccl_comms;
}
