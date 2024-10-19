
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
    *nccl_comms = new std::vector<std::pair<MPI_Comm, ncclComm_t>>();

  std::vector<MPI_Comm> allocated((*nccl_comms)->size()), comms(mpi_comms.size());
  std::transform((*nccl_comms)->begin(), (*nccl_comms)->end(), allocated.begin(), [](std::pair<MPI_Comm, ncclComm_t> x) { return x.first; });
  
  auto comms_end = std::copy_if(mpi_comms.begin(), mpi_comms.end(), comms.begin(), [&](MPI_Comm comm) { 
    return allocated.end() == std::find_if(allocated.begin(), allocated.end(), [=](MPI_Comm c) { 
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

  (*nccl_comms)->reserve((*nccl_comms)->size() + len);
  for (long long i = 0; i < len; i++)
    (*nccl_comms)->emplace_back(comms[i], nccl_alloc[i]);
}

ncclComm_t findNcclComm(const MPI_Comm mpi_comm, const ncclComms nccl_comms) {
  if (mpi_comm == MPI_COMM_NULL)
    return nullptr;
  auto iter = std::find_if(nccl_comms->begin(), nccl_comms->end(), [=](const std::pair<MPI_Comm, ncclComm_t>& comm) { 
    int result; MPI_Comm_compare(mpi_comm, comm.first, &result); return result == MPI_IDENT || result == MPI_CONGRUENT; });
  return iter == nccl_comms->end() ? nullptr : (*iter).second;
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
