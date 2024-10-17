#pragma once

#include <vector>
#include <map>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <mpi.h>
#include <nccl.h>

struct deviceHandle {
  cudaStream_t memory_stream = nullptr;
  cudaStream_t compute_stream = nullptr;
  cublasHandle_t cublasH = nullptr;
  cusparseHandle_t cusparseH = nullptr;
  cusolverDnHandle_t cusolverH = nullptr;
};

typedef struct deviceHandle* deviceHandle_t;
typedef std::map<MPI_Comm, ncclComm_t>* ncclComms;

void cudaSetDevice(MPI_Comm world = MPI_COMM_WORLD);
void initNcclComms(ncclComms* nccl_comms, const std::vector<MPI_Comm>& mpi_comms);
void initGpuEnvs(deviceHandle_t* handle);
void finalizeGpuEnvs(deviceHandle_t handle);
void finalizeNcclComms(ncclComms nccl_comms);
