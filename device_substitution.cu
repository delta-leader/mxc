
#include <factorize.cuh>
#include <comm-mpi.hpp>

void compute_forward_substitution(devicePreconditioner_t A, devicePreconditioner_t Al, cudaStream_t stream, cublasHandle_t cublasH, const ColCommMPI& comm, const std::map<const MPI_Comm, ncclComm_t>& nccl_comms) {
  long long bdim = A.bdim;
  long long rank = A.rank;
  long long block = bdim * bdim;
  long long rblock = rank * rank;

  long long D = comm.oLocal();
  long long M = comm.lenLocal();
  long long N = comm.lenNeighbors();
  long long lenA = comm.ARowOffsets[M];
}
