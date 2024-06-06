
#include <comm-mpi.hpp>

#include <cuda_runtime_api.h>
#include <nccl.h>

class ColCommNCCL : public ColCommMPI {
private:
  ncclComm_t MergeNCCL;
  std::vector<ncclComm_t> NeighborNCCL;
  ncclComm_t AllReduceNCCL;
  ncclComm_t DupNCCL;
  std::vector<ncclComm_t> allocedNCCL;

public:
  ColCommNCCL() : ColCommMPI(), MergeNCCL(nullptr), NeighborNCCL(), AllReduceNCCL(nullptr), DupNCCL(nullptr), allocedNCCL() {};
  ColCommNCCL(const std::pair<long long, long long> Tree[], std::pair<long long, long long> Mapping[], const long long Rows[], const long long Cols[], MPI_Comm world);

  void free_all_comms();
};

