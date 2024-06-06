
#include <comm-nccl.hpp>
#include <algorithm>

ColCommNCCL::ColCommNCCL(const std::pair<long long, long long> Tree[], std::pair<long long, long long> Mapping[], const long long Rows[], const long long Cols[], MPI_Comm world) : 
  ColCommMPI(Tree, Mapping, Rows, Cols, world) {

  long long len = allocedComm.size();
  std::vector<ncclUniqueId> ids(len);
  allocedNCCL = std::vector<ncclComm_t>(len);

  ncclGroupStart();
  for (long long i = 0; i < len; i++) {
    int rank, size;
    MPI_Comm_rank(allocedComm[i], &rank);
    MPI_Comm_size(allocedComm[i], &size);
    if (rank == 0)
      ncclGetUniqueId(&ids[i]);
    MPI_Bcast((void*)&ids[i], sizeof(ncclUniqueId), MPI_BYTE, 0, allocedComm[i]);
    ncclCommInitRank(&allocedNCCL[i], size, ids[i], rank);
  }
  ncclGroupEnd();

  auto find_nccl = [&](const MPI_Comm& c) {
    const std::vector<MPI_Comm>::iterator i = std::find(allocedComm.begin(), allocedComm.end(), c);
    return i == allocedComm.end() ? nullptr : *(allocedNCCL.begin() + std::distance(allocedComm.begin(), i));
  };

  NeighborNCCL = std::vector<ncclComm_t>(NeighborComm.size(), nullptr);
  MergeNCCL = find_nccl(MergeComm.second);
  for (long long i = 0; i < (long long)NeighborComm.size(); i++)
    NeighborNCCL[i] = find_nccl(NeighborComm[i].second);
  AllReduceNCCL = find_nccl(AllReduceComm);
  DupNCCL = find_nccl(DupComm);
}

void ColCommNCCL::free_all_comms() {
  MergeComm = std::make_pair(0, MPI_COMM_NULL);
  NeighborComm.clear();
  AllReduceComm = MPI_COMM_NULL;
  DupComm = MPI_COMM_NULL;
  
  for (MPI_Comm& c : allocedComm)
    MPI_Comm_free(&c);
  allocedComm.clear();

  MergeNCCL = nullptr;
  NeighborNCCL.clear();
  AllReduceNCCL = nullptr;
  DupNCCL = nullptr;
  
  for (ncclComm_t& c : allocedNCCL)
    ncclCommDestroy(c);
  allocedNCCL.clear();
}
