
#pragma once

#include <mpi.h>
#include <vector>
#include <complex>

#ifdef USE_NCCL
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <map>
#endif

class CellComm {
private:
  long long Proc;
  std::vector<std::pair<long long, long long>> Boxes;
  
  std::pair<int, MPI_Comm> MergeComm;
  std::vector<std::pair<int, MPI_Comm>> NeighborComm;
  MPI_Comm AllReduceComm;
  MPI_Comm DupComm;

  template<typename T> inline void level_merge(T* data, long long len) const;
  template<typename T> inline void level_sum(T* data, long long len) const;
  template<typename T> inline void neighbor_bcast(T* data, const long long box_dims[]) const;
  template<typename T> inline void neighbor_reduce(T* data, const long long box_dims[]) const;

#ifdef USE_NCCL
  ncclComm_t MergeNCCL = nullptr;
  std::vector<ncclComm_t> NeighborNCCL;
  ncclComm_t AllReduceNCCL = nullptr;
  ncclComm_t DupNCCL = nullptr;
#endif

public:
  std::pair<double, double>* timer;

  CellComm() : Proc(-1), Boxes(), MergeComm(0, MPI_COMM_NULL), NeighborComm(), AllReduceComm(MPI_COMM_NULL), DupComm(MPI_COMM_NULL), timer(nullptr) {};
  CellComm(const std::pair<long long, long long> Tree[], std::pair<long long, long long> Mapping[], const long long Rows[], const long long Cols[], std::vector<MPI_Comm>& unique_comms, MPI_Comm world);
  
  long long iLocal(long long iglobal) const;
  long long iGlobal(long long ilocal) const;
  long long oLocal() const;
  long long oGlobal() const;
  long long lenLocal() const;
  long long lenNeighbors() const;

  void level_merge(std::complex<double>* data, long long len) const;
  void level_sum(std::complex<double>* data, long long len) const;

  void neighbor_bcast(long long* data, const long long box_dims[]) const;
  void neighbor_bcast(double* data, const long long box_dims[]) const;
  void neighbor_bcast(std::complex<double>* data, const long long box_dims[]) const;

  void neighbor_reduce(long long* data, const long long box_dims[]) const;
  void neighbor_reduce(std::complex<double>* data, const long long box_dims[]) const;

  void record_mpi() const;

  static void free_mpi_comms(std::vector<MPI_Comm>& unique_comms);

#ifdef USE_NCCL
  void set_nccl_communicators(const std::map<MPI_Comm, ncclComm_t>& unique_comms);

  static std::map<MPI_Comm, ncclComm_t> create_nccl_communicators(const std::vector<MPI_Comm>& unique_comms);
  static void free_nccl_comms(std::map<MPI_Comm, ncclComm_t>& unique_comms);
#endif
};

