
#pragma once

#include <mpi.h>
#include <vector>
#include <complex>
#include <tuple>

class ColCommMPI {
public:
  long long Proc;
  std::vector<std::pair<long long, long long>> Boxes;
  std::vector<long long> BoxOffsets;
  
  std::vector<std::pair<int, MPI_Comm>> NeighborComm;
  MPI_Comm MergeComm;
  MPI_Comm AllReduceComm;
  MPI_Comm DupComm;

  std::vector<long long> ARowOffsets;
  std::vector<long long> AColumns;
  std::vector<long long> CRowOffsets;
  std::vector<long long> CColumns;

  long long LowerX;
  std::vector<std::tuple<long long, long long, long long>> LowerIndA;
  std::vector<std::tuple<long long, long long, long long>> LowerIndC;

  ColCommMPI(const std::pair<long long, long long> Tree[], std::pair<long long, long long> Mapping[], const long long ARows[], const long long ACols[], const long long CRows[], const long long CCols[], std::vector<MPI_Comm>& allocedComm, MPI_Comm world = MPI_COMM_WORLD);
  ColCommMPI(const ColCommMPI& comm, const std::vector<MPI_Comm>& allocedComm);
  
  long long iLocal(long long iglobal) const;
  long long iGlobal(long long ilocal) const;
  long long oLocal() const;
  long long oGlobal() const;
  long long lenLocal() const;
  long long lenNeighbors() const;
  long long dataSizesToNeighborOffsets(long long Dims[]) const;

  template <typename DT>
  void level_merge(DT* data, long long len) const;
  template <typename DT>
  void level_sum(DT* data, long long len) const;
  template <typename DT>
  void neighbor_bcast(DT data[], const long long noffsets[]) const;

  static double get_comm_time();
  static void record_mpi();
};

