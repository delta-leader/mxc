
#pragma once

#include <mpi.h>
#include <vector>
#include <complex>

template<class T> class MatrixDataContainer;

class ColCommMPI {
protected:
  long long Proc;
  std::vector<std::pair<long long, long long>> Boxes;
  std::vector<long long> BoxOffsets;
  
  std::vector<std::pair<int, MPI_Comm>> NeighborComm;
  MPI_Comm MergeComm;
  MPI_Comm AllReduceComm;
  MPI_Comm DupComm;

  template<class T> inline void level_merge(T* data, long long len) const;
  template<class T> inline void level_sum(T* data, long long len) const;
  template<class T> inline void neighbor_bcast(T* data) const;
  template<class T> inline void neighbor_bcast(T* data, const long long noffsets[]) const;
  template<class T> inline void neighbor_bcast(MatrixDataContainer<T>& dc, const long long noffsets[]) const;

public:
  ColCommMPI() : Proc(-1), Boxes(), NeighborComm(), MergeComm(MPI_COMM_NULL), AllReduceComm(MPI_COMM_NULL), DupComm(MPI_COMM_NULL) {};
  ColCommMPI(const std::pair<long long, long long> Tree[], std::pair<long long, long long> Mapping[], const long long Rows[], const long long Cols[], std::vector<MPI_Comm>& allocedComm, MPI_Comm world = MPI_COMM_WORLD);
  
  long long iLocal(long long iglobal) const;
  long long iGlobal(long long ilocal) const;
  long long oLocal() const;
  long long oGlobal() const;
  long long lenLocal() const;
  long long lenNeighbors() const;
  long long dataSizesToNeighborOffsets(long long Dims[]) const;

  void level_merge(std::complex<double>* data, long long len) const;
  void level_sum(std::complex<double>* data, long long len) const;

  void neighbor_bcast(long long* data) const;
  void neighbor_bcast(MatrixDataContainer<double>& dc) const;
  void neighbor_bcast(MatrixDataContainer<std::complex<double>>& dc) const;

  static double get_comm_time();
  static void record_mpi();
};

