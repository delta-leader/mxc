
#pragma once

#include <mpi.h>
#include <vector>
#include <complex>

template<class T> class MatrixDataContainer;

class ColCommMPI {
protected:
  long long Proc;
  std::vector<std::pair<long long, long long>> Boxes;
  
  std::vector<std::pair<int, MPI_Comm>> NeighborComm;
  MPI_Comm MergeComm;
  MPI_Comm AllReduceComm;
  MPI_Comm DupComm;
  std::vector<MPI_Comm> allocedComm;

  template<class T> inline void level_merge(T* data, long long len) const;
  template<class T> inline void level_sum(T* data, long long len) const;
  template<class T> inline void neighbor_bcast(T* data) const;
  template<class T> inline void neighbor_bcast(MatrixDataContainer<T>& dc) const;

public:
  std::pair<double, double>* timer;

  ColCommMPI() : Proc(-1), Boxes(), NeighborComm(), MergeComm(MPI_COMM_NULL), AllReduceComm(MPI_COMM_NULL), DupComm(MPI_COMM_NULL), allocedComm(), timer(nullptr) {};
  ColCommMPI(const std::pair<long long, long long> Tree[], std::pair<long long, long long> Mapping[], const long long Rows[], const long long Cols[], MPI_Comm world = MPI_COMM_WORLD);
  
  long long iLocal(long long iglobal) const;
  long long iGlobal(long long ilocal) const;
  long long oLocal() const;
  long long oGlobal() const;
  long long lenLocal() const;
  long long lenNeighbors() const;

  void level_merge(std::complex<double>* data, long long len) const;
  void level_sum(std::complex<double>* data, long long len) const;

  void neighbor_bcast(long long* data) const;
  void neighbor_bcast(MatrixDataContainer<double>& dc) const;
  void neighbor_bcast(MatrixDataContainer<std::complex<double>>& dc) const;

  void record_mpi() const;

  void free_all_comms();
};

