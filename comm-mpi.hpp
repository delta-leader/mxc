
#pragma once

#include <mpi.h>
#include <vector>
#include <complex>

template<class T> class MatrixDataContainer;

class ColCommMPI {
protected:
  // number of processes in the communicator
  long long Proc;
  std::vector<std::pair<long long, long long>> Boxes;
  
  // contains all the other procsses on the same level
  std::vector<std::pair<int, MPI_Comm>> NeighborComm;
  MPI_Comm MergeComm;
  MPI_Comm AllReduceComm;
  MPI_Comm DupComm;

  template<class T> inline void level_merge(T* data, long long len) const;
  template<class T> inline void level_sum(T* data, long long len) const;
  template<class T> inline void neighbor_bcast(T* data) const;
  template<class T> inline void neighbor_bcast(MatrixDataContainer<T>& dc) const;

public:
  ColCommMPI() : Proc(-1), Boxes(), NeighborComm(), MergeComm(MPI_COMM_NULL), AllReduceComm(MPI_COMM_NULL), DupComm(MPI_COMM_NULL) {};
  /*
  Tree: cluster tree containing the pairs of children
  Mapping: contains a pair of (0,1) for each mpi process
  Rows: The RowIndex variable from the Neighbor CSR
  Columns: The ColumnIndex variable from the Neighbor CSR
  allocComm: empty
  world: MPI world communicator
  */
  ColCommMPI(const std::pair<long long, long long> Tree[], std::pair<long long, long long> Mapping[], const long long Rows[], const long long Cols[], std::vector<MPI_Comm>& allocedComm, MPI_Comm world = MPI_COMM_WORLD);
  
  long long iLocal(long long iglobal) const;
  long long iGlobal(long long ilocal) const;
  long long oLocal() const;
  // the first index in the cell array (on the current level)
  long long oGlobal() const;
  // the number of cell on the current level
  long long lenLocal() const;
  long long lenNeighbors() const;

  void level_merge(std::complex<double>* data, long long len) const;
  void level_sum(std::complex<double>* data, long long len) const;

  void neighbor_bcast(long long* data) const;
  void neighbor_bcast(MatrixDataContainer<double>& dc) const;
  void neighbor_bcast(MatrixDataContainer<std::complex<double>>& dc) const;

  static double get_comm_time();
  static void record_mpi();
};

