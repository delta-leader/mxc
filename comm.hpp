
#pragma once

#include <mpi.h>
#include <vector>
#include <complex>

class CSR;
class Cell;

class CellComm {
private:
  long long Proc;
  std::vector<std::pair<long long, long long>> Boxes;
  
  std::vector<std::pair<int, MPI_Comm>> NeighborComm;
  MPI_Comm DupComm;
  MPI_Comm MergeComm;

  template<typename T> inline void level_merge(T* data, long long len) const;
  template<typename T> inline void dup_bast(T* data, long long len) const;
  template<typename T> inline void neighbor_bcast(T* data, const long long box_dims[]) const;
  template<typename T> inline void neighbor_reduce(T* data, const long long box_dims[]) const;

public:
  std::pair<double, double>* timer;

  CellComm() : Proc(-1), Boxes(), NeighborComm(), DupComm(MPI_COMM_NULL), MergeComm(MPI_COMM_NULL), timer(nullptr) {};
  CellComm(const Cell cells[], std::pair<long long, long long> Mapping[], const CSR& Near, const CSR& Far, std::vector<MPI_Comm>& unique_comms, MPI_Comm world);
  
  long long iLocal(long long iglobal) const;
  long long iGlobal(long long ilocal) const;
  long long oLocal() const;
  long long oGlobal() const;
  long long lenLocal() const;
  long long lenNeighbors() const;

  void level_merge(std::complex<double>* data, long long len) const;

  void dup_bcast(long long* data, long long len) const;
  void dup_bcast(double* data, long long len) const;
  void dup_bcast(std::complex<double>* data, long long len) const;

  void neighbor_bcast(long long* data, const long long box_dims[]) const;
  void neighbor_bcast(double* data, const long long box_dims[]) const;
  void neighbor_bcast(std::complex<double>* data, const long long box_dims[]) const;

  void neighbor_reduce(long long* data, const long long box_dims[]) const;
  void neighbor_reduce(std::complex<double>* data, const long long box_dims[]) const;

  void record_mpi() const;
};

