
#pragma once

#include "mpi.h"
#include <vector>
#include <cstdint>
#include <complex>

class CSR;
class Cell;

class CellComm {
private:
  int64_t Proc;
  std::vector<std::pair<int64_t, int64_t>> ProcBoxes;
  
  std::vector<std::pair<int, MPI_Comm>> Comm_box;
  MPI_Comm Comm_share, Comm_merge;

  template<typename T> inline void level_merge(T* data, int64_t len) const;
  template<typename T> inline void dup_bast(T* data, int64_t len) const;
  template<typename T> inline void neighbor_bcast(T* data, const int64_t box_dims[]) const;
  template<typename T> inline void neighbor_reduce(T* data, const int64_t box_dims[]) const;

public:
  std::pair<double, double>* timer;

  CellComm() : Proc(-1), ProcBoxes(), Comm_box(), Comm_share(MPI_COMM_NULL), Comm_merge(MPI_COMM_NULL), timer(nullptr) {};
  CellComm(int64_t P, const std::vector<int64_t>& Targets, int64_t cbegin, int64_t clen, int merge, int share, std::vector<MPI_Comm>& unique_comms, MPI_Comm world);
  
  int64_t iLocal(int64_t iglobal) const;
  int64_t iGlobal(int64_t ilocal) const;
  int64_t oLocal() const;
  int64_t oGlobal() const;
  int64_t lenLocal() const;
  int64_t lenNeighbors() const;

  void level_merge(std::complex<double>* data, int64_t len) const;

  void dup_bcast(int64_t* data, int64_t len) const;
  void dup_bcast(double* data, int64_t len) const;
  void dup_bcast(std::complex<double>* data, int64_t len) const;

  void neighbor_bcast(int64_t* data, const int64_t box_dims[]) const;
  void neighbor_bcast(double* data, const int64_t box_dims[]) const;
  void neighbor_bcast(std::complex<double>* data, const int64_t box_dims[]) const;

  void neighbor_reduce(std::complex<double>* data, const int64_t box_dims[]) const;

  void record_mpi() const;
};

std::vector<MPI_Comm> buildComm(CellComm* comms, int64_t ncells, const Cell* cells, const CSR& cellFar, const CSR& cellNear, int64_t levels, MPI_Comm world);

