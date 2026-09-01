#pragma once

#include <vector>

#include <mpi.h>


template<class T> class MatrixDataContainer {
private:
  std::vector<long long> offsets;
  T* data = nullptr;

public:
  MatrixDataContainer() = default;
  MatrixDataContainer(const MatrixDataContainer& container);
  MatrixDataContainer& operator=(const MatrixDataContainer& container);
  void alloc(long long len, const long long* dims);
  T* operator[](long long index);
  const T* operator[](long long index) const;
  long long size() const;
  void write(MPI_File& fh, MPI_Offset& offset, MPI_Status& status) const;
  void read(MPI_File& fh, MPI_Offset& offset, MPI_Status& status);
};

