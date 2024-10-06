#pragma once

#include <complex>
#include <vector>

template<class T> class MatrixDataContainer {
private:
  std::vector<long long> offsets;
  T* data = nullptr;

public:
  void alloc(long long len, const long long* dims);
  T* operator[](long long index);
  const T* operator[](long long index) const;
  long long size() const;
};

