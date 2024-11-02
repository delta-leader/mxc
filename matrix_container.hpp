#pragma once

#include <complex>
#include <vector>

template<class T> class MatrixDataContainer {
private:
  std::vector<long long> offsets;
  T* data = nullptr;

public:
  template <class U> friend class MatrixDataContainer;
  MatrixDataContainer() = default;
  MatrixDataContainer(const MatrixDataContainer& container);
  template <class U>
  MatrixDataContainer(const MatrixDataContainer<U>& container);
  ~MatrixDataContainer();
  void alloc(long long len, const long long* dims);
  T* operator[](long long index);
  const T* operator[](long long index) const;
  long long size() const;
};

