#pragma once

#include <vector>
#include <numeric>

template<class T> class MatrixDataContainer {
private:
  std::vector<long long> offsets;
  std::vector<T> data;

public:
  MatrixDataContainer() {}

  MatrixDataContainer(long long len, const long long* dims) : offsets(len + 1), data() {
    std::inclusive_scan(dims, &dims[len], offsets.begin() + 1);
    offsets[0] = 0;
    data = std::vector<T>(offsets.back());
  }

  inline T* operator[](long long index) {
    return data.data() + offsets[index];
  }

  inline const T* operator[](long long index) const {
    return data.data() + offsets[index];
  }

  inline long long size() const {
    return offsets.back();
  }

  inline long long nblocks() const {
    return offsets.size() - 1;
  }

};
