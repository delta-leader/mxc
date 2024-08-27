#pragma once

#include <vector>
#include <numeric>
#include <algorithm>

/*
Structure to store continuous blocks of data
with varying block sizes.
The constructor only allocates the memory and it needs
to be set seperately afterwards.
In principle, this class stores the blocks continously in memory
and uses a separate vector of offsets to indicate the
starting and ending indices of each block.
In this code this is mostly (always?) used on a per level basis.
For example, lets say that we have 4 cells on this level of sizes
2x2, 4x4, 3x3 and 2x2.
Then we would call the constructor with (4, [4, 16, 9, 4]) in
order to allocate storage for 33 elements.
*/
template<class T> class MatrixDataContainer {
private:
  // offsets for each block
  std::vector<long long> offsets;
  // actual data
  std::vector<T> data;

public:
  template <typename U> friend class MatrixDataContainer;

  MatrixDataContainer() {}

  /*
  Allocates the necessary memory.
  In:
    len: the number of blocks
    data: array containing the number of elements per block
  */
  MatrixDataContainer(const long long len, const long long* const data) : offsets(len + 1), data() {
    // computes prefix sum
    std::inclusive_scan(data, &data[len], offsets.begin() + 1);
    offsets[0] = 0;
    this->data = std::vector<T>(offsets.back());
  }

  template <typename U>
  MatrixDataContainer(const MatrixDataContainer<U>& container) : 
    offsets(container.offsets) {
    
    long long size = offsets.back();
    this->data = std::vector<T>(size);
    std::transform(container.data.data(), container.data.data() + size, data.begin(), [](U value) -> T {return T(value);});
  }

  /*
  Returns a pointer to first element of the block
  inidcated by index.
  */
  inline T* operator[](const long long index) {
    return data.data() + offsets[index];
  }

  inline const T* operator[](const long long index) const {
    return data.data() + offsets[index];
  }

  /*
  Returns the total amount of elements stored.
  */
  inline long long size() const {
    return offsets.back();
  }

  inline long long nblocks() const {
    return offsets.size() - 1;
  }

};
