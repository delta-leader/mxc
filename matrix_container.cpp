
#include <matrix_container.hpp>
#include <numeric>

template <class T>
void MatrixDataContainer<T>::alloc(long long len, const long long* dims) {
  offsets.resize(len + 1);
  std::inclusive_scan(dims, &dims[len], offsets.begin() + 1);
  offsets[0] = 0;
  long long data_len = offsets.back();

  if (0 < data_len) {
    data = (T*)std::realloc(data, offsets.back() * sizeof(T));
    std::fill(data, data + offsets.back(), static_cast<T>(0));
  }
  else {
    if (data)
      std::free(data);
    data = nullptr;
  }
}

template <class T>
T* MatrixDataContainer<T>::operator[](long long index) {
  return (0 <= index && index < (long long)offsets.size()) ? data + offsets[index] : nullptr;
}

template <class T>
const T* MatrixDataContainer<T>::operator[](long long index) const {
  return (0 <= index && index < (long long)offsets.size()) ? data + offsets[index] : nullptr;
}

template <class T>
long long MatrixDataContainer<T>::size() const {
  return offsets.back();
}

template class MatrixDataContainer<double>;
template class MatrixDataContainer<std::complex<double>>;
