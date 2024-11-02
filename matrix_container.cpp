
#include <matrix_container.hpp>
#include <numeric>
#include <algorithm>
#include <cstring>

/* explicit template instantiation */
// complex double
template class MatrixDataContainer<std::complex<double>>;
// complex float
template class MatrixDataContainer<std::complex<float>>;
// double
template class MatrixDataContainer<double>;
// float
template class MatrixDataContainer<float>;

/* supported type conversions */
// (complex) double to float
template MatrixDataContainer<std::complex<float>>::MatrixDataContainer(const MatrixDataContainer<std::complex<double>>&);
template MatrixDataContainer<float>::MatrixDataContainer(const MatrixDataContainer<double>&);
// (complex) float to double
template MatrixDataContainer<std::complex<double>>::MatrixDataContainer(const MatrixDataContainer<std::complex<float>>&);
template MatrixDataContainer<double>::MatrixDataContainer(const MatrixDataContainer<float>&);

template <class T>
MatrixDataContainer<T>::~MatrixDataContainer() {
  if (data)
    delete data;
}

template <class T>
MatrixDataContainer<T>::MatrixDataContainer(const MatrixDataContainer& container) : offsets(container.offsets) {
  long long data_len = offsets.back();
  if (0 < data_len) {
   data = (T*)std::malloc(data_len * sizeof(T));
   memcpy(data, container.data, data_len * sizeof(T));
  } else {
    data = nullptr;
  }
}

template <class T> template <class U>
MatrixDataContainer<T>::MatrixDataContainer(const MatrixDataContainer<U>& container) : offsets(container.offsets) {
  long long data_len = offsets.back();
  if (0 < data_len) {
   data = (T*)std::malloc(data_len * sizeof(T));
   std::transform(container.data, container.data + data_len, data, [](U value) -> T {return T(value);});
  } else {
    data = nullptr;
  }
}

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
