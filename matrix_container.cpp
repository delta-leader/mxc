#include <matrix_container.hpp>

#include <complex>
#include <cstring>
#include <numeric>


template <class T>
MatrixDataContainer<T>::MatrixDataContainer(const MatrixDataContainer& container) : offsets(container.offsets) {
  if (0 < offsets.size()){
    long long data_len = offsets.back();
    data = (T*)std::malloc(data_len * sizeof(T));
    memcpy(data, container.data, data_len * sizeof(T));
  } else {
    data = nullptr;
  }
}

template <class T>
MatrixDataContainer<T>& MatrixDataContainer<T>::operator=(const MatrixDataContainer<T>& container) {
  MatrixDataContainer<T> tmp(container);
  offsets = container.offsets;
  if (data) {
    delete data;
    data = nullptr;
  }
  if (0 < offsets.size()){
    long long data_len = offsets.back();
    data = (T*)std::malloc(data_len * sizeof(T));
    memcpy(data, container.data, data_len * sizeof(T));
  }
  return *this;
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
  if (data)
    return offsets.back();
  return 0;
}

template<class T> inline MPI_Datatype get_mpi_datatype() {
  if (typeid(T) == typeid(long long))
    return MPI_LONG_LONG_INT;
  if (typeid(T) == typeid(double))
    return MPI_DOUBLE;
  if (typeid(T) == typeid(std::complex<float>))
    return MPI_C_FLOAT_COMPLEX;
  if (typeid(T) == typeid(std::complex<double>))
    return MPI_C_DOUBLE_COMPLEX;
  return MPI_DATATYPE_NULL;
}

template <class T>
void MatrixDataContainer<T>::write(MPI_File& fh, MPI_Offset& offset, MPI_Status& status) const {
  long long size = offsets.size();
  MPI_File_write_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status);
  offset += sizeof(long long);
  MPI_File_write_at(fh, offset, offsets.data(), size, MPI_LONG_LONG_INT, &status);
  offset += size * sizeof(long long);
  size = this->size();
  MPI_File_write_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status);
  offset += sizeof(long long);
  MPI_File_write_at(fh, offset, data, size, get_mpi_datatype<T>(), &status);
  offset += size * sizeof(T);
}

template <class T>
void MatrixDataContainer<T>::read(MPI_File& fh, MPI_Offset& offset, MPI_Status& status) {
  long long size;
  MPI_File_read_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status);
  offsets.resize(size);
  offset += sizeof(long long);
  MPI_File_read_at(fh, offset, offsets.data(), size, MPI_LONG_LONG_INT, &status);
  offset += size * sizeof(long long);
  MPI_File_read_at(fh, offset, &size, 1, MPI_LONG_LONG_INT, &status);

  if (0 < size) {
    data = (T*)std::realloc(data, size * sizeof(T));
  }
  else {
    if (data)
      std::free(data);
    data = nullptr;
  }

  offset += sizeof(long long);
  MPI_File_read_at(fh, offset, data, size, get_mpi_datatype<T>(), &status);
  offset += size * sizeof(T);
}

template class MatrixDataContainer<long long>;
template class MatrixDataContainer<double>;
template class MatrixDataContainer<std::complex<double>>;
template class MatrixDataContainer<std::complex<float>>;
