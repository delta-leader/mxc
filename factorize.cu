
#include <factorize.cuh>
#include <comm-mpi.hpp>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <mkl.h>
#include <cstring>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/complex.h>
#include <thrust/sequence.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>

#include <cuda_fp16.h>
#include <iostream>

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

__global__ void tofloat(const __half * __restrict__ in, float * __restrict__ out, const long long N){
  size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx < N)
    out[idx] = __half2float(in[idx]);
}

__global__ void set(float* out, const long long N){
  size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx < N)
    out[idx] = 1;//__float2half(in[idx]);
}

__global__ void tohalf(const float * __restrict__ in, __half * __restrict__ out, const long long N){
  size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx < N)
    out[idx] = __float2half(in[idx]);
}

/* helper functions for different datatypes */
template <typename DT>
// complex double
inline void fill_zero(DT* start, DT* end) {
  std::fill(start, end, (DT) 0);
}

// complex double
template <>
inline void fill_zero(cuDoubleComplex* start, cuDoubleComplex* end) {
  std::fill(start, end, make_cuDoubleComplex(0., 0.));
}
void omatcopy(char ordering, char trans, size_t rows, size_t cols, const std::complex<double> *SRC, size_t src_stride, cuDoubleComplex *DST, size_t dst_stride) {
  MKL_Zomatcopy(ordering, trans, rows, cols, std::complex<double>(1., 0.), SRC, src_stride, reinterpret_cast<std::complex<double>*>(DST), dst_stride);
}
void omatcopy(char ordering, char trans, size_t rows, size_t cols, cuDoubleComplex *SRC, size_t src_stride, std::complex<double> *DST, size_t dst_stride) {
  MKL_Zomatcopy(ordering, trans, rows, cols, std::complex<double>(1., 0.), reinterpret_cast<std::complex<double>*>(SRC), src_stride, DST, dst_stride);
}
// complex float
template <>
inline void fill_zero(cuComplex* start, cuComplex* end) {
  std::fill(start, end, make_cuComplex(0., 0.));
}
void omatcopy(char ordering, char trans, size_t rows, size_t cols, const std::complex<float> *SRC, size_t src_stride, cuComplex *DST, size_t dst_stride) {
  MKL_Comatcopy(ordering, trans, rows, cols, std::complex<float>(1., 0.), SRC, src_stride, reinterpret_cast<std::complex<float>*>(DST), dst_stride);
}
void omatcopy(char ordering, char trans, size_t rows, size_t cols, cuComplex *SRC, size_t src_stride, std::complex<float> *DST, size_t dst_stride) {
  MKL_Comatcopy(ordering, trans, rows, cols, std::complex<float>(1., 0.), reinterpret_cast<std::complex<float>*>(SRC), src_stride, DST, dst_stride);
}
// double
void omatcopy(char ordering, char trans, size_t rows, size_t cols, const double *SRC, size_t src_stride, double *DST, size_t dst_stride) {
  MKL_Domatcopy(ordering, trans, rows, cols, 1, SRC, src_stride, DST, dst_stride);
}
void omatcopy(char ordering, char trans, size_t rows, size_t cols, double *SRC, size_t src_stride, double *DST, size_t dst_stride) {
  MKL_Domatcopy(ordering, trans, rows, cols, 1, SRC, src_stride, DST, dst_stride);
}
// float
void omatcopy(char ordering, char trans, size_t rows, size_t cols, const float *SRC, size_t src_stride, float *DST, size_t dst_stride) {
  MKL_Somatcopy(ordering, trans, rows, cols, 1, SRC, src_stride, DST, dst_stride);
}
void omatcopy(char ordering, char trans, size_t rows, size_t cols, float *SRC, size_t src_stride, float *DST, size_t dst_stride) {
  MKL_Somatcopy(ordering, trans, rows, cols, 1, SRC, src_stride, DST, dst_stride);
}

struct keysDLU {
  long long D, M, N;
  keysDLU(long long D, long long M, long long N) : D(D), M(M), N(N) {}
  __host__ __device__ long long operator()(long long y, long long x) const {
    long long diff = D + y - x;
    long long pred = (diff != 0) + (diff < 0);
    return (pred * M + y) * N + x;
  }
};

template<class T> struct setDevicePtr {
  T* data;
  long long block, brows;
  setDevicePtr(T* data, long long block, long long brows = 0) : data(data), block(block), brows(brows) {}
  __host__ __device__ T* operator()(long long i) const {
    return data + i * block;
  }
  __host__ __device__ T* operator()(long long y, long long x) const {
    return data + (y + x * brows) * block;
  }
};

struct swapXY {
  long long M, B;
  swapXY(long long M, long long B) : M(M), B(B) {}
  __host__ __device__ long long operator()(long long i) const {
    long long x = i / B; long long y = i - x * B;
    long long z = y / M; long long w = y - z * M;
    return x * B + z + w * M;
  }
};

template<class T> struct StridedBlock {
  long long M, B, LD;
  T **pA;
  StridedBlock(long long M, long long N, long long LD, T** pA) : M(M), B(M * N), LD(LD), pA(pA) {}
  __host__ __device__ T& operator()(long long i) const {
    long long x = i / B; long long y = i - x * B;
    long long z = y / M; long long w = y - z * M;
    return pA[x][z * LD + w];
  }
};

struct conjugateDouble {
  __host__ __device__ thrust::complex<double> operator()(const thrust::complex<double>& z) const {
    return thrust::conj(z);
  }
};

struct conjugateFloat {
  __host__ __device__ thrust::complex<float> operator()(const thrust::complex<float>& z) const {
    return thrust::conj(z);
  }
};

// needs explicit specialization due to cuBLAS calls
// TODO these need to be updated
/*template <>
void H2Factorize<cuDoubleComplex>::factorize(const long long lenA, const long long bdim, const long long rank, const long long block, const long long M, const long long D,
  cuDoubleComplex** A_SS, cuDoubleComplex** A_SR, cuDoubleComplex** A_RS, cuDoubleComplex** A_RR, cuDoubleComplex** U, cuDoubleComplex** V, cuDoubleComplex** V_R, cuDoubleComplex** B) {
  long long rdim = bdim - rank;
  int info_host = 0;
  cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.), minus_one = make_cuDoubleComplex(-1., 0.);

  thrust::device_ptr<const thrust::complex<double>> u_ptr = thrust::device_ptr<const thrust::complex<double>>(reinterpret_cast<const thrust::complex<double>*>(&Udata[D * block]));
  thrust::device_ptr<thrust::complex<double>> v_ptr = thrust::device_ptr<thrust::complex<double>>(reinterpret_cast<thrust::complex<double>*>(Vdata));

  auto map = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), swapXY(bdim, block));
  thrust::gather(thrust::cuda::par.on(stream), map, map + block * M, thrust::make_transform_iterator(u_ptr, conjugateDouble()), v_ptr);

  cublasZgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, A_SS, bdim, &zero, B, bdim, M);
  cublasZgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, B, bdim, &zero, A_SS, bdim, M);

  cublasZgetrfBatched(cublasH, rdim, A_RR, bdim, ipiv, info, M);
  cublasZgetrsBatched(cublasH, CUBLAS_OP_N, rdim, rank, A_RR, bdim, ipiv, A_RS, bdim, &info_host, M);
  cublasZgetrsBatched(cublasH, CUBLAS_OP_N, rdim, bdim, A_RR, bdim, ipiv, V_R, bdim, &info_host, M);

  cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, A_SR, bdim, A_RS, bdim, &one, A_SS, bdim, M);

  for (int64_t i = M; i < lenA; i += maxQ) {
    int64_t len = std::min(lenA - i, maxQ);
    cublasZgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, &U[i], bdim, &A_SS[i], bdim, &zero, B, bdim, len);
    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, &V[i], bdim, B, bdim, &zero, &A_SS[i], bdim, len);
  }
}

template <>
void H2Factorize<cuComplex>::factorize(const long long lenA, const long long bdim, const long long rank, const long long block, const long long M, const long long D,
  cuComplex** A_SS, cuComplex** A_SR, cuComplex** A_RS, cuComplex** A_RR, cuComplex** U, cuComplex** V, cuComplex** V_R, cuComplex** B) {
  long long rdim = bdim - rank;
  int info_host = 0;
  cuComplex one = make_cuComplex(1., 0.), zero = make_cuComplex(0., 0.), minus_one = make_cuComplex(-1., 0.);

  thrust::device_ptr<const thrust::complex<float>> u_ptr = thrust::device_ptr<const thrust::complex<float>>(reinterpret_cast<const thrust::complex<float>*>(&Udata[D * block]));
  thrust::device_ptr<thrust::complex<float>> v_ptr = thrust::device_ptr<thrust::complex<float>>(reinterpret_cast<thrust::complex<float>*>(Vdata));

  auto map = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), swapXY(bdim, block));
  thrust::gather(thrust::cuda::par.on(stream), map, map + block * M, thrust::make_transform_iterator(u_ptr, conjugateFloat()), v_ptr);

  cublasCgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, A_SS, bdim, &zero, B, bdim, M);
  cublasCgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, B, bdim, &zero, A_SS, bdim, M);

  cublasCgetrfBatched(cublasH, rdim, A_RR, bdim, ipiv, info, M);
  cublasCgetrsBatched(cublasH, CUBLAS_OP_N, rdim, rank, A_RR, bdim, ipiv, A_RS, bdim, &info_host, M);
  cublasCgetrsBatched(cublasH, CUBLAS_OP_N, rdim, bdim, A_RR, bdim, ipiv, V_R, bdim, &info_host, M);

  cublasCgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, A_SR, bdim, A_RS, bdim, &one, A_SS, bdim, M);

  for (int64_t i = M; i < lenA; i += maxQ) {
    int64_t len = std::min(lenA - i, maxQ);
    cublasCgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, &U[i], bdim, &A_SS[i], bdim, &zero, B, bdim, len);
    cublasCgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, &V[i], bdim, B, bdim, &zero, &A_SS[i], bdim, len);
  }
}

template <>
void H2Factorize<double>::factorize(const long long lenA, const long long bdim, const long long rank, const long long block, const long long M, const long long D,
  double** A_SS, double** A_SR, double** A_RS, double** A_RR, double** U, double** V, double** V_R, double** B) {
  long long rdim = bdim - rank;
  int info_host = 0;
  double one = 1., zero = 0., minus_one = -1.;

  thrust::device_ptr<const double> u_ptr = thrust::device_ptr<const double>(reinterpret_cast<const double*>(&Udata[D * block]));
  thrust::device_ptr<double> v_ptr = thrust::device_ptr<double>(reinterpret_cast<double*>(Vdata));

  auto map = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), swapXY(bdim, block));
  thrust::gather(thrust::cuda::par.on(stream), map, map + block * M, u_ptr, v_ptr);

  cublasDgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, A_SS, bdim, &zero, B, bdim, M);
  cublasDgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, B, bdim, &zero, A_SS, bdim, M);

  cublasDgetrfBatched(cublasH, rdim, A_RR, bdim, ipiv, info, M);
  cublasDgetrsBatched(cublasH, CUBLAS_OP_N, rdim, rank, A_RR, bdim, ipiv, A_RS, bdim, &info_host, M);
  cublasDgetrsBatched(cublasH, CUBLAS_OP_N, rdim, bdim, A_RR, bdim, ipiv, V_R, bdim, &info_host, M);

  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, A_SR, bdim, A_RS, bdim, &one, A_SS, bdim, M);

  for (int64_t i = M; i < lenA; i += maxQ) {
    int64_t len = std::min(lenA - i, maxQ);
    cublasDgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, &U[i], bdim, &A_SS[i], bdim, &zero, B, bdim, len);
    cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, &V[i], bdim, B, bdim, &zero, &A_SS[i], bdim, len);
  }
}*/

template<class T> struct copyFunc {
  const T** srcs;
  T** dsts;
  long long M, B, ls, ld;
  copyFunc(long long M, long long N, const T* srcs[], long long ls, T* dsts[], long long ld) :
    srcs(srcs), dsts(dsts), M(M), B(M * N), ls(ls), ld(ld) {}
  __host__ __device__ void operator()(long long i) const {
    long long x = i / B; long long y = i - x * B;
    long long z = y / M; long long w = y - z * M;
    T e = srcs[x][z * ls + w];
    dsts[x][z * ld + w] = e;
  }
};

void factorize(const cublasHandle_t cublasH, const long long lenA, const long long bdim, const long long rank, const long long block, const long long D, const long long M, const long long N, long long reduc_len,
  double* Udata, double* Vdata, double* Bdata, double* const R, double* ACdata,
  double** A_SS, double** A_SR, double** A_RS, double** A_RR, double** U, double** U_R, double** V, double** V_R, double** B, double** B_Cols, double** B_I_Cols, double** A_SR_Rows,
  double** ACC, double** ACC_Final, const ColCommMPI& comm) {}

void factorize(const cublasHandle_t cublasH, const long long lenA, const long long bdim, const long long rank, const long long block, const long long D, const long long M, const long long N, long long reduc_len,
  float* Udata, float* Vdata, float* Bdata, float* const R, float* ACdata,
  float** A_SS, float** A_SR, float** A_RS, float** A_RR, float** U, float** U_R, float** V, float** V_R, float** B, float** B_Cols, float** B_I_Cols, float** A_SR_Rows,
  float** ACC, float** ACC_Final, const ColCommMPI& comm) {
  cudaStream_t stream;
  cublasGetStream(cublasH, &stream);

  long long rdim = bdim - rank;
  int info_host = 0;
  float one = 1., zero = 0., minus_one = -1.;

  auto inc_iter = thrust::make_counting_iterator(0ll);
  std::vector<long long> Bsizes(N, block);
  comm.dataSizesToNeighborOffsets(Bsizes.data());
  thrust::device_vector<int> Ipiv(M * bdim);
  thrust::device_vector<int> Info(M);
  int* ipiv = thrust::raw_pointer_cast(Ipiv.data());
  int* info = thrust::raw_pointer_cast(Info.data());
  
  thrust::device_ptr<const float> u_ptr = thrust::device_ptr<const float>(reinterpret_cast<const float*>(&Udata[D * block]));
  thrust::device_ptr<float> v_ptr = thrust::device_ptr<float>(reinterpret_cast<float*>(Vdata));

  auto map = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), swapXY(bdim, block));
  thrust::gather(thrust::cuda::par.on(stream), map, map + block * M, u_ptr, v_ptr);
  
  cublasSgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, A_SS, bdim, &zero, B, bdim, M);
  cublasSgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, B, bdim, &zero, A_SS, bdim, M);

  cublasSgetrfBatched(cublasH, rdim, A_RR, bdim, ipiv, info, M);
  cublasSgetrsBatched(cublasH, CUBLAS_OP_N, rdim, bdim, A_RR, bdim, ipiv, V_R, bdim, &info_host, M);

  if (0 < rank) {
    cublasSgetrsBatched(cublasH, CUBLAS_OP_N, rdim, rank, A_RR, bdim, ipiv, A_RS, bdim, &info_host, M);

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      cublasSgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, &U[i], bdim, &A_SS[i], bdim, &zero, B, bdim, len);
      cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, &V[i], bdim, B, bdim, &zero, &A_SS[i], bdim, len);
    }
    cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, A_SR_Rows, bdim, A_RS, bdim, &one, A_SS, bdim, lenA);

    thrust::for_each(thrust::cuda::par.on(stream), inc_iter, inc_iter + (rdim * rank * M), copyFunc(rdim, rank, const_cast<const float**>(A_RS), bdim, &B[D], bdim));
    cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rdim, rdim, bdim, &one, V_R, bdim, U_R, bdim, &zero, B_I_Cols, bdim, M);
    cudaMemcpyAsync(&R[D * block], &Bdata[D * block], M * block * sizeof(float), cudaMemcpyDeviceToHost, stream);
    thrust::for_each(thrust::cuda::par.on(stream), inc_iter, inc_iter + (rank * rank * M), copyFunc(rank, rank, const_cast<const float**>(A_SS), bdim, ACC, rank));

    comm.neighbor_bcast(R, Bsizes.data());
    cudaMemcpyAsync(Bdata, R, N * block * sizeof(float), cudaMemcpyHostToDevice, stream);

    if (M < lenA)
      cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &A_SR[M], bdim, &B_Cols[M], bdim, &one, &A_SS[M], bdim, lenA - M);

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, rdim, &one, &B_I_Cols[i], bdim, &A_SR[i], bdim, &zero, B, bdim, len);
      cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &A_SR[i], bdim, B, bdim, &zero, &ACC[i], rank, len);
    }

    while (1 < reduc_len) {
      long long len = reduc_len * rank * rank * M;
      reduc_len = (reduc_len + 1) / 2;
      long long tail_start = reduc_len * rank * rank * M;
      long long tail_len = len - tail_start;
      cublasSaxpy(cublasH, tail_len, &one, &ACdata[tail_start], 1, ACdata, 1);
    }
    thrust::for_each(thrust::cuda::par.on(stream), inc_iter, inc_iter + (rank * rank * M), copyFunc(rank, rank, const_cast<const float**>(ACC_Final), rank, A_SS, bdim));
  }
}

/*template <typename DT> template <typename OT>
void H2Factorize<DT>::compute(const long long bdim, const long long rank, const long long D, const long long M, const long long N, const long long ARows[], const long long ACols[], OT* const A, OT* const R, const OT* const Q) {
  long long block = bdim * bdim;
  long long lenA = ARows[M];

  std::copy(Q, Q + block * N, reinterpret_cast<OT*>(hostA));
  cudaMemcpy(Udata, hostA, block * N * sizeof(DT), cudaMemcpyHostToDevice);

  std::copy(A, A + block * lenA, reinterpret_cast<OT*>(hostA));
  cudaMemcpy(Adata, hostA, block * lenA * sizeof(DT), cudaMemcpyHostToDevice);
*/
/*
template <typename DT>
void compute_factorize(const cublasHandle_t cublasH, const long long bdim, const long long rank, const long long D, const long long M, const long long N, const long long ARows[], const long long ACols[], DT* const A, DT* const R, const DT* const Q, const ColCommMPI& comm) {
  long long block = bdim * bdim;
  long long lenA = ARows[M];
  
  cudaStream_t stream;
  cublasGetStream(cublasH, &stream);

  thrust::device_vector<long long> row_offsets(ARows, ARows + M);
  thrust::device_vector<long long> rows(lenA, 0ll);
  thrust::device_vector<long long> cols(ACols, ACols + lenA);
  thrust::device_vector<long long> dist_cols(lenA);
  thrust::device_vector<long long> keys(lenA);
  thrust::device_vector<long long> indices(lenA);
  thrust::device_vector<DT*> a_ss(lenA), a_sr(lenA), a_rs(lenA), a_rr(lenA);
  thrust::device_vector<DT*> u(lenA), v(lenA), u_r(M), v_r(M), b(N), b_cols(lenA), b_i_cols(lenA);
  thrust::device_vector<DT*> a_sr_rows(lenA), acc(lenA), acc_final(M);

  thrust::device_vector<DT> Avec(lenA * block);
  thrust::device_vector<DT> Bvec(N * block);
  thrust::device_vector<DT> Uvec(N * block);
  thrust::device_vector<DT> Vvec(M * block);
  
  auto inc_iter = thrust::make_counting_iterator(0ll);
  auto one_iter = thrust::make_constant_iterator(1ll);
  auto rwise_diag_iter = thrust::make_permutation_iterator(indices.begin(), rows.begin());

  DT* Adata = thrust::raw_pointer_cast(Avec.data());
  DT* Udata = thrust::raw_pointer_cast(Uvec.data());
  DT* Vdata = thrust::raw_pointer_cast(Vvec.data());
  DT* Bdata = thrust::raw_pointer_cast(Bvec.data());

  cudaMemcpyAsync(Udata, Q, block * N * sizeof(DT), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(Adata, A, block * lenA * sizeof(DT), cudaMemcpyHostToDevice, stream);

  thrust::scatter(one_iter, one_iter + (M - 1), row_offsets.begin() + 1, rows.begin()); 
  thrust::inclusive_scan(rows.begin(), rows.end(), rows.begin());
  thrust::exclusive_scan_by_key(rows.begin(), rows.end(), one_iter, dist_cols.begin(), 0ll);

  thrust::transform(rows.begin(), rows.end(), cols.begin(), keys.begin(), keysDLU(D, M, N));
  thrust::sequence(indices.begin(), indices.end(), 0);
  thrust::sort_by_key(keys.begin(), keys.end(), thrust::make_zip_iterator(rows.begin(), cols.begin(), dist_cols.begin(), indices.begin()));

  long long reduc_len = 1ll + thrust::reduce(dist_cols.begin(), dist_cols.end(), 0ll, thrust::maximum<long long>());
  // TODO this is a problem
  //thrust::device_vector<DT> ACvec(reduc_len * M * rank * rank, make_cuDoubleComplex(0., 0.));
  thrust::device_vector<DT> ACvec(reduc_len * M * rank * rank, (DT) 0);
  DT* ACdata = thrust::raw_pointer_cast(ACvec.data());

  long long offset_SR = rank * bdim, offset_RS = rank, offset_RR = rank * (bdim + 1);
  thrust::transform(indices.begin(), indices.end(), a_ss.begin(), setDevicePtr(Adata, block));
  thrust::transform(indices.begin(), indices.end(), a_sr.begin(), setDevicePtr(Adata + offset_SR, block));
  thrust::transform(indices.begin(), indices.end(), a_rs.begin(), setDevicePtr(Adata + offset_RS, block));
  thrust::transform(indices.begin(), indices.end(), a_rr.begin(), setDevicePtr(Adata + offset_RR, block));
  thrust::transform(cols.begin(), cols.end(), u.begin(), setDevicePtr(Udata, block));
  thrust::transform(cols.begin(), cols.begin() + M, u_r.begin(), setDevicePtr(Udata + offset_SR, block));
  thrust::transform(rows.begin(), rows.end(), v.begin(), setDevicePtr(Vdata, block));
  thrust::transform(inc_iter, inc_iter + M, v_r.begin(), setDevicePtr(Vdata + offset_RS, block));

  thrust::transform(inc_iter, inc_iter + N, b.begin(), setDevicePtr(Bdata, block));
  thrust::transform(cols.begin(), cols.end(), b_cols.begin(), setDevicePtr(Bdata, block));
  thrust::transform(cols.begin(), cols.end(), b_i_cols.begin(), setDevicePtr(Bdata + offset_SR, block));
  thrust::transform(rwise_diag_iter, rwise_diag_iter + lenA, a_sr_rows.begin(), setDevicePtr(Adata + offset_SR, block));
  thrust::transform(rows.begin(), rows.end(), dist_cols.begin(), acc.begin(), setDevicePtr(ACdata, rank * rank, M));
  thrust::transform(inc_iter, inc_iter + M, acc_final.begin(), setDevicePtr(ACdata, rank * rank));

  DT** A_SS = thrust::raw_pointer_cast(a_ss.data());
  DT** A_SR = thrust::raw_pointer_cast(a_sr.data());
  DT** A_RS = thrust::raw_pointer_cast(a_rs.data());
  DT** A_RR = thrust::raw_pointer_cast(a_rr.data());
  DT** U = thrust::raw_pointer_cast(u.data());
  DT** U_R = thrust::raw_pointer_cast(u_r.data());
  DT** V = thrust::raw_pointer_cast(v.data());
  DT** V_R = thrust::raw_pointer_cast(v_r.data());
  DT** B = thrust::raw_pointer_cast(b.data());
  DT** B_Cols = thrust::raw_pointer_cast(b_cols.data());
  DT** B_I_Cols = thrust::raw_pointer_cast(b_i_cols.data());
  DT** A_SR_Rows = thrust::raw_pointer_cast(a_sr_rows.data());
  DT** ACC = thrust::raw_pointer_cast(acc.data());
  DT** ACC_Final = thrust::raw_pointer_cast(acc_final.data());

  factorize(cublasH, lenA, bdim, rank, block, D, M, N, reduc_len, Udata, Vdata, Bdata, R, ACdata, A_SS, A_SR, A_RS, A_RR, U, U_R, V, V_R, B, B_Cols, B_I_Cols, A_SR_Rows, ACC, ACC_Final, comm);
  cudaStreamSynchronize(stream);

  cudaMemcpy(A, Adata, block * lenA * sizeof(DT), cudaMemcpyDeviceToHost);
  cudaMemcpy(&R[block * D], Vdata, block * M * sizeof(DT), cudaMemcpyDeviceToHost);
}
*/

template<>
void compute_factorize(cublasHandle_t cublasH, long long bdim, long long rank, long long D, long long M, long long N, const long long ARows[], const long long ACols[], std::complex<double>* A, std::complex<double>* R, const std::complex<double>* Q, const ColCommMPI& comm) {
  long long block = bdim * bdim;
  long long lenA = ARows[M];
  
  cudaStream_t stream;
  cublasGetStream(cublasH, &stream);

  thrust::device_vector<long long> row_offsets(ARows, ARows + M);
  thrust::device_vector<long long> rows(lenA, 0ll);
  thrust::device_vector<long long> cols(ACols, ACols + lenA);
  thrust::device_vector<long long> dist_cols(lenA);
  thrust::device_vector<long long> keys(lenA);
  thrust::device_vector<long long> indices(lenA);
  thrust::device_vector<cuDoubleComplex*> a_ss(lenA), a_sr(lenA), a_rs(lenA), a_rr(lenA);
  thrust::device_vector<cuDoubleComplex*> u(lenA), v(lenA), u_r(M), v_r(M), b(N), b_cols(lenA), b_i_cols(lenA);
  thrust::device_vector<cuDoubleComplex*> a_sr_rows(lenA), acc(lenA), acc_final(M);

  thrust::device_vector<cuDoubleComplex> Avec(lenA * block);
  thrust::device_vector<cuDoubleComplex> Bvec(N * block);
  thrust::device_vector<cuDoubleComplex> Uvec(N * block);
  thrust::device_vector<cuDoubleComplex> Vvec(M * block);
  
  thrust::device_vector<int> Ipiv(M * bdim);
  thrust::device_vector<int> Info(M);
  std::vector<long long> Bsizes(N, block);
  comm.dataSizesToNeighborOffsets(Bsizes.data());

  auto inc_iter = thrust::make_counting_iterator(0ll);
  auto one_iter = thrust::make_constant_iterator(1ll);
  auto rwise_diag_iter = thrust::make_permutation_iterator(indices.begin(), rows.begin());

  cuDoubleComplex* Adata = thrust::raw_pointer_cast(Avec.data());
  cuDoubleComplex* Udata = thrust::raw_pointer_cast(Uvec.data());
  cuDoubleComplex* Vdata = thrust::raw_pointer_cast(Vvec.data());
  cuDoubleComplex* Bdata = thrust::raw_pointer_cast(Bvec.data());

  cudaMemcpyAsync(Udata, Q, block * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(Adata, A, block * lenA * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);

  thrust::scatter(one_iter, one_iter + (M - 1), row_offsets.begin() + 1, rows.begin()); 
  thrust::inclusive_scan(rows.begin(), rows.end(), rows.begin());
  thrust::exclusive_scan_by_key(rows.begin(), rows.end(), one_iter, dist_cols.begin(), 0ll);

  thrust::transform(rows.begin(), rows.end(), cols.begin(), keys.begin(), keysDLU(D, M, N));
  thrust::sequence(indices.begin(), indices.end(), 0);
  thrust::sort_by_key(keys.begin(), keys.end(), thrust::make_zip_iterator(rows.begin(), cols.begin(), dist_cols.begin(), indices.begin()));

  long long reduc_len = 1ll + thrust::reduce(dist_cols.begin(), dist_cols.end(), 0ll, thrust::maximum<long long>());
  thrust::device_vector<cuDoubleComplex> ACvec(reduc_len * M * rank * rank, make_cuDoubleComplex(0., 0.));
  cuDoubleComplex* ACdata = thrust::raw_pointer_cast(ACvec.data());

  long long offset_SR = rank * bdim, offset_RS = rank, offset_RR = rank * (bdim + 1);
  thrust::transform(indices.begin(), indices.end(), a_ss.begin(), setDevicePtr(Adata, block));
  thrust::transform(indices.begin(), indices.end(), a_sr.begin(), setDevicePtr(Adata + offset_SR, block));
  thrust::transform(indices.begin(), indices.end(), a_rs.begin(), setDevicePtr(Adata + offset_RS, block));
  thrust::transform(indices.begin(), indices.end(), a_rr.begin(), setDevicePtr(Adata + offset_RR, block));
  thrust::transform(cols.begin(), cols.end(), u.begin(), setDevicePtr(Udata, block));
  thrust::transform(cols.begin(), cols.begin() + M, u_r.begin(), setDevicePtr(Udata + offset_SR, block));
  thrust::transform(rows.begin(), rows.end(), v.begin(), setDevicePtr(Vdata, block));
  thrust::transform(inc_iter, inc_iter + M, v_r.begin(), setDevicePtr(Vdata + offset_RS, block));

  thrust::transform(inc_iter, inc_iter + N, b.begin(), setDevicePtr(Bdata, block));
  thrust::transform(cols.begin(), cols.end(), b_cols.begin(), setDevicePtr(Bdata, block));
  thrust::transform(cols.begin(), cols.end(), b_i_cols.begin(), setDevicePtr(Bdata + offset_SR, block));
  thrust::transform(rwise_diag_iter, rwise_diag_iter + lenA, a_sr_rows.begin(), setDevicePtr(Adata + offset_SR, block));
  thrust::transform(rows.begin(), rows.end(), dist_cols.begin(), acc.begin(), setDevicePtr(ACdata, rank * rank, M));
  thrust::transform(inc_iter, inc_iter + M, acc_final.begin(), setDevicePtr(ACdata, rank * rank));

  cuDoubleComplex** A_SS = thrust::raw_pointer_cast(a_ss.data());
  cuDoubleComplex** A_SR = thrust::raw_pointer_cast(a_sr.data());
  cuDoubleComplex** A_RS = thrust::raw_pointer_cast(a_rs.data());
  cuDoubleComplex** A_RR = thrust::raw_pointer_cast(a_rr.data());
  cuDoubleComplex** U = thrust::raw_pointer_cast(u.data());
  cuDoubleComplex** U_R = thrust::raw_pointer_cast(u_r.data());
  cuDoubleComplex** V = thrust::raw_pointer_cast(v.data());
  cuDoubleComplex** V_R = thrust::raw_pointer_cast(v_r.data());
  cuDoubleComplex** B = thrust::raw_pointer_cast(b.data());
  cuDoubleComplex** B_Cols = thrust::raw_pointer_cast(b_cols.data());
  cuDoubleComplex** B_I_Cols = thrust::raw_pointer_cast(b_i_cols.data());
  cuDoubleComplex** A_SR_Rows = thrust::raw_pointer_cast(a_sr_rows.data());
  cuDoubleComplex** ACC = thrust::raw_pointer_cast(acc.data());
  cuDoubleComplex** ACC_Final = thrust::raw_pointer_cast(acc_final.data());

  int* ipiv = thrust::raw_pointer_cast(Ipiv.data());
  int* info = thrust::raw_pointer_cast(Info.data());

  long long rdim = bdim - rank;
  int info_host = 0;
  cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.), minus_one = make_cuDoubleComplex(-1., 0.);

  thrust::device_ptr<const thrust::complex<double>> u_ptr = thrust::device_ptr<const thrust::complex<double>>(reinterpret_cast<const thrust::complex<double>*>(&Udata[D * block]));
  thrust::device_ptr<thrust::complex<double>> v_ptr = thrust::device_ptr<thrust::complex<double>>(reinterpret_cast<thrust::complex<double>*>(Vdata));

  auto mapV = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), swapXY(bdim, block));
  auto mapD = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), StridedBlock(rank, rank, bdim, reinterpret_cast<thrust::complex<double>**>(A_SS)));
  thrust::gather(thrust::cuda::par.on(stream), mapV, mapV + block * M, thrust::make_transform_iterator(u_ptr, conjugateDouble()), v_ptr);

  cublasZgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, A_SS, bdim, &zero, B, bdim, M);
  cublasZgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, B, bdim, &zero, A_SS, bdim, M);

  cublasZgetrfBatched(cublasH, rdim, A_RR, bdim, ipiv, info, M);
  cublasZgetrsBatched(cublasH, CUBLAS_OP_N, rdim, bdim, A_RR, bdim, ipiv, V_R, bdim, &info_host, M);

  if (0 < rank) {
    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, bdim, &one, V_R, bdim, B, bdim, &zero, A_RS, bdim, M);

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      cublasZgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, &U[i], bdim, &A_SS[i], bdim, &zero, B, bdim, len);
      cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, &V[i], bdim, B, bdim, &zero, &A_SS[i], bdim, len);
    }
    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, A_SR_Rows, bdim, A_RS, bdim, &one, A_SS, bdim, lenA);

    thrust::for_each(thrust::cuda::par.on(stream), inc_iter, inc_iter + (rdim * rank * M), copyFunc(rdim, rank, const_cast<const cuDoubleComplex**>(A_RS), bdim, &B[D], bdim));
    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rdim, rdim, bdim, &one, V_R, bdim, U_R, bdim, &zero, B_I_Cols, bdim, M);
    cudaMemcpyAsync(&R[D * block], &Bdata[D * block], M * block * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    comm.neighbor_bcast(R, Bsizes.data());
    cudaMemcpyAsync(Bdata, R, N * block * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);

    if (M < lenA)
      cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &A_SR[M], bdim, &B_Cols[M], bdim, &one, &A_SS[M], bdim, lenA - M);

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, rdim, &one, &B_I_Cols[i], bdim, &A_SR[i], bdim, &zero, B, bdim, len);
      cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &A_SR[i], bdim, B, bdim, &zero, &ACC[i], rank, len);
    }

    while (1 < reduc_len) {
      long long len = reduc_len * rank * rank * M;
      reduc_len = (reduc_len + 1) / 2;
      long long tail_start = reduc_len * rank * rank * M;
      long long tail_len = len - tail_start;
      cublasZaxpy(cublasH, tail_len, &one, &ACdata[tail_start], 1, ACdata, 1);
    }
    thrust::transform(thrust::cuda::par.on(stream), mapD, mapD + (rank * rank * M), reinterpret_cast<const thrust::complex<double>*>(ACdata), mapD, thrust::plus<thrust::complex<double>>());
  }
  cudaStreamSynchronize(stream);

  cudaMemcpy(A, Adata, block * lenA * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  cudaMemcpy(&R[block * D], Vdata, block * M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
}

template<>
void compute_factorize(cublasHandle_t cublasH, long long bdim, long long rank, long long D, long long M, long long N, const long long ARows[], const long long ACols[], std::complex<float>* A, std::complex<float>* R, const std::complex<float>* Q, const ColCommMPI& comm) {
  long long block = bdim * bdim;
  long long lenA = ARows[M];
  
  cudaStream_t stream;
  cublasGetStream(cublasH, &stream);

  thrust::device_vector<long long> row_offsets(ARows, ARows + M);
  thrust::device_vector<long long> rows(lenA, 0ll);
  thrust::device_vector<long long> cols(ACols, ACols + lenA);
  thrust::device_vector<long long> dist_cols(lenA);
  thrust::device_vector<long long> keys(lenA);
  thrust::device_vector<long long> indices(lenA);
  thrust::device_vector<cuComplex*> a_ss(lenA), a_sr(lenA), a_rs(lenA), a_rr(lenA);
  thrust::device_vector<cuComplex*> u(lenA), v(lenA), u_r(M), v_r(M), b(N), b_cols(lenA), b_i_cols(lenA);
  thrust::device_vector<cuComplex*> a_sr_rows(lenA), acc(lenA), acc_final(M);

  thrust::device_vector<cuComplex> Avec(lenA * block);
  thrust::device_vector<cuComplex> Bvec(N * block);
  thrust::device_vector<cuComplex> Uvec(N * block);
  thrust::device_vector<cuComplex> Vvec(M * block);
  
  thrust::device_vector<int> Ipiv(M * bdim);
  thrust::device_vector<int> Info(M);
  std::vector<long long> Bsizes(N, block);
  comm.dataSizesToNeighborOffsets(Bsizes.data());

  auto inc_iter = thrust::make_counting_iterator(0ll);
  auto one_iter = thrust::make_constant_iterator(1ll);
  auto rwise_diag_iter = thrust::make_permutation_iterator(indices.begin(), rows.begin());

  cuComplex* Adata = thrust::raw_pointer_cast(Avec.data());
  cuComplex* Udata = thrust::raw_pointer_cast(Uvec.data());
  cuComplex* Vdata = thrust::raw_pointer_cast(Vvec.data());
  cuComplex* Bdata = thrust::raw_pointer_cast(Bvec.data());

  cudaMemcpyAsync(Udata, Q, block * N * sizeof(cuComplex), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(Adata, A, block * lenA * sizeof(cuComplex), cudaMemcpyHostToDevice, stream);

  thrust::scatter(one_iter, one_iter + (M - 1), row_offsets.begin() + 1, rows.begin()); 
  thrust::inclusive_scan(rows.begin(), rows.end(), rows.begin());
  thrust::exclusive_scan_by_key(rows.begin(), rows.end(), one_iter, dist_cols.begin(), 0ll);

  thrust::transform(rows.begin(), rows.end(), cols.begin(), keys.begin(), keysDLU(D, M, N));
  thrust::sequence(indices.begin(), indices.end(), 0);
  thrust::sort_by_key(keys.begin(), keys.end(), thrust::make_zip_iterator(rows.begin(), cols.begin(), dist_cols.begin(), indices.begin()));

  long long reduc_len = 1ll + thrust::reduce(dist_cols.begin(), dist_cols.end(), 0ll, thrust::maximum<long long>());
  thrust::device_vector<cuComplex> ACvec(reduc_len * M * rank * rank, make_cuComplex(0., 0.));
  cuComplex* ACdata = thrust::raw_pointer_cast(ACvec.data());

  long long offset_SR = rank * bdim, offset_RS = rank, offset_RR = rank * (bdim + 1);
  thrust::transform(indices.begin(), indices.end(), a_ss.begin(), setDevicePtr(Adata, block));
  thrust::transform(indices.begin(), indices.end(), a_sr.begin(), setDevicePtr(Adata + offset_SR, block));
  thrust::transform(indices.begin(), indices.end(), a_rs.begin(), setDevicePtr(Adata + offset_RS, block));
  thrust::transform(indices.begin(), indices.end(), a_rr.begin(), setDevicePtr(Adata + offset_RR, block));
  thrust::transform(cols.begin(), cols.end(), u.begin(), setDevicePtr(Udata, block));
  thrust::transform(cols.begin(), cols.begin() + M, u_r.begin(), setDevicePtr(Udata + offset_SR, block));
  thrust::transform(rows.begin(), rows.end(), v.begin(), setDevicePtr(Vdata, block));
  thrust::transform(inc_iter, inc_iter + M, v_r.begin(), setDevicePtr(Vdata + offset_RS, block));

  thrust::transform(inc_iter, inc_iter + N, b.begin(), setDevicePtr(Bdata, block));
  thrust::transform(cols.begin(), cols.end(), b_cols.begin(), setDevicePtr(Bdata, block));
  thrust::transform(cols.begin(), cols.end(), b_i_cols.begin(), setDevicePtr(Bdata + offset_SR, block));
  thrust::transform(rwise_diag_iter, rwise_diag_iter + lenA, a_sr_rows.begin(), setDevicePtr(Adata + offset_SR, block));
  thrust::transform(rows.begin(), rows.end(), dist_cols.begin(), acc.begin(), setDevicePtr(ACdata, rank * rank, M));
  thrust::transform(inc_iter, inc_iter + M, acc_final.begin(), setDevicePtr(ACdata, rank * rank));

  cuComplex** A_SS = thrust::raw_pointer_cast(a_ss.data());
  cuComplex** A_SR = thrust::raw_pointer_cast(a_sr.data());
  cuComplex** A_RS = thrust::raw_pointer_cast(a_rs.data());
  cuComplex** A_RR = thrust::raw_pointer_cast(a_rr.data());
  cuComplex** U = thrust::raw_pointer_cast(u.data());
  cuComplex** U_R = thrust::raw_pointer_cast(u_r.data());
  cuComplex** V = thrust::raw_pointer_cast(v.data());
  cuComplex** V_R = thrust::raw_pointer_cast(v_r.data());
  cuComplex** B = thrust::raw_pointer_cast(b.data());
  cuComplex** B_Cols = thrust::raw_pointer_cast(b_cols.data());
  cuComplex** B_I_Cols = thrust::raw_pointer_cast(b_i_cols.data());
  cuComplex** A_SR_Rows = thrust::raw_pointer_cast(a_sr_rows.data());
  cuComplex** ACC = thrust::raw_pointer_cast(acc.data());
  cuComplex** ACC_Final = thrust::raw_pointer_cast(acc_final.data());

  int* ipiv = thrust::raw_pointer_cast(Ipiv.data());
  int* info = thrust::raw_pointer_cast(Info.data());

  long long rdim = bdim - rank;
  int info_host = 0;
  cuComplex one = make_cuComplex(1., 0.), zero = make_cuComplex(0., 0.), minus_one = make_cuComplex(-1., 0.);

  thrust::device_ptr<const thrust::complex<float>> u_ptr = thrust::device_ptr<const thrust::complex<float>>(reinterpret_cast<const thrust::complex<float>*>(&Udata[D * block]));
  thrust::device_ptr<thrust::complex<float>> v_ptr = thrust::device_ptr<thrust::complex<float>>(reinterpret_cast<thrust::complex<float>*>(Vdata));

  auto mapV = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), swapXY(bdim, block));
  auto mapD = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), StridedBlock(rank, rank, bdim, reinterpret_cast<thrust::complex<float>**>(A_SS)));
  thrust::gather(thrust::cuda::par.on(stream), mapV, mapV + block * M, thrust::make_transform_iterator(u_ptr, conjugateFloat()), v_ptr);

  cublasCgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, A_SS, bdim, &zero, B, bdim, M);
  cublasCgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, B, bdim, &zero, A_SS, bdim, M);

  cublasCgetrfBatched(cublasH, rdim, A_RR, bdim, ipiv, info, M);
  cublasCgetrsBatched(cublasH, CUBLAS_OP_N, rdim, bdim, A_RR, bdim, ipiv, V_R, bdim, &info_host, M);

  if (0 < rank) {
    cublasCgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, bdim, &one, V_R, bdim, B, bdim, &zero, A_RS, bdim, M);

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      cublasCgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, &U[i], bdim, &A_SS[i], bdim, &zero, B, bdim, len);
      cublasCgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, &V[i], bdim, B, bdim, &zero, &A_SS[i], bdim, len);
    }
    cublasCgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, A_SR_Rows, bdim, A_RS, bdim, &one, A_SS, bdim, lenA);

    thrust::for_each(thrust::cuda::par.on(stream), inc_iter, inc_iter + (rdim * rank * M), copyFunc(rdim, rank, const_cast<const cuComplex**>(A_RS), bdim, &B[D], bdim));
    cublasCgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rdim, rdim, bdim, &one, V_R, bdim, U_R, bdim, &zero, B_I_Cols, bdim, M);
    cudaMemcpyAsync(&R[D * block], &Bdata[D * block], M * block * sizeof(cuComplex), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    comm.neighbor_bcast(R, Bsizes.data());
    cudaMemcpyAsync(Bdata, R, N * block * sizeof(cuComplex), cudaMemcpyHostToDevice, stream);

    if (M < lenA)
      cublasCgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &A_SR[M], bdim, &B_Cols[M], bdim, &one, &A_SS[M], bdim, lenA - M);

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      cublasCgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, rdim, &one, &B_I_Cols[i], bdim, &A_SR[i], bdim, &zero, B, bdim, len);
      cublasCgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &A_SR[i], bdim, B, bdim, &zero, &ACC[i], rank, len);
    }

    while (1 < reduc_len) {
      long long len = reduc_len * rank * rank * M;
      reduc_len = (reduc_len + 1) / 2;
      long long tail_start = reduc_len * rank * rank * M;
      long long tail_len = len - tail_start;
      cublasCaxpy(cublasH, tail_len, &one, &ACdata[tail_start], 1, ACdata, 1);
    }
    thrust::transform(thrust::cuda::par.on(stream), mapD, mapD + (rank * rank * M), reinterpret_cast<const thrust::complex<float>*>(ACdata), mapD, thrust::plus<thrust::complex<float>>());
  }
  cudaStreamSynchronize(stream);

  cudaMemcpy(A, Adata, block * lenA * sizeof(cuComplex), cudaMemcpyDeviceToHost);
  cudaMemcpy(&R[block * D], Vdata, block * M * sizeof(cuComplex), cudaMemcpyDeviceToHost);
}

template<>
void compute_factorize(cublasHandle_t cublasH, long long bdim, long long rank, long long D, long long M, long long N, const long long ARows[], const long long ACols[], double* A, double* R, const double* Q, const ColCommMPI& comm) {
  long long block = bdim * bdim;
  long long lenA = ARows[M];
  
  cudaStream_t stream;
  cublasGetStream(cublasH, &stream);

  thrust::device_vector<long long> row_offsets(ARows, ARows + M);
  thrust::device_vector<long long> rows(lenA, 0ll);
  thrust::device_vector<long long> cols(ACols, ACols + lenA);
  thrust::device_vector<long long> dist_cols(lenA);
  thrust::device_vector<long long> keys(lenA);
  thrust::device_vector<long long> indices(lenA);
  thrust::device_vector<double*> a_ss(lenA), a_sr(lenA), a_rs(lenA), a_rr(lenA);
  thrust::device_vector<double*> u(lenA), v(lenA), u_r(M), v_r(M), b(N), b_cols(lenA), b_i_cols(lenA);
  thrust::device_vector<double*> a_sr_rows(lenA), acc(lenA), acc_final(M);

  thrust::device_vector<double> Avec(lenA * block);
  thrust::device_vector<double> Bvec(N * block);
  thrust::device_vector<double> Uvec(N * block);
  thrust::device_vector<double> Vvec(M * block);
  
  thrust::device_vector<int> Ipiv(M * bdim);
  thrust::device_vector<int> Info(M);
  std::vector<long long> Bsizes(N, block);
  comm.dataSizesToNeighborOffsets(Bsizes.data());

  auto inc_iter = thrust::make_counting_iterator(0ll);
  auto one_iter = thrust::make_constant_iterator(1ll);
  auto rwise_diag_iter = thrust::make_permutation_iterator(indices.begin(), rows.begin());

  double* Adata = thrust::raw_pointer_cast(Avec.data());
  double* Udata = thrust::raw_pointer_cast(Uvec.data());
  double* Vdata = thrust::raw_pointer_cast(Vvec.data());
  double* Bdata = thrust::raw_pointer_cast(Bvec.data());

  cudaMemcpyAsync(Udata, Q, block * N * sizeof(double), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(Adata, A, block * lenA * sizeof(double), cudaMemcpyHostToDevice, stream);

  thrust::scatter(one_iter, one_iter + (M - 1), row_offsets.begin() + 1, rows.begin()); 
  thrust::inclusive_scan(rows.begin(), rows.end(), rows.begin());
  thrust::exclusive_scan_by_key(rows.begin(), rows.end(), one_iter, dist_cols.begin(), 0ll);

  thrust::transform(rows.begin(), rows.end(), cols.begin(), keys.begin(), keysDLU(D, M, N));
  thrust::sequence(indices.begin(), indices.end(), 0);
  thrust::sort_by_key(keys.begin(), keys.end(), thrust::make_zip_iterator(rows.begin(), cols.begin(), dist_cols.begin(), indices.begin()));

  long long reduc_len = 1ll + thrust::reduce(dist_cols.begin(), dist_cols.end(), 0ll, thrust::maximum<long long>());
  thrust::device_vector<double> ACvec(reduc_len * M * rank * rank, 0.);
  double* ACdata = thrust::raw_pointer_cast(ACvec.data());

  long long offset_SR = rank * bdim, offset_RS = rank, offset_RR = rank * (bdim + 1);
  thrust::transform(indices.begin(), indices.end(), a_ss.begin(), setDevicePtr(Adata, block));
  thrust::transform(indices.begin(), indices.end(), a_sr.begin(), setDevicePtr(Adata + offset_SR, block));
  thrust::transform(indices.begin(), indices.end(), a_rs.begin(), setDevicePtr(Adata + offset_RS, block));
  thrust::transform(indices.begin(), indices.end(), a_rr.begin(), setDevicePtr(Adata + offset_RR, block));
  thrust::transform(cols.begin(), cols.end(), u.begin(), setDevicePtr(Udata, block));
  thrust::transform(cols.begin(), cols.begin() + M, u_r.begin(), setDevicePtr(Udata + offset_SR, block));
  thrust::transform(rows.begin(), rows.end(), v.begin(), setDevicePtr(Vdata, block));
  thrust::transform(inc_iter, inc_iter + M, v_r.begin(), setDevicePtr(Vdata + offset_RS, block));

  thrust::transform(inc_iter, inc_iter + N, b.begin(), setDevicePtr(Bdata, block));
  thrust::transform(cols.begin(), cols.end(), b_cols.begin(), setDevicePtr(Bdata, block));
  thrust::transform(cols.begin(), cols.end(), b_i_cols.begin(), setDevicePtr(Bdata + offset_SR, block));
  thrust::transform(rwise_diag_iter, rwise_diag_iter + lenA, a_sr_rows.begin(), setDevicePtr(Adata + offset_SR, block));
  thrust::transform(rows.begin(), rows.end(), dist_cols.begin(), acc.begin(), setDevicePtr(ACdata, rank * rank, M));
  thrust::transform(inc_iter, inc_iter + M, acc_final.begin(), setDevicePtr(ACdata, rank * rank));

  double** A_SS = thrust::raw_pointer_cast(a_ss.data());
  double** A_SR = thrust::raw_pointer_cast(a_sr.data());
  double** A_RS = thrust::raw_pointer_cast(a_rs.data());
  double** A_RR = thrust::raw_pointer_cast(a_rr.data());
  double** U = thrust::raw_pointer_cast(u.data());
  double** U_R = thrust::raw_pointer_cast(u_r.data());
  double** V = thrust::raw_pointer_cast(v.data());
  double** V_R = thrust::raw_pointer_cast(v_r.data());
  double** B = thrust::raw_pointer_cast(b.data());
  double** B_Cols = thrust::raw_pointer_cast(b_cols.data());
  double** B_I_Cols = thrust::raw_pointer_cast(b_i_cols.data());
  double** A_SR_Rows = thrust::raw_pointer_cast(a_sr_rows.data());
  double** ACC = thrust::raw_pointer_cast(acc.data());
  double** ACC_Final = thrust::raw_pointer_cast(acc_final.data());

  int* ipiv = thrust::raw_pointer_cast(Ipiv.data());
  int* info = thrust::raw_pointer_cast(Info.data());

  long long rdim = bdim - rank;
  int info_host = 0;
  double one = 1, zero = 0, minus_one = -1;

  thrust::device_ptr<const double> u_ptr = thrust::device_ptr<const double>(reinterpret_cast<const double*>(&Udata[D * block]));
  thrust::device_ptr<double> v_ptr = thrust::device_ptr<double>(reinterpret_cast<double*>(Vdata));

  auto mapV = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), swapXY(bdim, block));
  auto mapD = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), StridedBlock(rank, rank, bdim, reinterpret_cast<double**>(A_SS)));
  thrust::gather(thrust::cuda::par.on(stream), mapV, mapV + block * M, u_ptr, v_ptr);

  cublasDgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, A_SS, bdim, &zero, B, bdim, M);
  cublasDgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, B, bdim, &zero, A_SS, bdim, M);

  cublasDgetrfBatched(cublasH, rdim, A_RR, bdim, ipiv, info, M);
  cublasDgetrsBatched(cublasH, CUBLAS_OP_N, rdim, bdim, A_RR, bdim, ipiv, V_R, bdim, &info_host, M);

  if (0 < rank) {
    cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, bdim, &one, V_R, bdim, B, bdim, &zero, A_RS, bdim, M);

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      cublasDgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, &U[i], bdim, &A_SS[i], bdim, &zero, B, bdim, len);
      cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, &V[i], bdim, B, bdim, &zero, &A_SS[i], bdim, len);
    }
    cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, A_SR_Rows, bdim, A_RS, bdim, &one, A_SS, bdim, lenA);

    thrust::for_each(thrust::cuda::par.on(stream), inc_iter, inc_iter + (rdim * rank * M), copyFunc(rdim, rank, const_cast<const double**>(A_RS), bdim, &B[D], bdim));
    cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rdim, rdim, bdim, &one, V_R, bdim, U_R, bdim, &zero, B_I_Cols, bdim, M);
    cudaMemcpyAsync(&R[D * block], &Bdata[D * block], M * block * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    comm.neighbor_bcast(R, Bsizes.data());
    cudaMemcpyAsync(Bdata, R, N * block * sizeof(double), cudaMemcpyHostToDevice, stream);

    if (M < lenA)
      cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &A_SR[M], bdim, &B_Cols[M], bdim, &one, &A_SS[M], bdim, lenA - M);

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, rdim, &one, &B_I_Cols[i], bdim, &A_SR[i], bdim, &zero, B, bdim, len);
      cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &A_SR[i], bdim, B, bdim, &zero, &ACC[i], rank, len);
    }

    while (1 < reduc_len) {
      long long len = reduc_len * rank * rank * M;
      reduc_len = (reduc_len + 1) / 2;
      long long tail_start = reduc_len * rank * rank * M;
      long long tail_len = len - tail_start;
      cublasDaxpy(cublasH, tail_len, &one, &ACdata[tail_start], 1, ACdata, 1);
    }
    thrust::transform(thrust::cuda::par.on(stream), mapD, mapD + (rank * rank * M), reinterpret_cast<const double*>(ACdata), mapD, thrust::plus<double>());
  }
  cudaStreamSynchronize(stream);

  cudaMemcpy(A, Adata, block * lenA * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&R[block * D], Vdata, block * M * sizeof(double), cudaMemcpyDeviceToHost);
}

template<>
void compute_factorize(cublasHandle_t cublasH, long long bdim, long long rank, long long D, long long M, long long N, const long long ARows[], const long long ACols[], float* A, float* R, const float* Q, const ColCommMPI& comm) {
  long long block = bdim * bdim;
  long long lenA = ARows[M];
  
  cudaStream_t stream;
  cublasGetStream(cublasH, &stream);

  thrust::device_vector<long long> row_offsets(ARows, ARows + M);
  thrust::device_vector<long long> rows(lenA, 0ll);
  thrust::device_vector<long long> cols(ACols, ACols + lenA);
  thrust::device_vector<long long> dist_cols(lenA);
  thrust::device_vector<long long> keys(lenA);
  thrust::device_vector<long long> indices(lenA);
  thrust::device_vector<float*> a_ss(lenA), a_sr(lenA), a_rs(lenA), a_rr(lenA);
  thrust::device_vector<float*> u(lenA), v(lenA), u_r(M), v_r(M), b(N), b_cols(lenA), b_i_cols(lenA);
  thrust::device_vector<float*> a_sr_rows(lenA), acc(lenA), acc_final(M);

  thrust::device_vector<float> Avec(lenA * block);
  thrust::device_vector<float> Bvec(N * block);
  thrust::device_vector<float> Uvec(N * block);
  thrust::device_vector<float> Vvec(M * block);
  
  thrust::device_vector<int> Ipiv(M * bdim);
  thrust::device_vector<int> Info(M);
  std::vector<long long> Bsizes(N, block);
  comm.dataSizesToNeighborOffsets(Bsizes.data());

  auto inc_iter = thrust::make_counting_iterator(0ll);
  auto one_iter = thrust::make_constant_iterator(1ll);
  auto rwise_diag_iter = thrust::make_permutation_iterator(indices.begin(), rows.begin());

  float* Adata = thrust::raw_pointer_cast(Avec.data());
  float* Udata = thrust::raw_pointer_cast(Uvec.data());
  float* Vdata = thrust::raw_pointer_cast(Vvec.data());
  float* Bdata = thrust::raw_pointer_cast(Bvec.data());

  cudaMemcpyAsync(Udata, Q, block * N * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(Adata, A, block * lenA * sizeof(float), cudaMemcpyHostToDevice, stream);

  thrust::scatter(one_iter, one_iter + (M - 1), row_offsets.begin() + 1, rows.begin()); 
  thrust::inclusive_scan(rows.begin(), rows.end(), rows.begin());
  thrust::exclusive_scan_by_key(rows.begin(), rows.end(), one_iter, dist_cols.begin(), 0ll);

  thrust::transform(rows.begin(), rows.end(), cols.begin(), keys.begin(), keysDLU(D, M, N));
  thrust::sequence(indices.begin(), indices.end(), 0);
  thrust::sort_by_key(keys.begin(), keys.end(), thrust::make_zip_iterator(rows.begin(), cols.begin(), dist_cols.begin(), indices.begin()));

  long long reduc_len = 1ll + thrust::reduce(dist_cols.begin(), dist_cols.end(), 0ll, thrust::maximum<long long>());
  thrust::device_vector<float> ACvec(reduc_len * M * rank * rank, 0.);
  float* ACdata = thrust::raw_pointer_cast(ACvec.data());

  long long offset_SR = rank * bdim, offset_RS = rank, offset_RR = rank * (bdim + 1);
  thrust::transform(indices.begin(), indices.end(), a_ss.begin(), setDevicePtr(Adata, block));
  thrust::transform(indices.begin(), indices.end(), a_sr.begin(), setDevicePtr(Adata + offset_SR, block));
  thrust::transform(indices.begin(), indices.end(), a_rs.begin(), setDevicePtr(Adata + offset_RS, block));
  thrust::transform(indices.begin(), indices.end(), a_rr.begin(), setDevicePtr(Adata + offset_RR, block));
  thrust::transform(cols.begin(), cols.end(), u.begin(), setDevicePtr(Udata, block));
  thrust::transform(cols.begin(), cols.begin() + M, u_r.begin(), setDevicePtr(Udata + offset_SR, block));
  thrust::transform(rows.begin(), rows.end(), v.begin(), setDevicePtr(Vdata, block));
  thrust::transform(inc_iter, inc_iter + M, v_r.begin(), setDevicePtr(Vdata + offset_RS, block));

  thrust::transform(inc_iter, inc_iter + N, b.begin(), setDevicePtr(Bdata, block));
  thrust::transform(cols.begin(), cols.end(), b_cols.begin(), setDevicePtr(Bdata, block));
  thrust::transform(cols.begin(), cols.end(), b_i_cols.begin(), setDevicePtr(Bdata + offset_SR, block));
  thrust::transform(rwise_diag_iter, rwise_diag_iter + lenA, a_sr_rows.begin(), setDevicePtr(Adata + offset_SR, block));
  thrust::transform(rows.begin(), rows.end(), dist_cols.begin(), acc.begin(), setDevicePtr(ACdata, rank * rank, M));
  thrust::transform(inc_iter, inc_iter + M, acc_final.begin(), setDevicePtr(ACdata, rank * rank));

  float** A_SS = thrust::raw_pointer_cast(a_ss.data());
  float** A_SR = thrust::raw_pointer_cast(a_sr.data());
  float** A_RS = thrust::raw_pointer_cast(a_rs.data());
  float** A_RR = thrust::raw_pointer_cast(a_rr.data());
  float** U = thrust::raw_pointer_cast(u.data());
  float** U_R = thrust::raw_pointer_cast(u_r.data());
  float** V = thrust::raw_pointer_cast(v.data());
  float** V_R = thrust::raw_pointer_cast(v_r.data());
  float** B = thrust::raw_pointer_cast(b.data());
  float** B_Cols = thrust::raw_pointer_cast(b_cols.data());
  float** B_I_Cols = thrust::raw_pointer_cast(b_i_cols.data());
  float** A_SR_Rows = thrust::raw_pointer_cast(a_sr_rows.data());
  float** ACC = thrust::raw_pointer_cast(acc.data());
  float** ACC_Final = thrust::raw_pointer_cast(acc_final.data());

  int* ipiv = thrust::raw_pointer_cast(Ipiv.data());
  int* info = thrust::raw_pointer_cast(Info.data());

  long long rdim = bdim - rank;
  int info_host = 0;
  float one = 1, zero = 0, minus_one = -1;

  thrust::device_ptr<const float> u_ptr = thrust::device_ptr<const float>(reinterpret_cast<const float*>(&Udata[D * block]));
  thrust::device_ptr<float> v_ptr = thrust::device_ptr<float>(reinterpret_cast<float*>(Vdata));

  auto mapV = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), swapXY(bdim, block));
  auto mapD = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), StridedBlock(rank, rank, bdim, reinterpret_cast<float**>(A_SS)));
  thrust::gather(thrust::cuda::par.on(stream), mapV, mapV + block * M, u_ptr, v_ptr);

  cublasSgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, A_SS, bdim, &zero, B, bdim, M);
  cublasSgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, B, bdim, &zero, A_SS, bdim, M);

  cublasSgetrfBatched(cublasH, rdim, A_RR, bdim, ipiv, info, M);
  cublasSgetrsBatched(cublasH, CUBLAS_OP_N, rdim, bdim, A_RR, bdim, ipiv, V_R, bdim, &info_host, M);

  if (0 < rank) {
    cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, bdim, &one, V_R, bdim, B, bdim, &zero, A_RS, bdim, M);

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      cublasSgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, &U[i], bdim, &A_SS[i], bdim, &zero, B, bdim, len);
      cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, &V[i], bdim, B, bdim, &zero, &A_SS[i], bdim, len);
    }
    cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, A_SR_Rows, bdim, A_RS, bdim, &one, A_SS, bdim, lenA);

    thrust::for_each(thrust::cuda::par.on(stream), inc_iter, inc_iter + (rdim * rank * M), copyFunc(rdim, rank, const_cast<const float**>(A_RS), bdim, &B[D], bdim));
    cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rdim, rdim, bdim, &one, V_R, bdim, U_R, bdim, &zero, B_I_Cols, bdim, M);
    cudaMemcpyAsync(&R[D * block], &Bdata[D * block], M * block * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    comm.neighbor_bcast(R, Bsizes.data());
    cudaMemcpyAsync(Bdata, R, N * block * sizeof(float), cudaMemcpyHostToDevice, stream);

    if (M < lenA)
      cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &A_SR[M], bdim, &B_Cols[M], bdim, &one, &A_SS[M], bdim, lenA - M);

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, rdim, &one, &B_I_Cols[i], bdim, &A_SR[i], bdim, &zero, B, bdim, len);
      cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &A_SR[i], bdim, B, bdim, &zero, &ACC[i], rank, len);
    }

    while (1 < reduc_len) {
      long long len = reduc_len * rank * rank * M;
      reduc_len = (reduc_len + 1) / 2;
      long long tail_start = reduc_len * rank * rank * M;
      long long tail_len = len - tail_start;
      cublasSaxpy(cublasH, tail_len, &one, &ACdata[tail_start], 1, ACdata, 1);
    }
    thrust::transform(thrust::cuda::par.on(stream), mapD, mapD + (rank * rank * M), reinterpret_cast<const float*>(ACdata), mapD, thrust::plus<float>());
  }
  cudaStreamSynchronize(stream);

  cudaMemcpy(A, Adata, block * lenA * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&R[block * D], Vdata, block * M * sizeof(float), cudaMemcpyDeviceToHost);
}

void compute_factorize(const cublasComputeType_t COMP, cublasHandle_t cublasH, long long bdim, long long rank, long long D, long long M, long long N, const long long ARows[], const long long ACols[], float* A, float* R, const float* Q, const ColCommMPI& comm) {
  std::cout<<"COMP"<<std::endl;
  long long block = bdim * bdim;
  long long lenA = ARows[M];
  
  cudaStream_t stream;
  cublasGetStream(cublasH, &stream);

  thrust::device_vector<long long> row_offsets(ARows, ARows + M);
  thrust::device_vector<long long> rows(lenA, 0ll);
  thrust::device_vector<long long> cols(ACols, ACols + lenA);
  thrust::device_vector<long long> dist_cols(lenA);
  thrust::device_vector<long long> keys(lenA);
  thrust::device_vector<long long> indices(lenA);
  thrust::device_vector<float*> a_ss(lenA), a_sr(lenA), a_rs(lenA), a_rr(lenA);
  thrust::device_vector<float*> u(lenA), v(lenA), u_r(M), v_r(M), b(N), b_cols(lenA), b_i_cols(lenA);
  thrust::device_vector<float*> a_sr_rows(lenA), acc(lenA), acc_final(M);

  thrust::device_vector<float> Avec(lenA * block);
  thrust::device_vector<float> Bvec(N * block);
  thrust::device_vector<float> Uvec(N * block);
  thrust::device_vector<float> Vvec(M * block);
  
  thrust::device_vector<int> Ipiv(M * bdim);
  thrust::device_vector<int> Info(M);
  std::vector<long long> Bsizes(N, block);
  comm.dataSizesToNeighborOffsets(Bsizes.data());

  auto inc_iter = thrust::make_counting_iterator(0ll);
  auto one_iter = thrust::make_constant_iterator(1ll);
  auto rwise_diag_iter = thrust::make_permutation_iterator(indices.begin(), rows.begin());

  float* Adata = thrust::raw_pointer_cast(Avec.data());
  float* Udata = thrust::raw_pointer_cast(Uvec.data());
  float* Vdata = thrust::raw_pointer_cast(Vvec.data());
  float* Bdata = thrust::raw_pointer_cast(Bvec.data());

  cudaMemcpyAsync(Udata, Q, block * N * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(Adata, A, block * lenA * sizeof(float), cudaMemcpyHostToDevice, stream);

  thrust::scatter(one_iter, one_iter + (M - 1), row_offsets.begin() + 1, rows.begin()); 
  thrust::inclusive_scan(rows.begin(), rows.end(), rows.begin());
  thrust::exclusive_scan_by_key(rows.begin(), rows.end(), one_iter, dist_cols.begin(), 0ll);

  thrust::transform(rows.begin(), rows.end(), cols.begin(), keys.begin(), keysDLU(D, M, N));
  thrust::sequence(indices.begin(), indices.end(), 0);
  thrust::sort_by_key(keys.begin(), keys.end(), thrust::make_zip_iterator(rows.begin(), cols.begin(), dist_cols.begin(), indices.begin()));

  long long reduc_len = 1ll + thrust::reduce(dist_cols.begin(), dist_cols.end(), 0ll, thrust::maximum<long long>());
  thrust::device_vector<float> ACvec(reduc_len * M * rank * rank, 0.);
  float* ACdata = thrust::raw_pointer_cast(ACvec.data());

  long long offset_SR = rank * bdim, offset_RS = rank, offset_RR = rank * (bdim + 1);
  thrust::transform(indices.begin(), indices.end(), a_ss.begin(), setDevicePtr(Adata, block));
  thrust::transform(indices.begin(), indices.end(), a_sr.begin(), setDevicePtr(Adata + offset_SR, block));
  thrust::transform(indices.begin(), indices.end(), a_rs.begin(), setDevicePtr(Adata + offset_RS, block));
  thrust::transform(indices.begin(), indices.end(), a_rr.begin(), setDevicePtr(Adata + offset_RR, block));
  thrust::transform(cols.begin(), cols.end(), u.begin(), setDevicePtr(Udata, block));
  thrust::transform(cols.begin(), cols.begin() + M, u_r.begin(), setDevicePtr(Udata + offset_SR, block));
  thrust::transform(rows.begin(), rows.end(), v.begin(), setDevicePtr(Vdata, block));
  thrust::transform(inc_iter, inc_iter + M, v_r.begin(), setDevicePtr(Vdata + offset_RS, block));

  thrust::transform(inc_iter, inc_iter + N, b.begin(), setDevicePtr(Bdata, block));
  thrust::transform(cols.begin(), cols.end(), b_cols.begin(), setDevicePtr(Bdata, block));
  thrust::transform(cols.begin(), cols.end(), b_i_cols.begin(), setDevicePtr(Bdata + offset_SR, block));
  thrust::transform(rwise_diag_iter, rwise_diag_iter + lenA, a_sr_rows.begin(), setDevicePtr(Adata + offset_SR, block));
  thrust::transform(rows.begin(), rows.end(), dist_cols.begin(), acc.begin(), setDevicePtr(ACdata, rank * rank, M));
  thrust::transform(inc_iter, inc_iter + M, acc_final.begin(), setDevicePtr(ACdata, rank * rank));

  float** A_SS = thrust::raw_pointer_cast(a_ss.data());
  float** A_SR = thrust::raw_pointer_cast(a_sr.data());
  float** A_RS = thrust::raw_pointer_cast(a_rs.data());
  float** A_RR = thrust::raw_pointer_cast(a_rr.data());
  float** U = thrust::raw_pointer_cast(u.data());
  float** U_R = thrust::raw_pointer_cast(u_r.data());
  float** V = thrust::raw_pointer_cast(v.data());
  float** V_R = thrust::raw_pointer_cast(v_r.data());
  float** B = thrust::raw_pointer_cast(b.data());
  float** B_Cols = thrust::raw_pointer_cast(b_cols.data());
  float** B_I_Cols = thrust::raw_pointer_cast(b_i_cols.data());
  float** A_SR_Rows = thrust::raw_pointer_cast(a_sr_rows.data());
  float** ACC = thrust::raw_pointer_cast(acc.data());
  float** ACC_Final = thrust::raw_pointer_cast(acc_final.data());

  int* ipiv = thrust::raw_pointer_cast(Ipiv.data());
  int* info = thrust::raw_pointer_cast(Info.data());

  long long rdim = bdim - rank;
  int info_host = 0;
  float one = 1, zero = 0, minus_one = -1;

  thrust::device_ptr<const float> u_ptr = thrust::device_ptr<const float>(reinterpret_cast<const float*>(&Udata[D * block]));
  thrust::device_ptr<float> v_ptr = thrust::device_ptr<float>(reinterpret_cast<float*>(Vdata));

  auto mapV = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), swapXY(bdim, block));
  auto mapD = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), StridedBlock(rank, rank, bdim, reinterpret_cast<float**>(A_SS)));
  thrust::gather(thrust::cuda::par.on(stream), mapV, mapV + block * M, u_ptr, v_ptr);

  const auto ALGO = CUBLAS_GEMM_DEFAULT;
  std::cout<<"GemmBatcvhedEs"<<std::endl;
  //cublasSgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, A_SS, bdim, &zero, B, bdim, M);
  cublasGemmBatchedEx(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(U), CUDA_R_32F, bdim, reinterpret_cast<void**>(A_SS), CUDA_R_32F, bdim, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(B), CUDA_R_32F, bdim, M, COMP, ALGO);
  //cublasSgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, B, bdim, &zero, A_SS, bdim, M);
  cublasGemmBatchedEx(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(U), CUDA_R_32F, bdim, reinterpret_cast<void**>(B), CUDA_R_32F, bdim, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(A_SS), CUDA_R_32F, bdim, M, COMP, ALGO);

  cublasSgetrfBatched(cublasH, rdim, A_RR, bdim, ipiv, info, M);
  cublasSgetrsBatched(cublasH, CUBLAS_OP_N, rdim, bdim, A_RR, bdim, ipiv, V_R, bdim, &info_host, M);

  if (0 < rank) {
    //cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, bdim, &one, V_R, bdim, B, bdim, &zero, A_RS, bdim, M);
    cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, bdim, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(V_R), CUDA_R_32F, bdim, reinterpret_cast<void**>(B), CUDA_R_32F, bdim, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(A_RS), CUDA_R_32F, bdim, M, COMP, ALGO);

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      //cublasSgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, &U[i], bdim, &A_SS[i], bdim, &zero, B, bdim, len);
      cublasGemmBatchedEx(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(&U[i]), CUDA_R_32F, bdim, reinterpret_cast<void**>(&A_SS[i]), CUDA_R_32F, bdim, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(B), CUDA_R_32F, bdim, len, COMP, ALGO);
      //cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, &V[i], bdim, B, bdim, &zero, &A_SS[i], bdim, len);
      cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(&V[i]), CUDA_R_32F, bdim, reinterpret_cast<void**>(B), CUDA_R_32F, bdim, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(&A_SS[i]), CUDA_R_32F, bdim, len, COMP, ALGO);
    }
    //cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, A_SR_Rows, bdim, A_RS, bdim, &one, A_SS, bdim, lenA);
    cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, bdim, reinterpret_cast<void*>(&minus_one), reinterpret_cast<void**>(A_SR_Rows), CUDA_R_32F, bdim, reinterpret_cast<void**>(A_RS), CUDA_R_32F, bdim, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(A_SS), CUDA_R_32F, bdim, lenA, COMP, ALGO);

    thrust::for_each(thrust::cuda::par.on(stream), inc_iter, inc_iter + (rdim * rank * M), copyFunc(rdim, rank, const_cast<const float**>(A_RS), bdim, &B[D], bdim));
    //cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rdim, rdim, bdim, &one, V_R, bdim, U_R, bdim, &zero, B_I_Cols, bdim, M);
    cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rdim, rdim, bdim, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(V_R), CUDA_R_32F, bdim, reinterpret_cast<void**>(U_R), CUDA_R_32F, bdim, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(B_I_Cols), CUDA_R_32F, bdim, M, COMP, ALGO);
    cudaMemcpyAsync(&R[D * block], &Bdata[D * block], M * block * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    comm.neighbor_bcast(R, Bsizes.data());
    cudaMemcpyAsync(Bdata, R, N * block * sizeof(float), cudaMemcpyHostToDevice, stream);

    if (M < lenA) {
      //cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &A_SR[M], bdim, &B_Cols[M], bdim, &one, &A_SS[M], bdim, lenA - M);
      cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, reinterpret_cast<void*>(&minus_one), reinterpret_cast<void**>(&A_SR[M]), CUDA_R_32F, bdim, reinterpret_cast<void**>(&B_Cols[M]), CUDA_R_32F, bdim, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(A_SS[M]), CUDA_R_32F, bdim, lenA-M, COMP, ALGO);
    }

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      //cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, rdim, &one, &B_I_Cols[i], bdim, &A_SR[i], bdim, &zero, B, bdim, len);
      cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, rdim, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(&B_I_Cols[i]), CUDA_R_32F, bdim, reinterpret_cast<void**>(&A_SR[i]), CUDA_R_32F, bdim, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(B), CUDA_R_32F, bdim, len, COMP, ALGO);
      //cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &A_SR[i], bdim, B, bdim, &zero, &ACC[i], rank, len);
      cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, reinterpret_cast<void*>(&minus_one), reinterpret_cast<void**>(&A_SR[i]), CUDA_R_32F, bdim, reinterpret_cast<void**>(B), CUDA_R_32F, bdim, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(&ACC[i]), CUDA_R_32F, rank, len, COMP, ALGO);
    }

    while (1 < reduc_len) {
      long long len = reduc_len * rank * rank * M;
      reduc_len = (reduc_len + 1) / 2;
      long long tail_start = reduc_len * rank * rank * M;
      long long tail_len = len - tail_start;
      cublasSaxpy(cublasH, tail_len, &one, &ACdata[tail_start], 1, ACdata, 1);
    }
    thrust::transform(thrust::cuda::par.on(stream), mapD, mapD + (rank * rank * M), reinterpret_cast<const float*>(ACdata), mapD, thrust::plus<float>());
  }
  cudaStreamSynchronize(stream);

  cudaMemcpy(A, Adata, block * lenA * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&R[block * D], Vdata, block * M * sizeof(float), cudaMemcpyDeviceToHost);
}
/*
template <>
void H2Factorize<cuComplex>::compute(const cublasComputeType_t COMP) {
  long long N = bdim, S = rank, R = N - S;
  long long D = lenD;
  cuComplex one = make_cuComplex(1., 0.), zero = make_cuComplex(0., 0.), minus_one = make_cuComplex(-1., 0.);
  int info_host = 0;

  const auto ALGO = CUBLAS_GEMM_DEFAULT;

  cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(U), CUDA_C_32F, N, reinterpret_cast<void**>(A_SS), CUDA_C_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(B), CUDA_C_32F, N, D, COMP, ALGO);
  cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(U), CUDA_C_32F, N, reinterpret_cast<void**>(B), CUDA_C_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(A_SS), CUDA_C_32F, N, D, COMP, ALGO);

  cublasCgetrfBatched(cublasH, R, A_RR, N, ipiv, info, D);
  cublasCgetrsBatched(cublasH, CUBLAS_OP_N, R, S, A_RR, N, ipiv, A_RS, N, &info_host, D);
  cublasCgetrsBatched(cublasH, CUBLAS_OP_N, R, N, A_RR, N, ipiv, V_R, N, &info_host, D);

  cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, S, R, reinterpret_cast<void*>(&minus_one), reinterpret_cast<void**>(A_SR), CUDA_C_32F, N, reinterpret_cast<void**>(A_RS), CUDA_C_32F, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(A_SS), CUDA_C_32F, N, D, COMP, ALGO);

  for (int64_t i = D; i < lenA; i += maxQ) {
    int64_t len = std::min(lenA - i, maxQ);
    cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(&U[i]), CUDA_C_32F, N, reinterpret_cast<void**>(&A_SS[i]), CUDA_C_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(B), CUDA_C_32F, N, len, COMP, ALGO);
    cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(&V[i]), CUDA_C_32F, N, reinterpret_cast<void**>(B), CUDA_C_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(&A_SS[i]), CUDA_C_32F, N, len, COMP, ALGO);
  }

  cudaStreamSynchronize(stream);
}

template <>
void H2Factorize<cuComplex>::compute() {
  compute(CUBLAS_COMPUTE_32F);
}

template <>
void H2Factorize<double>::compute() {
  long long N = bdim, S = rank, R = N - S;
  long long D = lenD;
  double one = 1, zero = 0, minus_one = -1;
  int info_host = 0;

  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, U, N, A_SS, N, &zero, B, N, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, U, N, B, N, &zero, A_SS, N, D);

  cublasDgetrfBatched(cublasH, R, A_RR, N, ipiv, info, D);
  cublasDgetrsBatched(cublasH, CUBLAS_OP_N, R, S, A_RR, N, ipiv, A_RS, N, &info_host, D);
  cublasDgetrsBatched(cublasH, CUBLAS_OP_N, R, N, A_RR, N, ipiv, V_R, N, &info_host, D);

  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, S, R, &minus_one, A_SR, N, A_RS, N, &one, A_SS, N, D);

  for (int64_t i = D; i < lenA; i += maxQ) {
    int64_t len = std::min(lenA - i, maxQ);
    cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, &U[i], N, &A_SS[i], N, &zero, B, N, len);
    cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, &V[i], N, B, N, &zero, &A_SS[i], N, len);
  }

  cudaStreamSynchronize(stream);
}

template <>
void H2Factorize<float>::compute(const cublasComputeType_t COMP) {
  long long N = bdim, S = rank, R = N - S;
  long long D = lenD;
  float one = 1, zero = 0, minus_one = -1;
  int info_host = 0;

  const auto ALGO = CUBLAS_GEMM_DEFAULT;

  cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(U), CUDA_R_32F, N, reinterpret_cast<void**>(A_SS), CUDA_R_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(B), CUDA_R_32F, N, D, COMP, ALGO);
  cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(U), CUDA_R_32F, N, reinterpret_cast<void**>(B), CUDA_R_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(A_SS), CUDA_R_32F, N, D, COMP, ALGO);

  cublasSgetrfBatched(cublasH, R, A_RR, N, ipiv, info, D);
  cublasSgetrsBatched(cublasH, CUBLAS_OP_N, R, S, A_RR, N, ipiv, A_RS, N, &info_host, D);
  cublasSgetrsBatched(cublasH, CUBLAS_OP_N, R, N, A_RR, N, ipiv, V_R, N, &info_host, D);

  cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, S, R, reinterpret_cast<void*>(&minus_one), reinterpret_cast<void**>(A_SR), CUDA_R_32F, N, reinterpret_cast<void**>(A_RS), CUDA_R_32F, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(A_SS), CUDA_R_32F, N, D, COMP, ALGO);

  for (int64_t i = D; i < lenA; i += maxQ) {
    int64_t len = std::min(lenA - i, maxQ);
    cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(&U[i]), CUDA_R_32F, N, reinterpret_cast<void**>(&A_SS[i]), CUDA_R_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(B), CUDA_R_32F, N, len, COMP, ALGO);
    cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(&V[i]), CUDA_R_32F, N, reinterpret_cast<void**>(B), CUDA_R_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(&A_SS[i]), CUDA_R_32F, N, len, COMP, ALGO);
  }

  cudaStreamSynchronize(stream);
}

template <>
void H2Factorize<float>::compute() {
  compute(CUBLAS_COMPUTE_32F);
}
*/




/* explicit template instantiation */
// complex double
//template class H2Factorize<cuDoubleComplex>;
//template void H2Factorize<cuDoubleComplex>::compute(const long long, const long long, const long long, const long long, const long long, const long long[], const long long[], std::complex<double>* const A, std::complex<double>* const R, const std::complex<double>* const Q);
// complex float
//template class H2Factorize<cuComplex>;
//template void H2Factorize<cuComplex>::compute(const long long, const long long, const long long, const long long, const long long, const long long[], const long long[], std::complex<float>* const A, std::complex<float>* const R, const std::complex<float>* const Q);
// double
template void compute_factorize<double>(const cublasHandle_t, const long long, const long long, const long long, const long long, const long long, const long long[], const long long[], double* const, double* const, const double* const, const ColCommMPI&);
//template class H2Factorize<double>;
//template void H2Factorize<double>::compute(const long long, const long long, const long long, const long long, const long long, const long long[], const long long[], double* const A, double* const R, const double* const Q);
// float
template void compute_factorize<float>(const cublasHandle_t, const long long, const long long, const long long, const long long, const long long, const long long[], const long long[], float* const, float* const, const float* const, const ColCommMPI&);
//template class H2Factorize<float>;
//template void H2Factorize<float>::compute(const long long, const long long, const long long, const long long, const long long, const long long[], const long long[], float* const A, float* const R, const float* const Q);

/*
template <typename DT>
H2Factorize2<DT>::H2Factorize2(long long LD, long long lenA, long long lenQ, cudaStream_t stream) : maxA(lenA), maxQ(lenQ), bdim(LD), stream(stream) {
  cublasH = nullptr;
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);
  
  cublasLtH = nullptr;
  cublasLtCreate(&cublasLtH);

  long long bsize = LD * LD * sizeof(DT);
  cudaMalloc(reinterpret_cast<void**>(&Adata), bsize * lenA);
  cudaMalloc(reinterpret_cast<void**>(&Bdata), bsize * lenQ);
  cudaMalloc(reinterpret_cast<void**>(&Udata), bsize * lenQ);
  cudaMalloc(reinterpret_cast<void**>(&Vdata), bsize * lenQ);

  long long psize = sizeof(DT*);
  cudaMalloc(reinterpret_cast<void**>(&A_SS), psize * lenA);
  cudaMalloc(reinterpret_cast<void**>(&A_SR), psize * lenA);
  cudaMalloc(reinterpret_cast<void**>(&A_RS), psize * lenA);
  cudaMalloc(reinterpret_cast<void**>(&A_RR), psize * lenA);

  cudaMalloc(reinterpret_cast<void**>(&B), psize * lenQ);
  cudaMalloc(reinterpret_cast<void**>(&U), psize * lenA);
  cudaMalloc(reinterpret_cast<void**>(&V), psize * lenA);
  cudaMalloc(reinterpret_cast<void**>(&V_R), psize * lenQ);

  cudaMalloc(reinterpret_cast<void**>(&ipiv), LD * lenQ * sizeof(int));
  cudaMalloc(reinterpret_cast<void**>(&info), lenQ * sizeof(int));

  DT** hostB;
  cudaMallocHost(reinterpret_cast<void**>(&hostB), psize * lenQ);
  for (long long i = 0; i < lenQ; i++)
    hostB[i] = &Bdata[i * LD * LD];

  cudaMemcpy(B, hostB, psize * lenQ, cudaMemcpyHostToDevice);
  Bptr = std::vector<DT*>(lenQ);
  std::memcpy(Bptr.data(), hostB, psize * lenQ);
  cudaFreeHost(hostB);
}

template <typename DT>
H2Factorize2<DT>::~H2Factorize2() {
  cudaFree(Adata);
  cudaFree(Bdata);
  cudaFree(Udata);
  cudaFree(Vdata);
  cudaFree(A_SS);
  cudaFree(A_SR);
  cudaFree(A_RS);
  cudaFree(A_RR);
  cudaFree(B);
  cudaFree(U);
  cudaFree(V);
  cudaFree(V_R);
  cudaFree(ipiv);
  cudaFree(info);
  cublasDestroy(cublasH);
  cublasLtDestroy(cublasLtH);
}

template <typename DT> template <typename OT>
void H2Factorize2<DT>::setData(long long rank, long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], const MatrixDataContainer<OT>& A, const MatrixDataContainer<OT>& Q) {
  long long block = bdim * bdim;
  lenD = M;
  lenA = std::min(maxA, A.nblocks());
  long long lenQ = std::min(maxQ, Q.nblocks());
  H2Factorize2::rank = rank;

  DT* hostU, *hostA;
  cudaMallocHost(reinterpret_cast<void**>(&hostU), block * lenQ * sizeof(DT));
  //std::fill(hostU, &hostU[block * lenQ], make_cuDoubleComplex(0., 0.));
  fill_zero(hostU, &hostU[block * lenQ]);

  for (long long i = 0; i < lenQ; i++) {
    long long m = Dims[i];
    //MKL_Zomatcopy('C', 'C', m, m, std::complex<double>(1., 0.), Q[i], m, reinterpret_cast<std::complex<double>*>(&hostU[i * block]), bdim);
    omatcopy('C', 'C', m, m, Q[i], m, &hostU[i * block], bdim);
  }
  cudaMemcpy(Udata, hostU, block * lenQ * sizeof(DT), cudaMemcpyHostToDevice);
  cudaMemcpy(Vdata, Udata, block * lenQ * sizeof(DT), cudaMemcpyDeviceToDevice);
  
  cudaFreeHost(hostU);
  cudaMallocHost(reinterpret_cast<void**>(&hostA), block * lenA * sizeof(DT));
  //std::fill(hostA, &hostA[block * lenA], make_cuDoubleComplex(0., 0.));
  fill_zero(hostA, &hostA[block * lenA]);

  std::vector<std::tuple<long long, long long, long long>> coo_list(lenA);
  for (long long y = 0; y < M; y++) {
    long long begin = ARows[y];
    long long end = ARows[y + 1];
    std::transform(&ACols[begin], &ACols[end], coo_list.begin() + begin, 
      [&](const long long& x) { return std::make_tuple(y + D, x, std::distance(ACols, &x)); });
  }

  for (long long i = 0; i < lenA; i++) {
    long long y = std::get<0>(coo_list[i]);
    long long x = std::get<1>(coo_list[i]);
    long long M = Dims[y], N = Dims[x];
    //MKL_Zomatcopy('C', 'N', M, N, std::complex<double>(1., 0.), A[i], M, reinterpret_cast<std::complex<double>*>(&hostA[i * block]), bdim);
    omatcopy('C', 'N', M, N, A[i], M, &hostA[i * block], bdim);
  }

  cudaMemcpy(Adata, hostA, block * lenA * sizeof(DT), cudaMemcpyHostToDevice);
  cudaFreeHost(hostA);

  std::stable_partition(coo_list.begin(), coo_list.end(), 
    [](std::tuple<int64_t, int64_t, int64_t> i) { return std::get<0>(i) == std::get<1>(i); });

  DT** hostAptrs, **hostUptrs, **hostVptrs;
  cudaMallocHost(reinterpret_cast<void**>(&hostAptrs), lenA * sizeof(DT*));
  cudaMallocHost(reinterpret_cast<void**>(&hostUptrs), lenA * sizeof(DT*));
  cudaMallocHost(reinterpret_cast<void**>(&hostVptrs), lenA * sizeof(DT*));

  std::transform(coo_list.begin(), coo_list.end(), hostAptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Adata + std::get<2>(i) * block; });
  std::transform(coo_list.begin(), coo_list.end(), hostUptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Udata + std::get<1>(i) * block; });
  std::transform(coo_list.begin(), coo_list.end(), hostVptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Vdata + std::get<0>(i) * block; });

  cudaMemcpy(A_SS, hostAptrs, lenA * sizeof(DT*), cudaMemcpyHostToDevice);
  cudaMemcpy(U, hostUptrs, lenA * sizeof(DT*), cudaMemcpyHostToDevice);
  cudaMemcpy(V, hostVptrs, lenA * sizeof(DT*), cudaMemcpyHostToDevice);

  long long offset_SR = rank * bdim, offset_RS = rank, offset_RR = rank * (bdim + 1);
  std::transform(coo_list.begin(), coo_list.end(), hostAptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Adata + std::get<2>(i) * block + offset_SR; });
  std::transform(coo_list.begin(), coo_list.end(), hostUptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Adata + std::get<2>(i) * block + offset_RS; });
  std::transform(coo_list.begin(), coo_list.end(), hostVptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Adata + std::get<2>(i) * block + offset_RR; });

  cudaMemcpy(A_SR, hostAptrs, lenA * sizeof(DT*), cudaMemcpyHostToDevice);
  cudaMemcpy(A_RS, hostUptrs, lenA * sizeof(DT*), cudaMemcpyHostToDevice);
  cudaMemcpy(A_RR, hostVptrs, lenA * sizeof(DT*), cudaMemcpyHostToDevice);

  std::transform(coo_list.begin(), coo_list.begin() + M, hostVptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Vdata + std::get<0>(i) * block + offset_RS; });
  cudaMemcpy(V_R, hostVptrs, M * sizeof(DT*), cudaMemcpyHostToDevice);

  cudaFreeHost(hostAptrs);
  cudaFreeHost(hostUptrs);
  cudaFreeHost(hostVptrs);
}

// specialization for experimental purposes
template <> template<>
void H2Factorize2<float>::setData(long long rank, long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], const MatrixDataContainer<float>& A, const MatrixDataContainer<float>& Q) {
  long long block = bdim * bdim;
  lenD = M;
  lenA = std::min(maxA, A.nblocks());
  long long lenQ = std::min(maxQ, Q.nblocks());
  H2Factorize2::rank = rank;

  float* hostU, *hostA;
  cudaMallocHost(reinterpret_cast<void**>(&hostU), block * lenQ * sizeof(float));
  //std::fill(hostU, &hostU[block * lenQ], make_cuDoubleComplex(0., 0.));
  fill_zero(hostU, &hostU[block * lenQ]);

  for (long long i = 0; i < lenQ; i++) {
    long long m = Dims[i];
    //MKL_Zomatcopy('C', 'C', m, m, std::complex<double>(1., 0.), Q[i], m, reinterpret_cast<std::complex<double>*>(&hostU[i * block]), bdim);
    omatcopy('C', 'C', m, m, Q[i], m, &hostU[i * block], bdim);
  }
  cudaMemcpy(Udata, hostU, block * lenQ * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Vdata, Udata, block * lenQ * sizeof(float), cudaMemcpyDeviceToDevice);
  
  cudaFreeHost(hostU);
  cudaMallocHost(reinterpret_cast<void**>(&hostA), block * lenA * sizeof(float));
  //std::fill(hostA, &hostA[block * lenA], make_cuDoubleComplex(0., 0.));
  fill_zero(hostA, &hostA[block * lenA]);

  std::vector<std::tuple<long long, long long, long long>> coo_list(lenA);
  for (long long y = 0; y < M; y++) {
    long long begin = ARows[y];
    long long end = ARows[y + 1];
    std::transform(&ACols[begin], &ACols[end], coo_list.begin() + begin, 
      [&](const long long& x) { return std::make_tuple(y + D, x, std::distance(ACols, &x)); });
  }

  for (long long i = 0; i < lenA; i++) {
    long long y = std::get<0>(coo_list[i]);
    long long x = std::get<1>(coo_list[i]);
    long long M = Dims[y], N = Dims[x];
    //MKL_Zomatcopy('C', 'N', M, N, std::complex<double>(1., 0.), A[i], M, reinterpret_cast<std::complex<double>*>(&hostA[i * block]), bdim);
    omatcopy('C', 'N', M, N, A[i], M, &hostA[i * block], bdim);
  }

  cudaMemcpy(Adata, hostA, block * lenA * sizeof(float), cudaMemcpyHostToDevice);
  cudaFreeHost(hostA);

  std::stable_partition(coo_list.begin(), coo_list.end(), 
    [](std::tuple<int64_t, int64_t, int64_t> i) { return std::get<0>(i) == std::get<1>(i); });

  float** hostAptrs, **hostUptrs, **hostVptrs;
  cudaMallocHost(reinterpret_cast<void**>(&hostAptrs), lenA * sizeof(float*));
  cudaMallocHost(reinterpret_cast<void**>(&hostUptrs), lenA * sizeof(float*));
  cudaMallocHost(reinterpret_cast<void**>(&hostVptrs), lenA * sizeof(float*));

  std::transform(coo_list.begin(), coo_list.end(), hostAptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Adata + std::get<2>(i) * block; });
  std::transform(coo_list.begin(), coo_list.end(), hostUptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Udata + std::get<1>(i) * block; });
  std::transform(coo_list.begin(), coo_list.end(), hostVptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Vdata + std::get<0>(i) * block; });

  cudaMemcpy(A_SS, hostAptrs, lenA * sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(U, hostUptrs, lenA * sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(V, hostVptrs, lenA * sizeof(float*), cudaMemcpyHostToDevice);
  Aptr_SS = std::vector<float*>(lenA);
  Uptr = std::vector<float*>(lenA);
  Vptr = std::vector<float*>(lenA);
  std::memcpy(Aptr_SS.data(), hostAptrs, lenA * sizeof(float*));
  std::memcpy(Uptr.data(), hostUptrs, lenA * sizeof(float*));
  std::memcpy(Vptr.data(), hostVptrs, lenA * sizeof(float*));


  long long offset_SR = rank * bdim, offset_RS = rank, offset_RR = rank * (bdim + 1);
  std::transform(coo_list.begin(), coo_list.end(), hostAptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Adata + std::get<2>(i) * block + offset_SR; });
  std::transform(coo_list.begin(), coo_list.end(), hostUptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Adata + std::get<2>(i) * block + offset_RS; });
  std::transform(coo_list.begin(), coo_list.end(), hostVptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Adata + std::get<2>(i) * block + offset_RR; });

  cudaMemcpy(A_SR, hostAptrs, lenA * sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(A_RS, hostUptrs, lenA * sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(A_RR, hostVptrs, lenA * sizeof(float*), cudaMemcpyHostToDevice);
  Aptr_RS = std::vector<float*>(lenA);
  Aptr_SR = std::vector<float*>(lenA);
  std::memcpy(Aptr_SR.data(), hostAptrs, lenA * sizeof(float*));
  std::memcpy(Aptr_RS.data(), hostUptrs, lenA * sizeof(float*));

  std::transform(coo_list.begin(), coo_list.begin() + M, hostVptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Vdata + std::get<0>(i) * block + offset_RS; });
  cudaMemcpy(V_R, hostVptrs, M * sizeof(float*), cudaMemcpyHostToDevice);

  cudaFreeHost(hostAptrs);
  cudaFreeHost(hostUptrs);
  cudaFreeHost(hostVptrs);
}

// needs explicit specialization due to cuBLAS calls
template <>
void H2Factorize2<cuDoubleComplex>::compute() {
  long long N = bdim, S = rank, R = N - S;
  long long D = lenD;
  cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.), minus_one = make_cuDoubleComplex(-1., 0.);
  int info_host = 0;

  cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, U, N, A_SS, N, &zero, B, N, D);
  cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, U, N, B, N, &zero, A_SS, N, D);

  cublasZgetrfBatched(cublasH, R, A_RR, N, ipiv, info, D);
  cublasZgetrsBatched(cublasH, CUBLAS_OP_N, R, S, A_RR, N, ipiv, A_RS, N, &info_host, D);
  cublasZgetrsBatched(cublasH, CUBLAS_OP_N, R, N, A_RR, N, ipiv, V_R, N, &info_host, D);

  cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, S, R, &minus_one, A_SR, N, A_RS, N, &one, A_SS, N, D);

  for (int64_t i = D; i < lenA; i += maxQ) {
    int64_t len = std::min(lenA - i, maxQ);
    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, &U[i], N, &A_SS[i], N, &zero, B, N, len);
    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, &V[i], N, B, N, &zero, &A_SS[i], N, len);
  }

  cudaStreamSynchronize(stream);
}

template <>
void H2Factorize2<cuComplex>::compute(const cublasComputeType_t COMP) {
  long long N = bdim, S = rank, R = N - S;
  long long D = lenD;
  cuComplex one = make_cuComplex(1., 0.), zero = make_cuComplex(0., 0.), minus_one = make_cuComplex(-1., 0.);
  int info_host = 0;

  const auto ALGO = CUBLAS_GEMM_DEFAULT;

  cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(U), CUDA_C_32F, N, reinterpret_cast<void**>(A_SS), CUDA_C_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(B), CUDA_C_32F, N, D, COMP, ALGO);
  cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(U), CUDA_C_32F, N, reinterpret_cast<void**>(B), CUDA_C_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(A_SS), CUDA_C_32F, N, D, COMP, ALGO);

  cublasCgetrfBatched(cublasH, R, A_RR, N, ipiv, info, D);
  cublasCgetrsBatched(cublasH, CUBLAS_OP_N, R, S, A_RR, N, ipiv, A_RS, N, &info_host, D);
  cublasCgetrsBatched(cublasH, CUBLAS_OP_N, R, N, A_RR, N, ipiv, V_R, N, &info_host, D);

  cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, S, R, reinterpret_cast<void*>(&minus_one), reinterpret_cast<void**>(A_SR), CUDA_C_32F, N, reinterpret_cast<void**>(A_RS), CUDA_C_32F, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(A_SS), CUDA_C_32F, N, D, COMP, ALGO);

  for (int64_t i = D; i < lenA; i += maxQ) {
    int64_t len = std::min(lenA - i, maxQ);
    cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(&U[i]), CUDA_R_32F, N, reinterpret_cast<void**>(&A_SS[i]), CUDA_C_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(B), CUDA_C_32F, N, len, COMP, ALGO);
    cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(&V[i]), CUDA_R_32F, N, reinterpret_cast<void**>(B), CUDA_C_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(&A_SS[i]), CUDA_C_32F, N, len, COMP, ALGO);
  }

  cudaStreamSynchronize(stream);
}

template <>
void H2Factorize2<cuComplex>::compute() {
  compute(CUBLAS_COMPUTE_32F);
}

template <>
void H2Factorize2<double>::compute() {
  long long N = bdim, S = rank, R = N - S;
  long long D = lenD;
  double one = 1, zero = 0, minus_one = -1;
  int info_host = 0;

  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, U, N, A_SS, N, &zero, B, N, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, U, N, B, N, &zero, A_SS, N, D);

  cublasDgetrfBatched(cublasH, R, A_RR, N, ipiv, info, D);
  cublasDgetrsBatched(cublasH, CUBLAS_OP_N, R, S, A_RR, N, ipiv, A_RS, N, &info_host, D);
  cublasDgetrsBatched(cublasH, CUBLAS_OP_N, R, N, A_RR, N, ipiv, V_R, N, &info_host, D);

  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, S, R, &minus_one, A_SR, N, A_RS, N, &one, A_SS, N, D);

  for (int64_t i = D; i < lenA; i += maxQ) {
    int64_t len = std::min(lenA - i, maxQ);
    cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, &U[i], N, &A_SS[i], N, &zero, B, N, len);
    cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, &V[i], N, B, N, &zero, &A_SS[i], N, len);
  }

  cudaStreamSynchronize(stream);
}

template <>
void H2Factorize2<float>::compute(const cublasComputeType_t COMP) {
  long long N = bdim, S = rank, R = N - S;
  long long D = lenD;
  float one = 1, zero = 0, minus_one = -1;
  int info_host = 0;

  cublasLtMatmulDesc_t matmulDesc = NULL;
  checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t TRANS = CUBLAS_OP_T;
  const cublasOperation_t NO_TRANS = CUBLAS_OP_N;
  checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &NO_TRANS, sizeof(TRANS)));
  checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &TRANS, sizeof(TRANS)));
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, N, N, N));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, N, N, N));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, N, N, N));

  // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
  // directly come from cudaMalloc)
  cublasLtMatmulPreference_t preference = NULL;
  checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
  size_t workspaceSize = 1024 * 1024 * 4;
  void* workspace;
  cudaMalloc(&workspace, workspaceSize);
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
  // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
  // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
  int returnedResults                             = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(cublasLtH, matmulDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
  if (returnedResults == 0) {
    checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  for (long long i=0; i<D; ++i) {
    checkCublasStatus(cublasLtMatmul(cublasLtH, matmulDesc, &one, Uptr[i], Adesc, Aptr_SS[i], Bdesc, &zero, Bptr[i], Cdesc, Bptr[i], Cdesc,
                                     &heuristicResult.algo, workspace, workspaceSize, stream));
    //cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, Uptr[i], N, Aptr_SS[i], N, &zero, Bptr[i], N);
    checkCublasStatus(cublasLtMatmul(cublasLtH, matmulDesc, &one, Uptr[i], Adesc, Bptr[i], Cdesc, &zero, Aptr_SS[i], Bdesc, Aptr_SS[i], Bdesc,
                                     &heuristicResult.algo, workspace, workspaceSize, stream));
    //cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, Uptr[i], N, Bptr[i], N, &zero, Aptr_SS[i], N);
  }
  //cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, &U[0], N, &A_SS[0], N, &zero, &B[0], N, 1);
  //cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(U), CUDA_R_32F, N, reinterpret_cast<void**>(A_SS), CUDA_R_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(B), CUDA_R_32F, N, D, COMP, ALGO);
  //cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(U), CUDA_R_32F, N, reinterpret_cast<void**>(B), CUDA_R_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(A_SS), CUDA_R_32F, N, D, COMP, ALGO);

  cublasSgetrfBatched(cublasH, R, A_RR, N, ipiv, info, D);
  cublasSgetrsBatched(cublasH, CUBLAS_OP_N, R, S, A_RR, N, ipiv, A_RS, N, &info_host, D);
  cublasSgetrsBatched(cublasH, CUBLAS_OP_N, R, N, A_RR, N, ipiv, V_R, N, &info_host, D);

  for (long long i=0; i<D; ++i) {
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, S, R, &minus_one, Aptr_SR[i], N, Aptr_RS[i], N, &one, Aptr_SS[i], N);
  }
  //cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, S, R, reinterpret_cast<void*>(&minus_one), reinterpret_cast<void**>(A_SR), CUDA_R_32F, N, reinterpret_cast<void**>(A_RS), CUDA_R_32F, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(A_SS), CUDA_R_32F, N, D, COMP, ALGO);

  for (int64_t i = D; i < lenA; i += maxQ) {
    int64_t len = std::min(lenA - i, maxQ);
    for (long long j=0; j<len; ++j) {
      // TODO note that this loop does not deliver the exact same result as the batched version
      cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, Uptr[i+j], N, Aptr_SS[i+j], N, &zero, Bptr[j], N);
      cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, Vptr[i+j], N, Bptr[j], N, &zero, Aptr_SS[i+j], N);
    }
    //cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(&U[i]), CUDA_R_32F, N, reinterpret_cast<void**>(&A_SS[i]), CUDA_R_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(B), CUDA_R_32F, N, len, COMP, ALGO);
    //cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(&V[i]), CUDA_R_32F, N, reinterpret_cast<void**>(B), CUDA_R_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(&A_SS[i]), CUDA_R_32F, N, len, COMP, ALGO);
  }
  
  cudaStreamSynchronize(stream);
}

template <>
void H2Factorize2<float>::compute() {
  compute(CUBLAS_COMPUTE_32F);
}

template <typename DT> template <typename OT>
void H2Factorize2<DT>::getResults(long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], MatrixDataContainer<OT>& A, int* ipvts) {
  long long block = bdim * bdim;
  long long lenR = bdim - rank;

  DT* hostA;
  int* hostI;
  cudaMallocHost(reinterpret_cast<void**>(&hostA), block * lenA * sizeof(DT));
  cudaMallocHost(reinterpret_cast<void**>(&hostI), lenR * M * sizeof(int));
  cudaMemcpy(hostA, Adata, block * lenA * sizeof(DT), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostI, ipiv, lenR * M * sizeof(int), cudaMemcpyDeviceToHost);

  std::vector<long long> ipiv_offsets(M);
  std::exclusive_scan(&Dims[D], &Dims[D + M], ipiv_offsets.begin(), 0ll);

  for (long long i = 0; i < M; i++) {
    long long m = Dims[i + D];
    
    std::vector<int> rows(lenR);
    std::iota(rows.begin(), rows.end(), 0);
    for (long long j = lenR - 1; j >= 0; j--) {
      int p = hostI[i * lenR + j] - 1;
      if (p != j)
        std::iter_swap(rows.begin() + j, rows.begin() + p);
    }
    std::copy(rows.begin(), rows.begin() + m, &ipvts[ipiv_offsets[i]]);
    
    // used when not using eigen
    //std::copy(&hostI[i * lenR], &hostI[(i + 1) * lenR], &ipvts[ipiv_offsets[i]]);

    for (long long ij = ARows[i]; ij < ARows[i + 1]; ij++) {
      long long j = ACols[ij];
      long long n = Dims[j];
      //MKL_Zomatcopy('C', 'N', m, n, std::complex<double>(1., 0.), reinterpret_cast<std::complex<double>*>(&hostA[ij * block]), bdim, A[ij], m);
      omatcopy('C', 'N', m, n, &hostA[ij * block], bdim, A[ij], m);
    }
  }

  cudaFreeHost(hostA);
  cudaFreeHost(hostI);
}
*/
/* explicit template instantiation */
// complex double
/*template class H2Factorize2<cuDoubleComplex>;
template void H2Factorize2<cuDoubleComplex>::setData(long long, long long, long long, const long long[], const long long [], const long long[], const MatrixDataContainer<std::complex<double>>&, const MatrixDataContainer<std::complex<double>>&);
template void H2Factorize2<cuDoubleComplex>::getResults(long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], MatrixDataContainer<std::complex<double>>& A, int* ipvts);
// complex float
template class H2Factorize2<cuComplex>;
template void H2Factorize2<cuComplex>::setData(long long, long long, long long, const long long[], const long long [], const long long[], const MatrixDataContainer<std::complex<float>>&, const MatrixDataContainer<std::complex<float>>&);
template void H2Factorize2<cuComplex>::getResults(long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], MatrixDataContainer<std::complex<float>>& A, int* ipvts);
// double
template class H2Factorize2<double>;
template void H2Factorize2<double>::setData(long long, long long, long long, const long long[], const long long [], const long long[], const MatrixDataContainer<double>&, const MatrixDataContainer<double>&);
template void H2Factorize2<double>::getResults(long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], MatrixDataContainer<double>& A, int* ipvts);
// float
template class H2Factorize2<float>;
template void H2Factorize2<float>::setData(long long, long long, long long, const long long[], const long long [], const long long[], const MatrixDataContainer<float>&, const MatrixDataContainer<float>&);
template void H2Factorize2<float>::getResults(long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], MatrixDataContainer<float>& A, int* ipvts);
*/
