
#include <factorize.cuh>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <mkl.h>
#include <cstring>

#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/complex.h>
#include <thrust/sequence.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>

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

template <typename DT>
H2Factorize<DT>::H2Factorize(long long LD, long long lenA, long long lenQ, cudaStream_t stream) : maxA(lenA), maxQ(lenQ), stream(stream) {
  cublasH = nullptr;
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);

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

  long long len = std::max(lenA, lenQ);
  cudaMallocHost(reinterpret_cast<void**>(&hostA), bsize * len);
  cudaMallocHost(reinterpret_cast<void**>(&hostP), psize * len);

  for (long long i = 0; i < lenQ; i++)
    hostP[i] = &Bdata[i * LD * LD];

  cudaMemcpy(B, hostP, psize * lenQ, cudaMemcpyHostToDevice);
}

template <typename DT>
H2Factorize<DT>::~H2Factorize() {
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
  
  cudaFreeHost(hostA);
  cudaFreeHost(hostP);
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
  long long block;
  setDevicePtr(T* data, long long block) : data(data), block(block) {}
  __host__ __device__ T* operator()(long long i) const {
    return data + i * block;
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
template <>
void H2Factorize<cuDoubleComplex>::factorize(const long long bdim, const long long rank, const long long block, const long long M, const long long D) {
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
void H2Factorize<cuComplex>::factorize(const long long bdim, const long long rank, const long long block, const long long M, const long long D) {
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
void H2Factorize<double>::factorize(const long long bdim, const long long rank, const long long block, const long long M, const long long D) {
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
}

template <>
void H2Factorize<float>::factorize(const long long bdim, const long long rank, const long long block, const long long M, const long long D) {
 long long rdim = bdim - rank;
  int info_host = 0;
  float one = 1., zero = 0., minus_one = -1.;

  thrust::device_ptr<const float> u_ptr = thrust::device_ptr<const float>(reinterpret_cast<const float*>(&Udata[D * block]));
  thrust::device_ptr<float> v_ptr = thrust::device_ptr<float>(reinterpret_cast<float*>(Vdata));

  auto map = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), swapXY(bdim, block));
  thrust::gather(thrust::cuda::par.on(stream), map, map + block * M, u_ptr, v_ptr);

  cublasSgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, A_SS, bdim, &zero, B, bdim, M);
  cublasSgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, U, bdim, B, bdim, &zero, A_SS, bdim, M);

  cublasSgetrfBatched(cublasH, rdim, A_RR, bdim, ipiv, info, M);
  cublasSgetrsBatched(cublasH, CUBLAS_OP_N, rdim, rank, A_RR, bdim, ipiv, A_RS, bdim, &info_host, M);
  cublasSgetrsBatched(cublasH, CUBLAS_OP_N, rdim, bdim, A_RR, bdim, ipiv, V_R, bdim, &info_host, M);

  cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, A_SR, bdim, A_RS, bdim, &one, A_SS, bdim, M);

  for (int64_t i = M; i < lenA; i += maxQ) {
    int64_t len = std::min(lenA - i, maxQ);
    cublasSgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, &U[i], bdim, &A_SS[i], bdim, &zero, B, bdim, len);
    cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, &V[i], bdim, B, bdim, &zero, &A_SS[i], bdim, len);
  }
}

template <typename DT> template <typename OT>
void H2Factorize<DT>::compute(const long long bdim, const long long rank, const long long D, const long long M, const long long N, const long long ARows[], const long long ACols[], OT* const A, OT* const R, const OT* const Q) {
  long long block = bdim * bdim;
  long long lenA = ARows[M];

  std::copy(Q, Q + block * N, reinterpret_cast<OT*>(hostA));
  cudaMemcpy(Udata, hostA, block * N * sizeof(DT), cudaMemcpyHostToDevice);

  std::copy(A, A + block * lenA, reinterpret_cast<OT*>(hostA));
  cudaMemcpy(Adata, hostA, block * lenA * sizeof(DT), cudaMemcpyHostToDevice);

  thrust::device_vector<long long> row_offsets(ARows, ARows + M);
  thrust::device_vector<long long> rows(lenA, 0ll);
  thrust::device_vector<long long> cols(ACols, ACols + lenA);

  auto one_iter = thrust::make_constant_iterator(1ll);
  thrust::scatter(one_iter, one_iter + (M - 1), row_offsets.begin() + 1, rows.begin());
  thrust::inclusive_scan(rows.begin(), rows.end(), rows.begin());

  thrust::device_vector<long long> keys(lenA);
  thrust::device_vector<long long> indices(lenA);

  thrust::transform(rows.begin(), rows.end(), cols.begin(), keys.begin(), keysDLU(D, M, N));
  thrust::sequence(indices.begin(), indices.end(), 0);
  thrust::sort_by_key(keys.begin(), keys.end(), thrust::make_zip_iterator(rows.begin(), cols.begin(), indices.begin()));

  thrust::device_vector<DT*> a_ss(lenA), a_sr(lenA), a_rs(lenA), a_rr(lenA);
  thrust::device_vector<DT*> u(lenA), v(lenA), v_r(lenA), b(maxQ);

  long long offset_SR = rank * bdim, offset_RS = rank, offset_RR = rank * (bdim + 1);
  auto inc_iter = thrust::make_counting_iterator(0ll);
  thrust::transform(indices.begin(), indices.end(), a_ss.begin(), setDevicePtr(Adata, block));
  thrust::transform(indices.begin(), indices.end(), a_sr.begin(), setDevicePtr(Adata + offset_SR, block));
  thrust::transform(indices.begin(), indices.end(), a_rs.begin(), setDevicePtr(Adata + offset_RS, block));
  thrust::transform(indices.begin(), indices.end(), a_rr.begin(), setDevicePtr(Adata + offset_RR, block));
  thrust::transform(cols.begin(), cols.end(), u.begin(), setDevicePtr(Udata, block));
  thrust::transform(rows.begin(), rows.end(), v.begin(), setDevicePtr(Vdata, block));
  thrust::transform(rows.begin(), rows.end(), v_r.begin(), setDevicePtr(Vdata + offset_RS, block));
  thrust::transform(inc_iter, inc_iter + maxQ, b.begin(), setDevicePtr(Bdata, block));

  DT** A_SS = thrust::raw_pointer_cast(a_ss.data());
  DT** A_SR = thrust::raw_pointer_cast(a_sr.data());
  DT** A_RS = thrust::raw_pointer_cast(a_rs.data());
  DT** A_RR = thrust::raw_pointer_cast(a_rr.data());
  DT** U = thrust::raw_pointer_cast(u.data());
  DT** V = thrust::raw_pointer_cast(v.data());
  DT** V_R = thrust::raw_pointer_cast(v_r.data());
  DT** B = thrust::raw_pointer_cast(b.data());

  factorize(bdim, rank, block, M, D);
  cudaStreamSynchronize(stream);

  cudaMemcpy(hostA, Adata, block * lenA * sizeof(DT), cudaMemcpyDeviceToHost);
  std::copy(reinterpret_cast<OT*>(hostA), reinterpret_cast<OT*>(&hostA[block * lenA]), A);

  cudaMemcpy(hostA, Vdata, block * M * sizeof(DT), cudaMemcpyDeviceToHost);
  std::copy(reinterpret_cast<OT*>(hostA), reinterpret_cast<OT*>(&hostA[block * M]), R + block * D);
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
template class H2Factorize<cuDoubleComplex>;
template void H2Factorize<cuDoubleComplex>::compute(const long long, const long long, const long long, const long long, const long long, const long long[], const long long[], std::complex<double>* const A, std::complex<double>* const R, const std::complex<double>* const Q);
// complex float
template class H2Factorize<cuComplex>;
template void H2Factorize<cuComplex>::compute(const long long, const long long, const long long, const long long, const long long, const long long[], const long long[], std::complex<float>* const A, std::complex<float>* const R, const std::complex<float>* const Q);
// double
template class H2Factorize<double>;
template void H2Factorize<double>::compute(const long long, const long long, const long long, const long long, const long long, const long long[], const long long[], double* const A, double* const R, const double* const Q);
// float
template class H2Factorize<float>;
template void H2Factorize<float>::compute(const long long, const long long, const long long, const long long, const long long, const long long[], const long long[], float* const A, float* const R, const float* const Q);

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
