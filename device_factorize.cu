
#include <device_factorize.cuh>

#include <thrust/device_ptr.h>
#include <thrust/complex.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

/* explicit template instantiation */
// complex double
template void compute_factorize(deviceHandle_t handle, deviceMatrixDesc_t<std::complex<double>> A, deviceMatrixDesc_t<std::complex<double>> Al);
template void compute_forward_substitution(deviceHandle_t handle, deviceMatrixDesc_t<std::complex<double>> A, const std::complex<double>* X);
template void compute_backward_substitution(deviceHandle_t handle, deviceMatrixDesc_t<std::complex<double>> A, std::complex<double>* X);
template void matSolvePreconditionDeviceH2(deviceHandle_t handle, long long levels, deviceMatrixDesc_t<std::complex<double>> A[], std::complex<double>* devX);
// complex float
template void compute_factorize(deviceHandle_t handle, deviceMatrixDesc_t<std::complex<float>> A, deviceMatrixDesc_t<std::complex<float>> Al);
template void compute_forward_substitution(deviceHandle_t handle, deviceMatrixDesc_t<std::complex<float>> A, const std::complex<float>* X);
template void compute_backward_substitution(deviceHandle_t handle, deviceMatrixDesc_t<std::complex<float>> A, std::complex<float>* X);
template void matSolvePreconditionDeviceH2(deviceHandle_t handle, long long levels, deviceMatrixDesc_t<std::complex<float>> A[], std::complex<float>* devX);
// double
template void compute_factorize(deviceHandle_t handle, deviceMatrixDesc_t<double> A, deviceMatrixDesc_t<double> Al);
template void compute_forward_substitution(deviceHandle_t handle, deviceMatrixDesc_t<double> A, const double* X);
template void compute_backward_substitution(deviceHandle_t handle, deviceMatrixDesc_t<double> A, double* X);
template void matSolvePreconditionDeviceH2(deviceHandle_t handle, long long levels, deviceMatrixDesc_t<double> A[], double* devX);
// float
template void compute_factorize(deviceHandle_t handle, deviceMatrixDesc_t<float> A, deviceMatrixDesc_t<float> Al);
template void compute_forward_substitution(deviceHandle_t handle, deviceMatrixDesc_t<float> A, const float* X);
template void compute_backward_substitution(deviceHandle_t handle, deviceMatrixDesc_t<float> A, float* X);
template void matSolvePreconditionDeviceH2(deviceHandle_t handle, long long levels, deviceMatrixDesc_t<float> A[], float* devX);

struct swapXY {
  long long M, B;
  swapXY(long long M, long long N) : M(M), B(M * N) {}
  __host__ __device__ long long operator()(long long i) const {
    long long x = i / B; long long y = i - x * B;
    long long z = y / M; long long w = y - z * M;
    return x * B + z + w * M;
  }
};

struct conjugateFunc {
  __host__ __device__ thrust::complex<float> operator()(const thrust::complex<float>& z) const {
    return thrust::conj(z);
  }
  __host__ __device__ thrust::complex<double> operator()(const thrust::complex<double>& z) const {
    return thrust::conj(z);
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

template <typename DT>
inline void conjugate_transpose(cudaStream_t stream, const long long bdim, const long long block, const long long D, const long long M, DT* Udata, DT* Vdata) {
  auto inc_iter = thrust::make_counting_iterator(0ll);
  auto mapV = thrust::make_transform_iterator(inc_iter, swapXY(bdim, bdim));
  thrust::device_ptr<DT> Uptr(&(Udata)[D * block]);
  thrust::device_ptr<DT> Vptr(Vdata);
  thrust::gather(thrust::cuda::par.on(stream), mapV, mapV + (block * M), Uptr, Vptr);
}

template <>
inline void conjugate_transpose(cudaStream_t stream, const long long bdim, const long long block, const long long D, const long long M, cuDoubleComplex* Udata, cuDoubleComplex* Vdata) {
  auto inc_iter = thrust::make_counting_iterator(0ll);
  auto mapV = thrust::make_transform_iterator(inc_iter, swapXY(bdim, bdim));
  thrust::device_ptr<thrust::complex<double>> Uptr(reinterpret_cast<thrust::complex<double>*>(&(Udata)[D * block]));
  thrust::device_ptr<thrust::complex<double>> Vptr(reinterpret_cast<thrust::complex<double>*>(Vdata));
  thrust::gather(thrust::cuda::par.on(stream), mapV, mapV + (block * M), thrust::make_transform_iterator(Uptr, conjugateFunc()), Vptr);
}

template <>
inline void conjugate_transpose(cudaStream_t stream, const long long bdim, const long long block, const long long D, const long long M, cuComplex* Udata, cuComplex* Vdata) {
  auto inc_iter = thrust::make_counting_iterator(0ll);
  auto mapV = thrust::make_transform_iterator(inc_iter, swapXY(bdim, bdim));
  thrust::device_ptr<thrust::complex<float>> Uptr(reinterpret_cast<thrust::complex<float>*>(&(Udata)[D * block]));
  thrust::device_ptr<thrust::complex<float>> Vptr(reinterpret_cast<thrust::complex<float>*>(Vdata));
  thrust::gather(thrust::cuda::par.on(stream), mapV, mapV + (block * M), thrust::make_transform_iterator(Uptr, conjugateFunc()), Vptr);
}

template <typename DT>
inline void transform(cudaStream_t stream, const long long rank, const long long bdim, const long long rblock, const long long M, DT* Bdata, DT** A_ss) {
  auto inc_iter = thrust::make_counting_iterator(0ll);
  thrust::device_ptr<DT> ACptr(reinterpret_cast<DT*>(Bdata));
  auto Aiter = thrust::make_transform_iterator(inc_iter, StridedBlock(rank, rank, bdim, reinterpret_cast<DT**>(A_ss)));
  thrust::transform(thrust::cuda::par.on(stream), Aiter, Aiter + (rblock * M), ACptr, Aiter, thrust::plus<DT>());
}

template <>
inline void transform(cudaStream_t stream, const long long rank, const long long bdim, const long long rblock, const long long M, cuDoubleComplex* Bdata, cuDoubleComplex** A_ss) {
  auto inc_iter = thrust::make_counting_iterator(0ll);
  thrust::device_ptr<thrust::complex<double>> ACptr(reinterpret_cast<thrust::complex<double>*>(Bdata));
  auto Aiter = thrust::make_transform_iterator(inc_iter, StridedBlock(rank, rank, bdim, reinterpret_cast<thrust::complex<double>**>(A_ss)));
  thrust::transform(thrust::cuda::par.on(stream), Aiter, Aiter + (rblock * M), ACptr, Aiter, thrust::plus<thrust::complex<double>>());
}

template <>
inline void transform(cudaStream_t stream, const long long rank, const long long bdim, const long long rblock, const long long M, cuComplex* Bdata, cuComplex** A_ss) {
  auto inc_iter = thrust::make_counting_iterator(0ll);
  thrust::device_ptr<thrust::complex<float>> ACptr(reinterpret_cast<thrust::complex<float>*>(Bdata));
  auto Aiter = thrust::make_transform_iterator(inc_iter, StridedBlock(rank, rank, bdim, reinterpret_cast<thrust::complex<float>**>(A_ss)));
  thrust::transform(thrust::cuda::par.on(stream), Aiter, Aiter + (rblock * M), ACptr, Aiter, thrust::plus<thrust::complex<float>>());
}

inline void cublasXgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha,  const cuDoubleComplex *const Aarray[], int lda,
  const cuDoubleComplex *const Barray[], int ldb, const cuDoubleComplex *beta, cuDoubleComplex *const Carray[], int ldc, int batchCount) {
    cublasZgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

inline void cublasXgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha,  const cuComplex *const Aarray[], int lda,
  const cuComplex *const Barray[], int ldb, const cuComplex *beta, cuComplex *const Carray[], int ldc, int batchCount) {
    cublasCgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

inline void cublasXgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha,  const double *const Aarray[], int lda,
  const double *const Barray[], int ldb, const double *beta, double *const Carray[], int ldc, int batchCount) {
    cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

inline void cublasXgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha,  const float *const Aarray[], int lda,
  const float *const Barray[], int ldb, const float *beta, float *const Carray[], int ldc, int batchCount) {
    cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

inline void cublasXgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize) {
  cublasZgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
}

inline void cublasXgetrfBatched(cublasHandle_t handle, int n, cuComplex *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize) {
  cublasCgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
}

inline void cublasXgetrfBatched(cublasHandle_t handle, int n, double *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize) {
  cublasDgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
}

inline void cublasXgetrfBatched(cublasHandle_t handle, int n, float *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize) {
  cublasSgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
}

inline void cublasXgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex *const Aarray[], int lda,
  const int *devIpiv, cuDoubleComplex *const Barray[], int ldb, int *info, int batchSize) {
    cublasZgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
}

inline void cublasXgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex *const Aarray[], int lda,
  const int *devIpiv, cuComplex *const Barray[], int ldb, int *info, int batchSize) {
    cublasCgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
}

inline void cublasXgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *const Aarray[], int lda,
  const int *devIpiv, double *const Barray[], int ldb, int *info, int batchSize) {
    cublasDgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
}

inline void cublasXgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float *const Aarray[], int lda,
  const int *devIpiv, float *const Barray[], int ldb, int *info, int batchSize) {
    cublasSgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
}

inline void cublasXgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
  long long int strideA, const cuDoubleComplex *x, int incx, long long int stridex, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy, long long int stridey, int batchCount) {
    cublasZgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
}

inline void cublasXgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda,
  long long int strideA, const cuComplex *x, int incx, long long int stridex, const cuComplex *beta, cuComplex *y, int incy, long long int stridey, int batchCount) {
    cublasCgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
}

inline void cublasXgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda,
  long long int strideA, const double *x, int incx, long long int stridex, const double *beta, double *y, int incy, long long int stridey, int batchCount) {
    cublasDgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
}

inline void cublasXgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda,
  long long int strideA, const float *x, int incx, long long int stridex, const float *beta, float *y, int incy, long long int stridey, int batchCount) {
    cublasSgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount);
}

template <typename DT>
void compute_factorize(deviceHandle_t handle, deviceMatrixDesc_t<DT> A, deviceMatrixDesc_t<DT> Al) {
  typedef typename deviceMatrixDesc_t<DT>::CT CT;

  long long bdim = A.bdim;
  long long rank = A.rank;
  long long block = bdim * bdim;
  long long rblock = rank * rank;

  long long D = A.diag_offset;
  long long M = A.lenM;
  long long N = A.lenN;
  long long lenA = A.lenA;
  long long lenL = Al.lenA;

  cudaStream_t stream = handle->compute_stream;
  cublasHandle_t cublasH = handle->cublasH;

  long long rdim = bdim - rank;
  long long reduc_len = A.reducLen;
  int info_host = 0;
  DT constants[3] = { 1., 0., -1. };
  CT& one = reinterpret_cast<CT&>(constants[0]);
  CT& zero = reinterpret_cast<CT&>(constants[1]); 
  CT& minus_one = reinterpret_cast<CT&>(constants[2]); 

  auto inc_iter = thrust::make_counting_iterator(0ll);
  conjugate_transpose(stream, bdim, block, D, M, A.Udata, A.Vdata);
  //auto mapV = thrust::make_transform_iterator(inc_iter, swapXY(bdim, bdim));
  //thrust::device_ptr<THRUST_CTYPE> Uptr(reinterpret_cast<THRUST_CTYPE*>(&(A.Udata)[D * block]));
  //thrust::device_ptr<THRUST_CTYPE> Vptr(reinterpret_cast<THRUST_CTYPE*>(A.Vdata));
  //thrust::gather(thrust::cuda::par.on(stream), mapV, mapV + (block * M), thrust::make_transform_iterator(Uptr, conjugateFunc()), Vptr);
  
  if (0 < lenL) {
    long long len = Al.rank * Al.rank * lenL;
    thrust::for_each(thrust::cuda::par.on(stream), inc_iter, inc_iter + len, copyFunc(Al.rank, Al.rank, Al.A_unsort, Al.bdim, A.A_dst, bdim));
  }

  if (M == 1) {
    if (A.MergeComm)
      ncclAllReduce(const_cast<const CT*>(A.Adata), A.Adata, block * lenA * 2, ncclDouble, ncclSum, A.MergeComm, stream);
    if (A.DupComm)
      ncclBroadcast(const_cast<const CT*>(A.Adata), A.Adata, block * lenA * 2, ncclDouble, 0, A.DupComm, stream);
  }

  cublasXgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, A.V_rows, bdim, A.A_ss, bdim, &zero, A.B_ind, bdim, M);
  cublasXgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, A.V_rows, bdim, A.B_ind, bdim, &zero, A.A_ss, bdim, M);

  cublasXgetrfBatched(cublasH, rdim, A.A_rr, bdim, A.Ipiv, A.Info, M);
  cublasXgetrsBatched(cublasH, CUBLAS_OP_N, rdim, bdim, A.A_rr, bdim, A.Ipiv, A.V_R, bdim, &info_host, M);

  if (0 < rank) {
    cublasXgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, bdim, &one, A.V_R, bdim, A.B_ind, bdim, &zero, A.A_rs, bdim, M);
    cudaMemsetAsync(A.ACdata, 0, reduc_len * M * rblock * sizeof(CT), stream);

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      cublasXgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, &(A.U_cols)[i], bdim, &(A.A_ss)[i], bdim, &zero, A.B_ind, bdim, len);
      cublasXgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, &(A.V_rows)[i], bdim, A.B_ind, bdim, &zero, &(A.A_ss)[i], bdim, len);
    }

    cublasXgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, A.A_sr_rows, bdim, A.A_rs, bdim, &one, A.A_ss, bdim, lenA);
    cublasXgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rdim, rdim, bdim, &one, A.V_R, bdim, A.U_R, bdim, &zero, A.B_R, bdim, M);
    thrust::for_each(thrust::cuda::par.on(stream), inc_iter, inc_iter + (rdim * rank * M), copyFunc(rdim, rank, const_cast<const CT**>(A.A_rs), bdim, &(A.B_ind)[D], bdim));
    
    ncclGroupStart();
    for (long long p = 0; p < A.LenComms; p++) {
      long long start = A.Neighbor[p] * block;
      long long len = A.Neighbor[p + 1] * block - start;
      ncclBroadcast(const_cast<const CT*>(&(A.Bdata)[start]), &(A.Bdata)[start], len * 2, ncclDouble, A.NeighborRoots[p], A.NeighborComms[p], stream);
    }

    if (A.DupComm)
      ncclBroadcast(const_cast<const CT*>(A.Bdata), A.Bdata, block * N * 2, ncclDouble, 0, A.DupComm, stream);
    ncclGroupEnd();

    if (M < lenA)
      cublasXgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &(A.A_sr)[M], bdim, &(A.B_cols)[M], bdim, &one, &(A.A_ss)[M], bdim, lenA - M);

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      cublasXgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, rdim, &one, &(A.B_R)[i], bdim, &(A.A_sr)[i], bdim, &zero, A.B_ind, bdim, len);
      cublasXgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &(A.A_sr)[i], bdim, A.B_ind, bdim, &zero, &A.AC_ind[i], rank, len);
    }
    cublasXgemvStridedBatched(cublasH, CUBLAS_OP_N, M * rank, reduc_len, &one, A.ACdata, M * rblock, M * rank, A.ONEdata, 1, 0, &zero, A.Bdata, 1, M * rank, rank);

    transform(stream, rank, bdim, rblock, M, A.Bdata, A.A_ss);
    //thrust::device_ptr<THRUST_CTYPE> ACptr(reinterpret_cast<THRUST_CTYPE*>(A.Bdata));
    //auto Aiter = thrust::make_transform_iterator(inc_iter, StridedBlock(rank, rank, bdim, reinterpret_cast<THRUST_CTYPE**>(A.A_ss)));
    //thrust::transform(thrust::cuda::par.on(stream), Aiter, Aiter + (rblock * M), ACptr, Aiter, thrust::plus<THRUST_CTYPE>());
  }
}

inline void cublasXgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
  long long int strideA, const cuDoubleComplex *B, int ldb, long long int strideB, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc, long long int strideC, int batchCount) {
    cublasZgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

inline void cublasXgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda,
  long long int strideA, const cuComplex *B, int ldb, long long int strideB, const cuComplex *beta, cuComplex *C, int ldc, long long int strideC, int batchCount) {
    cublasCgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

inline void cublasXgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda,
  long long int strideA, const double *B, int ldb, long long int strideB, const double *beta, double *C, int ldc, long long int strideC, int batchCount) {
    cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

inline void cublasXgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda,
  long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount) {
    cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

inline void cublasXgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
  const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
  cublasZgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline void cublasXgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda,
  const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
  cublasCgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline void cublasXgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda,
  const double *x, int incx, const double *beta, double *y, int incy) {
  cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline void cublasXgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda,
  const float *x, int incx, const float *beta, float *y, int incy) {
  cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <typename DT>
void compute_forward_substitution(deviceHandle_t handle, deviceMatrixDesc_t<DT> A, const DT* X) {
  typedef typename deviceMatrixDesc_t<DT>::CT CT;

  long long bdim = A.bdim;
  long long rank = A.rank;
  long long rdim = bdim - rank;
  long long block = bdim * bdim;

  long long D = A.diag_offset;
  long long M = A.lenM;
  long long N = A.lenN;
  long long lenA = A.lenA;
  long long reduc_len = A.reducLen;

  cudaStream_t stream = handle->compute_stream;
  cublasHandle_t cublasH = handle->cublasH;

  DT constants[3] = { 1., 0., -1. };
  CT& one = reinterpret_cast<CT&>(constants[0]);
  CT& zero = reinterpret_cast<CT&>(constants[1]);
  CT& minus_one = reinterpret_cast<CT&>(constants[2]);
  const CT* X_in = reinterpret_cast<const CT*>(&X[A.lower_offset]);

  cublasXgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, bdim, 1, bdim, &one, A.Vdata, bdim, block, X_in, bdim, bdim, &zero, &(A.Ydata)[D * bdim], bdim, bdim, M);

  if (1 < N) {
    ncclGroupStart();
    for (long long p = 0; p < A.LenComms; p++) {
      long long start = A.Neighbor[p] * bdim;
      long long len = A.Neighbor[p + 1] * bdim - start;
      ncclBroadcast(const_cast<const CT*>(&(A.Ydata)[start]), &(A.Ydata)[start], len * 2, ncclDouble, A.NeighborRoots[p], A.NeighborComms[p], stream);
    }
    if (A.DupComm)
      ncclBroadcast(const_cast<const CT*>(A.Ydata), A.Ydata, bdim * N * 2, ncclDouble, 0, A.DupComm, stream);
    ncclGroupEnd();
  }

  size_t sizeX = rank * sizeof(CT);
  cudaMemcpy2DAsync(&(A.Xdata)[D * rank], sizeX, &(A.Ydata)[D * bdim], bdim * sizeof(CT), sizeX, M, cudaMemcpyDeviceToDevice, stream);
  if (0 < rank && 0 < rdim) {
    cudaMemsetAsync(A.ACdata, 0, reduc_len * M * rank * sizeof(CT), stream);
    cublasXgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, 1, rdim, &minus_one, A.A_sr, bdim, A.Y_R_cols, bdim, &zero, A.AC_X, rank, lenA);
    cublasXgemv(cublasH, CUBLAS_OP_N, M * rank, reduc_len, &one, A.ACdata, M * rank, A.ONEdata, 1, &one, &(A.Xdata)[D * rank], 1);
  }

  if (1 < N) {
    ncclGroupStart();
    for (long long p = 0; p < A.LenComms; p++) {
      long long start = A.Neighbor[p] * rank;
      long long len = A.Neighbor[p + 1] * rank - start;
      ncclBroadcast(const_cast<const CT*>(&(A.Xdata)[start]), &(A.Xdata)[start], len * 2, ncclDouble, A.NeighborRoots[p], A.NeighborComms[p], stream);
    }
    if (A.DupComm)
      ncclBroadcast(const_cast<const CT*>(A.Xdata), A.Xdata, rank * N * 2, ncclDouble, 0, A.DupComm, stream);
    ncclGroupEnd();
  }
}

template <typename DT>
void compute_backward_substitution(deviceHandle_t handle, deviceMatrixDesc_t<DT> A, DT* X) {
  typedef typename deviceMatrixDesc_t<DT>::CT CT;

  long long bdim = A.bdim;
  long long rank = A.rank;
  long long rdim = bdim - rank;
  long long block = bdim * bdim;

  long long D = A.diag_offset;
  long long M = A.lenM;
  long long N = A.lenN;
  long long lenA = A.lenA;
  long long reduc_len = A.reducLen;

  cudaStream_t stream = handle->compute_stream;
  cublasHandle_t cublasH = handle->cublasH;

  DT constants[3] = { 1., 0., -1. };
  CT& one = reinterpret_cast<CT&>(constants[0]);
  CT& zero = reinterpret_cast<CT&>(constants[1]);
  CT& minus_one = reinterpret_cast<CT&>(constants[2]);
  CT* X_out = reinterpret_cast<CT*>(&X[A.lower_offset]);

  if (1 < N) {
    ncclGroupStart();
    for (long long p = 0; p < A.LenComms; p++) {
      long long start = A.Neighbor[p] * rank;
      long long len = A.Neighbor[p + 1] * rank - start;
      ncclBroadcast(const_cast<const CT*>(&(A.Xdata)[start]), &(A.Xdata)[start], len * 2, ncclDouble, A.NeighborRoots[p], A.NeighborComms[p], stream);
    }
    if (A.DupComm)
      ncclBroadcast(const_cast<const CT*>(A.Xdata), A.Xdata, rank * N * 2, ncclDouble, 0, A.DupComm, stream);
    ncclGroupEnd();
  }

  size_t sizeX = rank * sizeof(CT);
  cudaMemcpy2DAsync(&(A.Ydata)[D * bdim], bdim * sizeof(CT), &(A.Xdata)[D * rank], sizeX, sizeX, M, cudaMemcpyDeviceToDevice, stream);
  if (0 < rank && 0 < rdim) {
    cudaMemsetAsync(A.ACdata, 0, reduc_len * M * bdim * sizeof(CT), stream);
    cublasXgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rdim, 1, rank, &minus_one, A.A_rs, bdim, A.X_cols, bdim, &zero, A.AC_X_R, bdim, lenA);
    cublasXgemv(cublasH, CUBLAS_OP_N, M * bdim, reduc_len, &one, A.ACdata, M * bdim, A.ONEdata, 1, &one, &(A.Ydata)[D * bdim], 1);
  }

  cublasXgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_C, 1, bdim, bdim, &one, &(A.Ydata)[D * bdim], 1, bdim, &(A.Udata)[D * block], bdim, block, &zero, X_out, 1, bdim, M);
}

template <typename DT>
void matSolvePreconditionDeviceH2(deviceHandle_t handle, long long levels, deviceMatrixDesc_t<DT> desc[], DT* devX) {
  if (0 <= levels) {
    compute_forward_substitution(handle, desc[levels], devX);
    for (long long l = levels - 1; l >= 0; l--)
      compute_forward_substitution(handle, desc[l], reinterpret_cast<DT*>(desc[l + 1].Xdata));

    for (long long l = 0; l < levels; l++)
      compute_backward_substitution(handle, desc[l], reinterpret_cast<DT*>(desc[l + 1].Xdata));
    compute_backward_substitution(handle, desc[levels], devX);
  }
}
