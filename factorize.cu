
#include <factorize.cuh>
#include <comm-mpi.hpp>
#include <algorithm>
#include <numeric>
#include <tuple>

#include <cuComplex.h>
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

void compute_factorize(cublasHandle_t cublasH, long long bdim, long long rank, std::complex<double>* A, std::complex<double>* R, const std::complex<double>* Q, const ColCommMPI& comm) {
  long long block = bdim * bdim;
  long long D = comm.oLocal();
  long long M = comm.lenLocal();
  long long N = comm.lenNeighbors();
  
  const long long* ARows = comm.ARowOffsets.data();
  const long long* ACols = comm.AColumns.data();
  long long lenA = ARows[M];
  
  cudaStream_t stream;
  cublasGetStream(cublasH, &stream);

  thrust::device_vector<long long> ARowOffset_vec(ARows, ARows + M);
  thrust::device_vector<long long> ARows_vec(lenA, 0ll);
  thrust::device_vector<long long> ACols_vec(ACols, ACols + lenA);
  thrust::device_vector<long long> ADistCols_vec(lenA);
  thrust::device_vector<long long> AInd_vec(lenA);
  thrust::device_vector<long long> keys(lenA);
  thrust::device_vector<thrust::complex<double>*> A_ss_vec(lenA), A_sr_vec(lenA), A_rs_vec(lenA), A_rr_vec(lenA);
  thrust::device_vector<thrust::complex<double>*> U_cols_vec(lenA), V_rows_vec(lenA), U_R_vec(M), V_R_vec(M), B_ind_vec(N), B_cols_vec(lenA), B_R_vec(lenA);
  thrust::device_vector<thrust::complex<double>*> A_sr_rows_vec(lenA), AC_ind_vec(lenA);

  thrust::device_vector<thrust::complex<double>> Avec(lenA * block);
  thrust::device_vector<thrust::complex<double>> Bvec(N * block);
  thrust::device_vector<thrust::complex<double>> Uvec(N * block);
  thrust::device_vector<thrust::complex<double>> Vvec(M * block);
  
  thrust::device_vector<int> Ipiv(M * bdim);
  thrust::device_vector<int> Info(M);
  std::vector<long long> Bsizes(N, block);
  comm.dataSizesToNeighborOffsets(Bsizes.data());

  auto inc_iter = thrust::make_counting_iterator(0ll);
  auto one_iter = thrust::make_constant_iterator(1ll);
  auto rwise_diag_iter = thrust::make_permutation_iterator(AInd_vec.begin(), ARows_vec.begin());

  thrust::scatter(one_iter, one_iter + (M - 1), ARowOffset_vec.begin() + 1, ARows_vec.begin()); 
  thrust::inclusive_scan(ARows_vec.begin(), ARows_vec.end(), ARows_vec.begin());
  thrust::exclusive_scan_by_key(ARows_vec.begin(), ARows_vec.end(), one_iter, ADistCols_vec.begin(), 0ll);

  thrust::transform(ARows_vec.begin(), ARows_vec.end(), ACols_vec.begin(), keys.begin(), keysDLU(D, M, N));
  thrust::sequence(AInd_vec.begin(), AInd_vec.end(), 0);
  thrust::sort_by_key(keys.begin(), keys.end(), thrust::make_zip_iterator(ARows_vec.begin(), ACols_vec.begin(), ADistCols_vec.begin(), AInd_vec.begin()));

  long long reduc_len = 1ll + thrust::reduce(ADistCols_vec.begin(), ADistCols_vec.end(), 0ll, thrust::maximum<long long>());
  thrust::device_vector<thrust::complex<double>> ACvec(reduc_len * M * rank * rank, thrust::complex<double>(0., 0.));

  long long offset_SR = rank * bdim, offset_RS = rank, offset_RR = rank * (bdim + 1);
  thrust::transform(AInd_vec.begin(), AInd_vec.end(), A_ss_vec.begin(), setDevicePtr(thrust::raw_pointer_cast(Avec.data()), block));
  thrust::transform(AInd_vec.begin(), AInd_vec.end(), A_sr_vec.begin(), setDevicePtr(thrust::raw_pointer_cast(Avec.data()) + offset_SR, block));
  thrust::transform(AInd_vec.begin(), AInd_vec.end(), A_rs_vec.begin(), setDevicePtr(thrust::raw_pointer_cast(Avec.data()) + offset_RS, block));
  thrust::transform(AInd_vec.begin(), AInd_vec.end(), A_rr_vec.begin(), setDevicePtr(thrust::raw_pointer_cast(Avec.data()) + offset_RR, block));
  thrust::transform(ACols_vec.begin(), ACols_vec.end(), U_cols_vec.begin(), setDevicePtr(thrust::raw_pointer_cast(Uvec.data()), block));
  thrust::transform(ACols_vec.begin(), ACols_vec.begin() + M, U_R_vec.begin(), setDevicePtr(thrust::raw_pointer_cast(Uvec.data()) + offset_SR, block));
  thrust::transform(ARows_vec.begin(), ARows_vec.end(), V_rows_vec.begin(), setDevicePtr(thrust::raw_pointer_cast(Vvec.data()), block));
  thrust::transform(inc_iter, inc_iter + M, V_R_vec.begin(), setDevicePtr(thrust::raw_pointer_cast(Vvec.data()) + offset_RS, block));

  thrust::transform(inc_iter, inc_iter + N, B_ind_vec.begin(), setDevicePtr(thrust::raw_pointer_cast(Bvec.data()), block));
  thrust::transform(ACols_vec.begin(), ACols_vec.end(), B_cols_vec.begin(), setDevicePtr(thrust::raw_pointer_cast(Bvec.data()), block));
  thrust::transform(ACols_vec.begin(), ACols_vec.end(), B_R_vec.begin(), setDevicePtr(thrust::raw_pointer_cast(Bvec.data()) + offset_SR, block));
  thrust::transform(rwise_diag_iter, rwise_diag_iter + lenA, A_sr_rows_vec.begin(), setDevicePtr(thrust::raw_pointer_cast(Avec.data()) + offset_SR, block));
  thrust::transform(ARows_vec.begin(), ARows_vec.end(), ADistCols_vec.begin(), AC_ind_vec.begin(), setDevicePtr(thrust::raw_pointer_cast(ACvec.data()), rank * rank, M));

  cuDoubleComplex* Adata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(Avec.data()));
  cuDoubleComplex* Udata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(Uvec.data()));
  cuDoubleComplex* Vdata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(Vvec.data()));
  cuDoubleComplex* Bdata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(Bvec.data()));
  cuDoubleComplex* ACdata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(ACvec.data()));

  cuDoubleComplex** A_SS = reinterpret_cast<cuDoubleComplex**>(thrust::raw_pointer_cast(A_ss_vec.data()));
  cuDoubleComplex** A_SR = reinterpret_cast<cuDoubleComplex**>(thrust::raw_pointer_cast(A_sr_vec.data()));
  cuDoubleComplex** A_RS = reinterpret_cast<cuDoubleComplex**>(thrust::raw_pointer_cast(A_rs_vec.data()));
  cuDoubleComplex** A_RR = reinterpret_cast<cuDoubleComplex**>(thrust::raw_pointer_cast(A_rr_vec.data()));
  cuDoubleComplex** U = reinterpret_cast<cuDoubleComplex**>(thrust::raw_pointer_cast(U_cols_vec.data()));
  cuDoubleComplex** U_R = reinterpret_cast<cuDoubleComplex**>(thrust::raw_pointer_cast(U_R_vec.data()));
  cuDoubleComplex** V = reinterpret_cast<cuDoubleComplex**>(thrust::raw_pointer_cast(V_rows_vec.data()));
  cuDoubleComplex** V_R = reinterpret_cast<cuDoubleComplex**>(thrust::raw_pointer_cast(V_R_vec.data()));
  cuDoubleComplex** B = reinterpret_cast<cuDoubleComplex**>(thrust::raw_pointer_cast(B_ind_vec.data()));
  cuDoubleComplex** B_Cols = reinterpret_cast<cuDoubleComplex**>(thrust::raw_pointer_cast(B_cols_vec.data()));
  cuDoubleComplex** B_I_Cols = reinterpret_cast<cuDoubleComplex**>(thrust::raw_pointer_cast(B_R_vec.data()));
  cuDoubleComplex** A_SR_Rows = reinterpret_cast<cuDoubleComplex**>(thrust::raw_pointer_cast(A_sr_rows_vec.data()));
  cuDoubleComplex** ACC = reinterpret_cast<cuDoubleComplex**>(thrust::raw_pointer_cast(AC_ind_vec.data()));

  int* ipiv = thrust::raw_pointer_cast(Ipiv.data());
  int* info = thrust::raw_pointer_cast(Info.data());

  long long rdim = bdim - rank;
  int info_host = 0;
  cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.), minus_one = make_cuDoubleComplex(-1., 0.);

  auto mapV = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), swapXY(bdim, block));
  auto mapD = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), StridedBlock(rank, rank, bdim, reinterpret_cast<thrust::complex<double>**>(A_SS)));

  if (M == 1)
    comm.level_merge(A, block * lenA);

  cudaMemcpyAsync(Udata, Q, block * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(Adata, A, block * lenA * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);

  thrust::gather(thrust::cuda::par.on(stream), mapV, mapV + block * M, thrust::make_transform_iterator(Uvec.begin() + (D * block), conjugateDouble()), Vvec.begin());
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
    thrust::transform(thrust::cuda::par.on(stream), mapD, mapD + (rank * rank * M), ACvec.begin(), mapD, thrust::plus<thrust::complex<double>>());
  }
  cudaStreamSynchronize(stream);

  cudaMemcpy(A, Adata, block * lenA * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  cudaMemcpy(&R[block * D], Vdata, block * M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
}
