
#include <factorize.cuh>
#include <comm-mpi.hpp>
#include <algorithm>
#include <numeric>
#include <tuple>

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

void compute_factorize(cublasHandle_t cublasH, long long bdim, long long rank, long long D, long long M, long long N, const long long ARows[], const long long ACols[], std::complex<double>* A, std::complex<double>* R, const std::complex<double>* Q, const ColCommMPI& comm) {
  long long block = bdim * bdim;
  long long lenA = ARows[M];
  long long lenB = std::max(std::min(16 * M, lenA), N);

  cudaStream_t stream;
  cublasGetStream(cublasH, &stream);

  thrust::device_vector<long long> row_offsets(ARows, ARows + M);
  thrust::device_vector<long long> rows(lenA, 0ll);
  thrust::device_vector<long long> cols(ACols, ACols + lenA);
  thrust::device_vector<long long> keys(lenA);
  thrust::device_vector<long long> indices(lenA);
  thrust::device_vector<cuDoubleComplex*> a_ss(lenA), a_sr(lenA), a_rs(lenA), a_rr(lenA);
  thrust::device_vector<cuDoubleComplex*> u(lenA), v(lenA), u_r(M), v_r(M), b(lenB), b_i(M);
  thrust::device_vector<cuDoubleComplex*> a_sr_cols(lenA), b_cols(lenA);

  thrust::device_vector<cuDoubleComplex> Avec(lenA * block);
  thrust::device_vector<cuDoubleComplex> Bvec(lenB * block);
  thrust::device_vector<cuDoubleComplex> Uvec(N * block);
  thrust::device_vector<cuDoubleComplex> Vvec(M * block);
  thrust::device_vector<cuDoubleComplex> ACvec(lenB * rank * rank);
  
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
  cuDoubleComplex* ACdata = thrust::raw_pointer_cast(ACvec.data());

  cudaMemcpyAsync(Udata, Q, block * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(Adata, A, block * lenA * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);

  thrust::scatter(one_iter, one_iter + (M - 1), row_offsets.begin() + 1, rows.begin()); 
  thrust::inclusive_scan(rows.begin(), rows.end(), rows.begin());

  thrust::transform(rows.begin(), rows.end(), cols.begin(), keys.begin(), keysDLU(D, M, N));
  thrust::sequence(indices.begin(), indices.end(), 0);
  thrust::sort_by_key(keys.begin(), keys.end(), thrust::make_zip_iterator(rows.begin(), cols.begin(), indices.begin()));

  long long offset_SR = rank * bdim, offset_RS = rank, offset_RR = rank * (bdim + 1);
  thrust::transform(indices.begin(), indices.end(), a_ss.begin(), setDevicePtr(Adata, block));
  thrust::transform(indices.begin(), indices.end(), a_sr.begin(), setDevicePtr(Adata + offset_SR, block));
  thrust::transform(indices.begin(), indices.end(), a_rs.begin(), setDevicePtr(Adata + offset_RS, block));
  thrust::transform(indices.begin(), indices.end(), a_rr.begin(), setDevicePtr(Adata + offset_RR, block));
  thrust::transform(cols.begin(), cols.end(), u.begin(), setDevicePtr(Udata, block));
  thrust::transform(cols.begin(), cols.begin() + M, u_r.begin(), setDevicePtr(Udata + offset_SR, block));
  thrust::transform(rows.begin(), rows.end(), v.begin(), setDevicePtr(Vdata, block));
  thrust::transform(rows.begin(), rows.begin() + M, v_r.begin(), setDevicePtr(Vdata + offset_RS, block));

  thrust::transform(inc_iter, inc_iter + lenB, b.begin(), setDevicePtr(Bdata, block));
  thrust::transform(cols.begin(), cols.begin() + M, b_i.begin(), setDevicePtr(Bdata + offset_SR, block));
  thrust::transform(rwise_diag_iter, rwise_diag_iter + lenA, a_sr_cols.begin(), setDevicePtr(Adata + offset_SR, block));
  thrust::transform(cols.begin(), cols.end(), b_cols.begin(), setDevicePtr(Bdata, block));

  cuDoubleComplex** A_SS = thrust::raw_pointer_cast(a_ss.data());
  cuDoubleComplex** A_SR = thrust::raw_pointer_cast(a_sr.data());
  cuDoubleComplex** A_RS = thrust::raw_pointer_cast(a_rs.data());
  cuDoubleComplex** A_RR = thrust::raw_pointer_cast(a_rr.data());
  cuDoubleComplex** U = thrust::raw_pointer_cast(u.data());
  cuDoubleComplex** U_R = thrust::raw_pointer_cast(u_r.data());
  cuDoubleComplex** V = thrust::raw_pointer_cast(v.data());
  cuDoubleComplex** V_R = thrust::raw_pointer_cast(v_r.data());
  cuDoubleComplex** B = thrust::raw_pointer_cast(b.data());
  cuDoubleComplex** B_I = thrust::raw_pointer_cast(b_i.data());
  cuDoubleComplex** A_SR_Cols = thrust::raw_pointer_cast(a_sr_cols.data());
  cuDoubleComplex** B_Cols = thrust::raw_pointer_cast(b_cols.data());

  int* ipiv = thrust::raw_pointer_cast(Ipiv.data());
  int* info = thrust::raw_pointer_cast(Info.data());

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
  cublasZgetrsBatched(cublasH, CUBLAS_OP_N, rdim, bdim, A_RR, bdim, ipiv, V_R, bdim, &info_host, M);

  if (0 < rank) {
    cublasZgetrsBatched(cublasH, CUBLAS_OP_N, rdim, rank, A_RR, bdim, ipiv, A_RS, bdim, &info_host, M);

    for (long long i = M; i < lenA; i += lenB) {
      long long len = std::min(lenA - i, lenB);
      cublasZgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, &U[i], bdim, &A_SS[i], bdim, &zero, B, bdim, len);
      cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, &V[i], bdim, B, bdim, &zero, &A_SS[i], bdim, len);
    }
    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, A_SR_Cols, bdim, A_RS, bdim, &one, A_SS, bdim, lenA);

    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rdim, rdim, bdim, &one, V_R, bdim, U_R, bdim, &zero, B_I, bdim, M);
    thrust::for_each(thrust::cuda::par.on(stream), inc_iter, inc_iter + (rdim * rank * M), copyFunc(rdim, rank, const_cast<const cuDoubleComplex**>(A_RS), bdim, &B[D], bdim));
    cudaMemcpyAsync(&R[D * block], &Bdata[D * block], M * block * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    comm.neighbor_bcast(R, Bsizes.data());
    cudaMemcpyAsync(Bdata, R, N * block * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);

    if (M < lenA)
      cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &A_SR[M], bdim, &B_Cols[M], bdim, &one, &A_SS[M], bdim, lenA - M);


  }
  cudaStreamSynchronize(stream);

  cudaMemcpy(A, Adata, block * lenA * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  cudaMemcpy(&R[block * D], Vdata, block * M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
}
