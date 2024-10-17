
#include <factorize.cuh>
#include <comm-mpi.hpp>

#include <thrust/device_ptr.h>
#include <thrust/complex.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/inner_product.h>

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

void compute_factorize(deviceHandle_t handle, deviceMatrixDesc_t A, deviceMatrixDesc_t Al, const ColCommMPI& comm, const ncclComms nccl_comms) {
  long long bdim = A.bdim;
  long long rank = A.rank;
  long long block = bdim * bdim;
  long long rblock = rank * rank;

  long long D = comm.oLocal();
  long long M = comm.lenLocal();
  long long N = comm.lenNeighbors();
  long long lenA = comm.ARowOffsets[M];
  long long lenL = comm.LowerIndA.size();

  cudaStream_t stream = handle->compute_stream;
  cublasHandle_t cublasH = handle->cublasH;

  long long rdim = bdim - rank;
  long long reduc_len = A.reducLen;
  int info_host = 0;
  STD_CTYPE constants[3] = { 1., 0., -1. };
  CUDA_CTYPE& one = reinterpret_cast<CUDA_CTYPE&>(constants[0]);
  CUDA_CTYPE& zero = reinterpret_cast<CUDA_CTYPE&>(constants[1]); 
  CUDA_CTYPE& minus_one = reinterpret_cast<CUDA_CTYPE&>(constants[2]); 

  auto inc_iter = thrust::make_counting_iterator(0ll);
  auto mapV = thrust::make_transform_iterator(inc_iter, swapXY(bdim, bdim));
  thrust::device_ptr<THRUST_CTYPE> Uptr(reinterpret_cast<THRUST_CTYPE*>(&(A.Udata)[D * block]));
  thrust::device_ptr<THRUST_CTYPE> Vptr(reinterpret_cast<THRUST_CTYPE*>(A.Vdata));
  thrust::gather(thrust::cuda::par.on(stream), mapV, mapV + (block * M), thrust::make_transform_iterator(Uptr, conjugateFunc()), Vptr);
  
  if (0 < lenL) {
    long long len = Al.rank * Al.rank * lenL;
    thrust::for_each(thrust::cuda::par.on(stream), inc_iter, inc_iter + len, copyFunc(Al.rank, Al.rank, Al.A_unsort, Al.bdim, A.A_dst, bdim));
  }

  auto dup = nccl_comms->find(comm.DupComm);
  if (M == 1) {
    auto merge = nccl_comms->find(comm.MergeComm);
    if (comm.MergeComm != MPI_COMM_NULL)
      ncclAllReduce(const_cast<const CUDA_CTYPE*>(A.Adata), A.Adata, block * lenA * 2, ncclDouble, ncclSum, (*merge).second, stream);
    if (comm.DupComm != MPI_COMM_NULL)
      ncclBroadcast(const_cast<const CUDA_CTYPE*>(A.Adata), A.Adata, block * lenA * 2, ncclDouble, 0, (*dup).second, stream);
  }

  cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, A.V_rows, bdim, A.A_ss, bdim, &zero, A.B_ind, bdim, M);
  cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, A.V_rows, bdim, A.B_ind, bdim, &zero, A.A_ss, bdim, M);

  cublasZgetrfBatched(cublasH, rdim, A.A_rr, bdim, A.Ipiv, A.Info, M);
  cublasZgetrsBatched(cublasH, CUBLAS_OP_N, rdim, bdim, A.A_rr, bdim, A.Ipiv, A.V_R, bdim, &info_host, M);

  if (0 < rank) {
    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, bdim, &one, A.V_R, bdim, A.B_ind, bdim, &zero, A.A_rs, bdim, M);
    cudaMemsetAsync(A.ACdata, 0, reduc_len * M * rblock * sizeof(CUDA_CTYPE), stream);

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      cublasZgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, bdim, bdim, bdim, &one, &(A.U_cols)[i], bdim, &(A.A_ss)[i], bdim, &zero, A.B_ind, bdim, len);
      cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, bdim, bdim, bdim, &one, &(A.V_rows)[i], bdim, A.B_ind, bdim, &zero, &(A.A_ss)[i], bdim, len);
    }

    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, A.A_sr_rows, bdim, A.A_rs, bdim, &one, A.A_ss, bdim, lenA);
    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rdim, rdim, bdim, &one, A.V_R, bdim, A.U_R, bdim, &zero, A.B_R, bdim, M);
    thrust::for_each(thrust::cuda::par.on(stream), inc_iter, inc_iter + (rdim * rank * M), copyFunc(rdim, rank, const_cast<const CUDA_CTYPE**>(A.A_rs), bdim, &(A.B_ind)[D], bdim));
    
    ncclGroupStart();
    for (long long p = 0; p < (long long)comm.NeighborComm.size(); p++) {
      long long start = comm.BoxOffsets[p] * block;
      long long len = comm.BoxOffsets[p + 1] * block - start;
      auto neighbor = nccl_comms->find(comm.NeighborComm[p].second);
      ncclBroadcast(const_cast<const CUDA_CTYPE*>(&(A.Bdata)[start]), &(A.Bdata)[start], len * 2, ncclDouble, comm.NeighborComm[p].first, (*neighbor).second, stream);
    }

    if (comm.DupComm != MPI_COMM_NULL)
      ncclBroadcast(const_cast<const CUDA_CTYPE*>(A.Bdata), A.Bdata, block * N * 2, ncclDouble, 0, (*dup).second, stream);
    ncclGroupEnd();

    if (M < lenA)
      cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &(A.A_sr)[M], bdim, &(A.B_cols)[M], bdim, &one, &(A.A_ss)[M], bdim, lenA - M);

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, rdim, &one, &(A.B_R)[i], bdim, &(A.A_sr)[i], bdim, &zero, A.B_ind, bdim, len);
      cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &(A.A_sr)[i], bdim, A.B_ind, bdim, &zero, &A.AC_ind[i], rank, len);
    }
    cublasZgemvStridedBatched(cublasH, CUBLAS_OP_N, M * rank, reduc_len, &one, A.ACdata, M * rblock, M * rank, A.ONEdata, 1, 0, &zero, A.Bdata, 1, M * rank, rank);

    thrust::device_ptr<THRUST_CTYPE> ACptr(reinterpret_cast<THRUST_CTYPE*>(A.Bdata));
    auto Aiter = thrust::make_transform_iterator(inc_iter, StridedBlock(rank, rank, bdim, reinterpret_cast<THRUST_CTYPE**>(A.A_ss)));
    thrust::transform(thrust::cuda::par.on(stream), Aiter, Aiter + (rblock * M), ACptr, Aiter, thrust::plus<THRUST_CTYPE>());
  }
}

int check_info(deviceMatrixDesc_t A, const ColCommMPI& comm) {
  long long M = comm.lenLocal();
  thrust::device_ptr<int> info_ptr(A.Info);
  int sum = thrust::inner_product(info_ptr, info_ptr + M, info_ptr, 0);
  return 0 < sum;
}
