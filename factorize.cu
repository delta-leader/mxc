
#include <factorize.cuh>
#include <comm-mpi.hpp>
#include <algorithm>
#include <numeric>
#include <tuple>

#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>

void init_gpu_envs(cudaStream_t* stream, cublasHandle_t* cublasH, std::map<const MPI_Comm, ncclComm_t>& nccl_comms, const std::vector<MPI_Comm>& comms, MPI_Comm world) {
  int mpi_rank, num_device;
  if (cudaGetDeviceCount(&num_device) != cudaSuccess)
    return;

  MPI_Comm_rank(world, &mpi_rank);
  cudaSetDevice(mpi_rank % num_device);
  cudaStreamCreate(stream);
  cublasCreate(cublasH);
  cublasSetStream(*cublasH, *stream);

  long long len = comms.size();
  std::vector<ncclUniqueId> ids(len);
  std::vector<ncclComm_t> nccl_alloc(len);

  ncclGroupStart();
  for (long long i = 0; i < len; i++) {
    int rank, size;
    MPI_Comm_rank(comms[i], &rank);
    MPI_Comm_size(comms[i], &size);
    if (rank == 0)
      ncclGetUniqueId(&ids[i]);
    MPI_Bcast(reinterpret_cast<void*>(&ids[i]), sizeof(ncclUniqueId), MPI_BYTE, 0, comms[i]);
    ncclCommInitRank(&nccl_alloc[i], size, ids[i], rank);
  }
  ncclGroupEnd();

  for (long long i = 0; i < len; i++)
    nccl_comms.insert(std::make_pair(comms[i], nccl_alloc[i]));
}

void finalize_gpu_envs(cudaStream_t stream, cublasHandle_t cublasH, std::map<const MPI_Comm, ncclComm_t>& nccl_comms) {
  cudaDeviceSynchronize();
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  for (auto& c : nccl_comms)
    ncclCommDestroy(c.second);
  nccl_comms.clear();
  cudaDeviceReset();
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
  long long ldx, ldy, ldz;
  setDevicePtr(T* data, long long ldx, long long ldy = 0, long long ldz = 0) : 
    data(data), ldx(ldx), ldy(ldy), ldz(ldz) {}
  __host__ __device__ T* operator()(long long x) const {
    return data + x * ldx;
  }
  __host__ __device__ T* operator()(long long y, long long x) const {
    return data + (x * ldx + y * ldy);
  }
  __host__ __device__ T* operator()(thrust::tuple<long long, long long, long long> x) const {
    return data + (thrust::get<0>(x) * ldx + thrust::get<1>(x) * ldy + thrust::get<2>(x) * ldz);
  }
};

void createMatrixDesc(deviceMatrixDesc_t* desc, long long bdim, long long rank, long long lower_rank, const ColCommMPI& comm) {
  desc->bdim = bdim;
  desc->rank = rank;
  long long lenA = desc->lenA = comm.ARowOffsets.back();
  long long M = desc->M = comm.lenLocal();
  long long N = desc->N =comm.lenNeighbors();
  long long diag_offset = desc->diag_offset = comm.oLocal();

  thrust::device_vector<long long> ARowOffset(comm.ARowOffsets.begin(), comm.ARowOffsets.end());
  thrust::device_vector<long long> ARows(lenA, 0ll);
  thrust::device_vector<long long> ACols(comm.AColumns.begin(), comm.AColumns.end());
  thrust::device_vector<long long> ADistCols(lenA);
  thrust::device_vector<long long> AInd(lenA);
  thrust::device_vector<long long> keys(lenA);
  
  auto one_iter = thrust::make_constant_iterator(1ll);
  thrust::scatter(one_iter, one_iter + (M - 1), ARowOffset.begin() + 1, ARows.begin()); 
  thrust::inclusive_scan(ARows.begin(), ARows.end(), ARows.begin());
  thrust::exclusive_scan_by_key(ARows.begin(), ARows.end(), one_iter, ADistCols.begin(), 0ll);

  thrust::transform(ARows.begin(), ARows.end(), ACols.begin(), keys.begin(), keysDLU(diag_offset, M, N));
  thrust::sequence(AInd.begin(), AInd.end(), 0);
  thrust::sort_by_key(keys.begin(), keys.end(), thrust::make_zip_iterator(ARows.begin(), ACols.begin(), ADistCols.begin(), AInd.begin()));

  long long ReducLen = desc->ReducLen = 1ll + thrust::reduce(ADistCols.begin(), ADistCols.end(), 0ll, thrust::maximum<long long>());
  long long lenL = comm.LowerIndA.size();
  const thrust::tuple<long long, long long, long long>* commL = reinterpret_cast<const thrust::tuple<long long, long long, long long>*>(comm.LowerIndA.data());
  thrust::device_vector<thrust::tuple<long long, long long, long long>> LInd(commL, commL + lenL);

  cudaMalloc(reinterpret_cast<void**>(&desc->A_ss), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_sr), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_rs), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_rr), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_sr_rows), lenA * sizeof(CUDA_CTYPE*));

  cudaMalloc(reinterpret_cast<void**>(&desc->U_cols), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->U_R), M * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->V_rows), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->V_R), M * sizeof(CUDA_CTYPE*));

  cudaMalloc(reinterpret_cast<void**>(&desc->B_ind), M * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->B_cols), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->B_R), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->AC_ind), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->L_dst), lenL * sizeof(CUDA_CTYPE*));

  long long block = bdim * bdim;
  long long rblock = rank * rank;

  cudaMalloc(reinterpret_cast<void**>(&desc->Adata), lenA * block * sizeof(CUDA_CTYPE));
  cudaMalloc(reinterpret_cast<void**>(&desc->Udata), N * block * sizeof(CUDA_CTYPE));
  cudaMalloc(reinterpret_cast<void**>(&desc->Vdata), M * block * sizeof(CUDA_CTYPE));
  cudaMalloc(reinterpret_cast<void**>(&desc->Bdata), N * block * sizeof(CUDA_CTYPE));
  cudaMalloc(reinterpret_cast<void**>(&desc->ACdata), ReducLen * M * rblock* sizeof(CUDA_CTYPE));
  cudaMalloc(reinterpret_cast<void**>(&desc->Ipiv), M * bdim * sizeof(int));
  cudaMalloc(reinterpret_cast<void**>(&desc->Info), M * sizeof(int));

  auto inc_iter = thrust::make_counting_iterator(0ll);
  auto rwise_diag_iter = thrust::make_permutation_iterator(AInd.begin(), ARows.begin());
  long long offset_SR = rank * bdim, offset_RS = rank, offset_RR = rank * (bdim + 1);
  thrust::transform(AInd.begin(), AInd.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->A_ss), setDevicePtr(desc->Adata, block));
  thrust::transform(AInd.begin(), AInd.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->A_sr), setDevicePtr(&(desc->Adata)[offset_SR], block));
  thrust::transform(AInd.begin(), AInd.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->A_rs), setDevicePtr(&(desc->Adata)[offset_RS], block));
  thrust::transform(AInd.begin(), AInd.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->A_rr), setDevicePtr(&(desc->Adata)[offset_RR], block));
  thrust::transform(rwise_diag_iter, rwise_diag_iter + lenA, thrust::device_ptr<CUDA_CTYPE*>(desc->A_sr_rows), setDevicePtr(&(desc->Adata)[offset_SR], block));

  thrust::transform(ACols.begin(), ACols.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->U_cols), setDevicePtr(desc->Udata, block));
  thrust::transform(ACols.begin(), ACols.begin() + M, thrust::device_ptr<CUDA_CTYPE*>(desc->U_R), setDevicePtr(&(desc->Udata)[offset_SR], block));
  thrust::transform(ARows.begin(), ARows.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->V_rows), setDevicePtr(desc->Vdata, block));
  thrust::transform(inc_iter, inc_iter + M, thrust::device_ptr<CUDA_CTYPE*>(desc->V_R), setDevicePtr(&(desc->Vdata)[offset_RS], block));

  thrust::transform(inc_iter, inc_iter + N, thrust::device_ptr<CUDA_CTYPE*>(desc->B_ind), setDevicePtr(desc->Bdata, block));
  thrust::transform(ACols.begin(), ACols.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->B_cols), setDevicePtr(desc->Bdata, block));
  thrust::transform(ACols.begin(), ACols.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->B_R), setDevicePtr(&(desc->Bdata)[offset_SR], block));
  thrust::transform(ARows.begin(), ARows.end(), ADistCols.begin(), thrust::device_ptr<CUDA_CTYPE*>(desc->AC_ind), setDevicePtr(desc->ACdata, M * rblock, rblock));
  thrust::transform(LInd.begin(), LInd.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->L_dst), setDevicePtr(desc->Adata, block, bdim * lower_rank, lower_rank));
}

void destroyMatrixDesc(deviceMatrixDesc_t desc) {
  cudaFree(desc.A_ss);
  cudaFree(desc.A_sr);
  cudaFree(desc.A_rs);
  cudaFree(desc.A_rr);
  cudaFree(desc.A_sr_rows);

  cudaFree(desc.U_cols);
  cudaFree(desc.U_R);
  cudaFree(desc.V_rows);
  cudaFree(desc.V_R);

  cudaFree(desc.B_ind);
  cudaFree(desc.B_cols);
  cudaFree(desc.B_R);
  cudaFree(desc.AC_ind);
  cudaFree(desc.L_dst);

  cudaFree(desc.Adata);
  cudaFree(desc.Udata);
  cudaFree(desc.Vdata);
  cudaFree(desc.Bdata);
  cudaFree(desc.ACdata);
  cudaFree(desc.Ipiv);
  cudaFree(desc.Info);
}

void copyDataInMatrixDesc(deviceMatrixDesc_t desc, const STD_CTYPE* A, const STD_CTYPE* U, cudaStream_t stream) {
  long long block = desc.bdim * desc.bdim * sizeof(CUDA_CTYPE);
  cudaMemcpyAsync(desc.Adata, A, block * desc.lenA, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(desc.Udata, U, block * desc.N, cudaMemcpyHostToDevice, stream);
}

void copyDataOutMatrixDesc(deviceMatrixDesc_t desc, STD_CTYPE* A, STD_CTYPE* R, cudaStream_t stream) {
  long long block = desc.bdim * desc.bdim * sizeof(CUDA_CTYPE);
  cudaMemcpyAsync(A, desc.Adata, block * desc.lenA, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(R, desc.Vdata, block * desc.M, cudaMemcpyDeviceToHost, stream);
}

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

void compute_factorize(cudaStream_t stream, cublasHandle_t cublasH, long long bdim, long long rank, CUDA_CTYPE* A, CUDA_CTYPE* R, const CUDA_CTYPE* Q, long long ldim, long long lrank, const CUDA_CTYPE* L, const ColCommMPI& comm, const std::map<const MPI_Comm, ncclComm_t>& nccl_comms) {
  long long block = bdim * bdim;
  long long rblock = rank * rank;
  long long D = comm.oLocal();
  long long M = comm.lenLocal();
  long long N = comm.lenNeighbors();
  
  const long long* ARows = comm.ARowOffsets.data();
  const long long* ACols = comm.AColumns.data();
  long long lenA = comm.ARowOffsets[M];
  long long lenL = comm.LowerIndA.size();
  
  const thrust::tuple<long long, long long, long long>* commL = reinterpret_cast<const thrust::tuple<long long, long long, long long>*>(comm.LowerIndA.data());

  thrust::device_vector<long long> ARowOffset_vec(comm.ARowOffsets.begin(), comm.ARowOffsets.end());
  thrust::device_vector<long long> ARows_vec(lenA, 0ll);
  thrust::device_vector<long long> ACols_vec(comm.AColumns.begin(), comm.AColumns.end());
  thrust::device_vector<long long> ADistCols_vec(lenA);
  thrust::device_vector<long long> AInd_vec(lenA);
  thrust::device_vector<long long> keys(lenA);
  thrust::device_vector<thrust::tuple<long long, long long, long long>> LInd_vec(commL, commL + lenL);

  thrust::device_vector<thrust::complex<double>*> A_ss_vec(lenA), A_sr_vec(lenA), A_rs_vec(lenA), A_rr_vec(lenA);
  thrust::device_vector<thrust::complex<double>*> U_cols_vec(lenA), V_rows_vec(lenA), U_R_vec(M), V_R_vec(M), B_ind_vec(N), B_cols_vec(lenA), B_R_vec(lenA);
  thrust::device_vector<thrust::complex<double>*> A_sr_rows_vec(lenA), AC_ind_vec(lenA);
  thrust::device_vector<thrust::complex<double>*> L_ss_vec(lenL), L_dst_vec(lenL);

  thrust::device_vector<thrust::complex<double>> Avec(lenA * block);
  thrust::device_vector<thrust::complex<double>> Bvec(N * block);
  thrust::device_vector<thrust::complex<double>> Uvec(N * block);
  thrust::device_vector<thrust::complex<double>> Vvec(M * block);
  thrust::device_vector<thrust::complex<double>> Lvec(lenL * ldim * ldim);
  
  thrust::device_vector<int> Ipiv(M * bdim);
  thrust::device_vector<int> Info(M);

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
  thrust::device_vector<thrust::complex<double>> ACvec(reduc_len * M * rblock, thrust::complex<double>(0., 0.));

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
  thrust::transform(ARows_vec.begin(), ARows_vec.end(), ADistCols_vec.begin(), AC_ind_vec.begin(), setDevicePtr(thrust::raw_pointer_cast(ACvec.data()), M * rblock, rblock));
  thrust::transform(inc_iter, inc_iter + lenL, L_ss_vec.begin(), setDevicePtr(thrust::raw_pointer_cast(Lvec.data()), ldim * ldim));
  thrust::transform(LInd_vec.begin(), LInd_vec.end(), L_dst_vec.begin(), setDevicePtr(thrust::raw_pointer_cast(Avec.data()), block, bdim * lrank, lrank));

  cuDoubleComplex* Adata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(Avec.data()));
  cuDoubleComplex* Udata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(Uvec.data()));
  cuDoubleComplex* Vdata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(Vvec.data()));
  cuDoubleComplex* Bdata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(Bvec.data()));
  cuDoubleComplex* ACdata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(ACvec.data()));
  cuDoubleComplex* Ldata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(Lvec.data()));

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
  cuDoubleComplex** L_SS = reinterpret_cast<cuDoubleComplex**>(thrust::raw_pointer_cast(L_ss_vec.data()));
  cuDoubleComplex** L_DST = reinterpret_cast<cuDoubleComplex**>(thrust::raw_pointer_cast(L_dst_vec.data()));

  int* ipiv = thrust::raw_pointer_cast(Ipiv.data());
  int* info = thrust::raw_pointer_cast(Info.data());

  long long rdim = bdim - rank;
  int info_host = 0;
  cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.), minus_one = make_cuDoubleComplex(-1., 0.);

  auto mapV = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), swapXY(bdim, block));
  auto mapD = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), StridedBlock(rank, rank, bdim, reinterpret_cast<thrust::complex<double>**>(A_SS)));

  cudaMemcpy(Adata, A, block * lenA * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
  cudaMemcpy(Udata, Q, block * N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
  thrust::gather(thrust::cuda::par.on(stream), mapV, mapV + block * M, thrust::make_transform_iterator(Uvec.begin() + (D * block), conjugateDouble()), Vvec.begin());

  if (0 < lenL) {
    cudaMemcpy(Ldata, L, ldim * ldim * lenL * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    thrust::for_each(thrust::cuda::par.on(stream), inc_iter, inc_iter + (lrank * lrank * lenL), copyFunc(lrank, lrank, const_cast<const cuDoubleComplex**>(L_SS), ldim, L_DST, bdim));
  }

  auto dup = nccl_comms.find(comm.DupComm);
  if (M == 1) {
    auto merge = nccl_comms.find(comm.MergeComm);
    if (comm.MergeComm != MPI_COMM_NULL)
      ncclAllReduce((const void*)Adata, Adata, block * lenA * 2, ncclDouble, ncclSum, (*merge).second, stream);
    if (comm.DupComm != MPI_COMM_NULL)
      ncclBroadcast((const void*)Adata, Adata, block * lenA * 2, ncclDouble, 0, (*dup).second, stream);
  }

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

    ncclGroupStart();
    for (long long p = 0; p < (long long)comm.NeighborComm.size(); p++) {
      long long start = comm.BoxOffsets[p] * block;
      long long len = comm.BoxOffsets[p + 1] * block - start;
      auto neighbor = nccl_comms.find(comm.NeighborComm[p].second);
      ncclBroadcast((const void*)&Bdata[start], &Bdata[start], len * 2, ncclDouble, comm.NeighborComm[p].first, (*neighbor).second, stream);
    }
    ncclGroupEnd();

    if (comm.DupComm != MPI_COMM_NULL)
      ncclBroadcast((const void*)Bdata, Bdata, block * N * 2, ncclDouble, 0, (*dup).second, stream);

    if (M < lenA)
      cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &A_SR[M], bdim, &B_Cols[M], bdim, &one, &A_SS[M], bdim, lenA - M);

    for (long long i = M; i < lenA; i += N) {
      long long len = std::min(lenA - i, N);
      cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rdim, rank, rdim, &one, &B_I_Cols[i], bdim, &A_SR[i], bdim, &zero, B, bdim, len);
      cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, rank, rank, rdim, &minus_one, &A_SR[i], bdim, B, bdim, &zero, &ACC[i], rank, len);
    }

    while (1 < reduc_len) {
      long long len = reduc_len * rblock * M;
      reduc_len = (reduc_len + 1) / 2;
      long long tail_start = reduc_len * rblock * M;
      long long tail_len = len - tail_start;
      cublasZaxpy(cublasH, tail_len, &one, &ACdata[tail_start], 1, ACdata, 1);
    }
    thrust::transform(thrust::cuda::par.on(stream), mapD, mapD + (rblock * M), ACvec.begin(), mapD, thrust::plus<thrust::complex<double>>());
  }
  cudaStreamSynchronize(stream);

  cudaMemcpy(A, Adata, block * lenA * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
  cudaMemcpy(R, Vdata, block * M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
}
