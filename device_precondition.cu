
#include <factorize.cuh>
#include <comm-mpi.hpp>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>
#include <thrust/iterator/constant_iterator.h>

struct keysDLU {
  long long D, M, N, U;
  keysDLU(long long D, long long M, long long N) : D(D), M(M), N(N), U(2 * M * N) {}
  __host__ __device__ long long operator()(long long y, long long x) const {
    long long diff = D + y - x;
    long long pred = (diff != 0) + (diff < 0);
    return (pred * M + y) * N + x;
  }
  __host__ __device__ bool operator()(long long i) const {
    return U <= i;
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

struct StridedSequence {
  long long begin, M, LD;
  StridedSequence(long long begin, long long M, long long LD) : begin(begin), M(M), LD(LD) {}
  __host__ __device__ long long operator()(long long i) const {
    long long x = i / M; long long y = i - x * M;
    return begin + y + x * LD;
  }
};

void createMatrixDesc(devicePreconditioner_t* desc, long long bdim, long long rank, devicePreconditioner_t lower, const ColCommMPI& comm) {
  desc->bdim = bdim;
  desc->rank = rank;
  desc->diag_offset = comm.oLocal();
  long long lenA = comm.ARowOffsets.back();
  long long M = comm.lenLocal();
  long long N = comm.lenNeighbors();

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

  keysDLU key(desc->diag_offset, M, N);
  thrust::transform(ARows.begin(), ARows.end(), ACols.begin(), keys.begin(), key);
  thrust::sequence(AInd.begin(), AInd.end(), 0);
  thrust::sort_by_key(keys.begin(), keys.end(), thrust::make_zip_iterator(ARows.begin(), ACols.begin(), ADistCols.begin(), AInd.begin()));

  desc->reducLen = 1ll + thrust::reduce(ADistCols.begin(), ADistCols.end(), 0ll, thrust::maximum<long long>());
  long long lenLA = comm.LowerIndA.size();
  const thrust::tuple<long long, long long, long long>* commLA = reinterpret_cast<const thrust::tuple<long long, long long, long long>*>(comm.LowerIndA.data());
  thrust::device_vector<thrust::tuple<long long, long long, long long>> LInd(commLA, commLA + lenLA);

  cudaMalloc(reinterpret_cast<void**>(&desc->A_ss), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_sr), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_rs), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_rr), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_sr_rows), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_unsort), lenA * sizeof(CUDA_CTYPE*));

  cudaMalloc(reinterpret_cast<void**>(&desc->U_cols), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->U_R), M * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->V_rows), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->V_R), M * sizeof(CUDA_CTYPE*));

  cudaMalloc(reinterpret_cast<void**>(&desc->B_ind), M * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->B_cols), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->B_R), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->AC_ind), lenA * sizeof(CUDA_CTYPE*));

  cudaMalloc(reinterpret_cast<void**>(&desc->Y_cols), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->Y_R_cols), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->AC_X), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->AC_X_R), lenA * sizeof(CUDA_CTYPE*));

  cudaMalloc(reinterpret_cast<void**>(&desc->L_dst), lenLA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->Xlocs), M * bdim * sizeof(long long));

  long long block = bdim * bdim;
  long long rblock = rank * rank;
  long long acc_len = desc->reducLen * M * std::max(rblock, bdim);

  cudaMalloc(reinterpret_cast<void**>(&desc->Adata), lenA * block * sizeof(CUDA_CTYPE));
  cudaMalloc(reinterpret_cast<void**>(&desc->Udata), N * block * sizeof(CUDA_CTYPE));
  cudaMalloc(reinterpret_cast<void**>(&desc->Vdata), M * block * sizeof(CUDA_CTYPE));
  cudaMalloc(reinterpret_cast<void**>(&desc->Bdata), N * block * sizeof(CUDA_CTYPE));
  cudaMalloc(reinterpret_cast<void**>(&desc->ACdata), desc->reducLen * acc_len * sizeof(CUDA_CTYPE));

  cudaMalloc(reinterpret_cast<void**>(&desc->Xdata), N * bdim * sizeof(CUDA_CTYPE));
  cudaMalloc(reinterpret_cast<void**>(&desc->Ydata), N * bdim * sizeof(CUDA_CTYPE));
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
  thrust::transform(inc_iter, inc_iter + lenA, thrust::device_ptr<const CUDA_CTYPE*>(desc->A_unsort), setDevicePtr(desc->Adata, block));

  thrust::transform(ACols.begin(), ACols.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->U_cols), setDevicePtr(desc->Udata, block));
  thrust::transform(ACols.begin(), ACols.begin() + M, thrust::device_ptr<CUDA_CTYPE*>(desc->U_R), setDevicePtr(&(desc->Udata)[offset_SR], block));
  thrust::transform(ARows.begin(), ARows.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->V_rows), setDevicePtr(desc->Vdata, block));
  thrust::transform(inc_iter, inc_iter + M, thrust::device_ptr<CUDA_CTYPE*>(desc->V_R), setDevicePtr(&(desc->Vdata)[offset_RS], block));

  thrust::transform(inc_iter, inc_iter + N, thrust::device_ptr<CUDA_CTYPE*>(desc->B_ind), setDevicePtr(desc->Bdata, block));
  thrust::transform(ACols.begin(), ACols.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->B_cols), setDevicePtr(desc->Bdata, block));
  thrust::transform(ACols.begin(), ACols.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->B_R), setDevicePtr(&(desc->Bdata)[offset_SR], block));
  thrust::transform(ARows.begin(), ARows.end(), ADistCols.begin(), thrust::device_ptr<CUDA_CTYPE*>(desc->AC_ind), setDevicePtr(desc->ACdata, M * rblock, rblock));

  thrust::transform(ACols.begin(), ACols.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->Y_cols), setDevicePtr(desc->Ydata, bdim));
  thrust::transform(ACols.begin(), ACols.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->Y_R_cols), setDevicePtr(&(desc->Ydata)[offset_RS], bdim));
  thrust::transform(ARows.begin(), ARows.end(), ADistCols.begin(), thrust::device_ptr<CUDA_CTYPE*>(desc->AC_X), setDevicePtr(desc->ACdata, M * bdim, bdim));
  thrust::transform(ARows.begin(), ARows.end(), ADistCols.begin(), thrust::device_ptr<CUDA_CTYPE*>(desc->AC_X_R), setDevicePtr(&(desc->ACdata)[offset_RS], M * bdim, bdim));
  
  long long startLX = (lower.diag_offset + comm.LowerX) * lower.bdim;
  thrust::transform(LInd.begin(), LInd.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->L_dst), setDevicePtr(desc->Adata, block, bdim * lower.rank, lower.rank));
  thrust::transform(inc_iter, inc_iter + (M * bdim), thrust::device_ptr<long long>(desc->Xlocs), StridedSequence(startLX, std::max(lower.rank, 1ll), std::max(lower.bdim, 1ll)));
}

void destroyMatrixDesc(devicePreconditioner_t desc) {
  cudaFree(desc.A_ss);
  cudaFree(desc.A_sr);
  cudaFree(desc.A_rs);
  cudaFree(desc.A_rr);
  cudaFree(desc.A_sr_rows);
  cudaFree(desc.A_unsort);

  cudaFree(desc.U_cols);
  cudaFree(desc.U_R);
  cudaFree(desc.V_rows);
  cudaFree(desc.V_R);

  cudaFree(desc.B_ind);
  cudaFree(desc.B_cols);
  cudaFree(desc.B_R);
  cudaFree(desc.AC_ind);

  cudaFree(desc.Y_cols);
  cudaFree(desc.Y_R_cols);
  cudaFree(desc.AC_X);
  cudaFree(desc.AC_X_R);

  cudaFree(desc.L_dst);
  cudaFree(desc.Xlocs);

  cudaFree(desc.Adata);
  cudaFree(desc.Udata);
  cudaFree(desc.Vdata);
  cudaFree(desc.Bdata);
  cudaFree(desc.ACdata);

  cudaFree(desc.Xdata);
  cudaFree(desc.Ydata);
  cudaFree(desc.Ipiv);
  cudaFree(desc.Info);
}

void createHostMatrix(hostMatrix_t* h, long long bdim, long long lenA) {
  long long block = bdim * bdim * sizeof(CUDA_CTYPE);
  h->lenA = lenA;
  cudaMallocHost(reinterpret_cast<void**>(&h->Adata), h->lenA * block);
}

void destroyHostMatrix(hostMatrix_t h) {
  cudaFreeHost(h.Adata);
}

void copyDataInMatrixDesc(devicePreconditioner_t desc, long long lenA, const STD_CTYPE* A, long long lenU, const STD_CTYPE* U, cudaStream_t stream) {
  long long block = desc.bdim * desc.bdim * sizeof(CUDA_CTYPE);
  cudaMemcpyAsync(desc.Adata, A, block * lenA, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(desc.Udata, U, block * lenU, cudaMemcpyHostToDevice, stream);
}

void copyDataOutMatrixDesc(devicePreconditioner_t desc, long long lenA, STD_CTYPE* A, long long lenV, STD_CTYPE* V, cudaStream_t stream) {
  long long block = desc.bdim * desc.bdim * sizeof(CUDA_CTYPE);
  cudaMemcpyAsync(A, desc.Adata, block * lenA, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(V, desc.Vdata, block * lenV, cudaMemcpyDeviceToHost, stream);
}

