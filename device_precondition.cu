
#include <factorize.cuh>
#include <comm-mpi.hpp>

#include <numeric>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/partition.h>
#include <thrust/iterator/constant_iterator.h>

struct keysD {
  long long D;
  keysD(long long D) : D(D) {}
  __host__ __device__ bool operator()(thrust::tuple<long long, long long, long long, long long> x) const {
    return D + thrust::get<0>(x) == thrust::get<1>(x);
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

void createMatrixDesc(deviceMatrixDesc_t* desc, long long bdim, long long rank, deviceMatrixDesc_t lower, const ColCommMPI& comm) {
  desc->bdim = bdim;
  desc->rank = rank;
  desc->diag_offset = comm.oLocal();
  desc->lower_offset = (comm.LowerX + lower.diag_offset) * lower.rank;
  long long lenA = comm.ARowOffsets.back();
  long long M = comm.lenLocal();
  long long N = comm.lenNeighbors();

  thrust::device_vector<long long> ARowOffset(comm.ARowOffsets.begin(), comm.ARowOffsets.end());
  thrust::device_vector<long long> ARows(lenA, 0ll);
  thrust::device_vector<long long> ACols(comm.AColumns.begin(), comm.AColumns.end());
  thrust::device_vector<long long> ADistCols(lenA);
  thrust::device_vector<long long> AInd(lenA);
  
  auto one_iter = thrust::make_constant_iterator(1ll);
  auto A_iter = thrust::make_zip_iterator(ARows.begin(), ACols.begin(), ADistCols.begin(), AInd.begin());
  thrust::scatter(one_iter, one_iter + (M - 1), ARowOffset.begin() + 1, ARows.begin()); 
  thrust::inclusive_scan(ARows.begin(), ARows.end(), ARows.begin());
  thrust::exclusive_scan_by_key(ARows.begin(), ARows.end(), one_iter, ADistCols.begin(), 0ll);

  thrust::sequence(AInd.begin(), AInd.end(), 0);
  thrust::stable_partition(A_iter, A_iter + lenA, keysD(desc->diag_offset));

  desc->reducLen = 1ll + thrust::reduce(ADistCols.begin(), ADistCols.end(), 0ll, thrust::maximum<long long>());
  long long lenLA = comm.LowerIndA.size();
  const thrust::tuple<long long, long long, long long>* commLA = reinterpret_cast<const thrust::tuple<long long, long long, long long>*>(comm.LowerIndA.data());
  thrust::device_vector<thrust::tuple<long long, long long, long long>> LInd(commLA, commLA + lenLA);

  cudaMalloc(reinterpret_cast<void**>(&desc->A_ss), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_sr), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_rs), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_rr), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_sr_rows), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_dst), lenLA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->A_unsort), lenA * sizeof(CUDA_CTYPE*));

  cudaMalloc(reinterpret_cast<void**>(&desc->U_cols), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->U_R), M * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->V_rows), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->V_R), M * sizeof(CUDA_CTYPE*));

  cudaMalloc(reinterpret_cast<void**>(&desc->B_ind), N * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->B_cols), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->B_R), lenA * sizeof(CUDA_CTYPE*));

  cudaMalloc(reinterpret_cast<void**>(&desc->X_cols), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->Y_R_cols), lenA * sizeof(CUDA_CTYPE*));

  cudaMalloc(reinterpret_cast<void**>(&desc->AC_X), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->AC_X_R), lenA * sizeof(CUDA_CTYPE*));
  cudaMalloc(reinterpret_cast<void**>(&desc->AC_ind), lenA * sizeof(CUDA_CTYPE*));

  long long block = bdim * bdim;
  long long rblock = rank * rank;
  long long acc_len = desc->reducLen * M * std::max(rblock, bdim);

  cudaMalloc(reinterpret_cast<void**>(&desc->Adata), lenA * block * sizeof(CUDA_CTYPE));
  cudaMalloc(reinterpret_cast<void**>(&desc->Udata), N * block * sizeof(CUDA_CTYPE));
  cudaMalloc(reinterpret_cast<void**>(&desc->Vdata), M * block * sizeof(CUDA_CTYPE));
  cudaMalloc(reinterpret_cast<void**>(&desc->Bdata), N * block * sizeof(CUDA_CTYPE));
  cudaMalloc(reinterpret_cast<void**>(&desc->ACdata), acc_len * sizeof(CUDA_CTYPE));

  cudaMalloc(reinterpret_cast<void**>(&desc->Xdata), N * bdim * sizeof(CUDA_CTYPE));
  cudaMalloc(reinterpret_cast<void**>(&desc->Ydata), N * bdim * sizeof(CUDA_CTYPE));
  cudaMalloc(reinterpret_cast<void**>(&desc->ONEdata), desc->reducLen * sizeof(CUDA_CTYPE));
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
  thrust::transform(LInd.begin(), LInd.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->A_dst), setDevicePtr(desc->Adata, block, bdim * lower.rank, lower.rank));
  thrust::transform(inc_iter, inc_iter + lenA, thrust::device_ptr<const CUDA_CTYPE*>(desc->A_unsort), setDevicePtr(desc->Adata, block));

  thrust::transform(ACols.begin(), ACols.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->U_cols), setDevicePtr(desc->Udata, block));
  thrust::transform(ACols.begin(), ACols.begin() + M, thrust::device_ptr<CUDA_CTYPE*>(desc->U_R), setDevicePtr(&(desc->Udata)[offset_SR], block));
  thrust::transform(ARows.begin(), ARows.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->V_rows), setDevicePtr(desc->Vdata, block));
  thrust::transform(inc_iter, inc_iter + M, thrust::device_ptr<CUDA_CTYPE*>(desc->V_R), setDevicePtr(&(desc->Vdata)[offset_RS], block));

  thrust::transform(inc_iter, inc_iter + N, thrust::device_ptr<CUDA_CTYPE*>(desc->B_ind), setDevicePtr(desc->Bdata, block));
  thrust::transform(ACols.begin(), ACols.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->B_cols), setDevicePtr(desc->Bdata, block));
  thrust::transform(ACols.begin(), ACols.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->B_R), setDevicePtr(&(desc->Bdata)[offset_SR], block));

  thrust::transform(ACols.begin(), ACols.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->X_cols), setDevicePtr(desc->Xdata, rank));
  thrust::transform(ACols.begin(), ACols.end(), thrust::device_ptr<CUDA_CTYPE*>(desc->Y_R_cols), setDevicePtr(&(desc->Ydata)[offset_RS], bdim));

  thrust::transform(ARows.begin(), ARows.end(), ADistCols.begin(), thrust::device_ptr<CUDA_CTYPE*>(desc->AC_X), setDevicePtr(desc->ACdata, M * rank, rank));
  thrust::transform(ARows.begin(), ARows.end(), ADistCols.begin(), thrust::device_ptr<CUDA_CTYPE*>(desc->AC_X_R), setDevicePtr(&(desc->ACdata)[offset_RS], M * bdim, bdim));
  thrust::transform(ARows.begin(), ARows.end(), ADistCols.begin(), thrust::device_ptr<CUDA_CTYPE*>(desc->AC_ind), setDevicePtr(desc->ACdata, M * rblock, rblock));
  
  thrust::fill(thrust::device_ptr<CUDA_CTYPE>(desc->ONEdata), thrust::device_ptr<CUDA_CTYPE>(&(desc->ONEdata)[desc->reducLen]), make_cuDoubleComplex(1., 0.));
}

void destroyMatrixDesc(deviceMatrixDesc_t desc) {
  cudaFree(desc.A_ss);
  cudaFree(desc.A_sr);
  cudaFree(desc.A_rs);
  cudaFree(desc.A_rr);
  cudaFree(desc.A_sr_rows);
  cudaFree(desc.A_dst);
  cudaFree(desc.A_unsort);

  cudaFree(desc.U_cols);
  cudaFree(desc.U_R);
  cudaFree(desc.V_rows);
  cudaFree(desc.V_R);

  cudaFree(desc.B_ind);
  cudaFree(desc.B_cols);
  cudaFree(desc.B_R);
  cudaFree(desc.AC_ind);

  cudaFree(desc.X_cols);
  cudaFree(desc.Y_R_cols);
  cudaFree(desc.AC_X);
  cudaFree(desc.AC_X_R);

  cudaFree(desc.Adata);
  cudaFree(desc.Udata);
  cudaFree(desc.Vdata);
  cudaFree(desc.Bdata);
  cudaFree(desc.ACdata);

  cudaFree(desc.Xdata);
  cudaFree(desc.Ydata);
  cudaFree(desc.ONEdata);
  cudaFree(desc.Ipiv);
  cudaFree(desc.Info);
}

struct genXY {
  const long long* M, *A, *Y, *X;
  genXY(const long long* M, const long long* A, const long long* Y, const long long* X) : M(M), A(A), Y(Y), X(X) {}
  __host__ __device__ thrust::tuple<long long, long long> operator()(long long i, thrust::tuple<long long, long long> c) const {
    long long b = thrust::get<0>(c);
    long long m = M[b]; long long id = i - A[b];
    long long x = id / m; long long y = id - x * m;
    return thrust::make_tuple(Y[b] + y, X[b] + x);
  }
};

struct cmpXY {
  __host__ __device__ bool operator()(const thrust::tuple<long long, long long, thrust::complex<double>>& l, const thrust::tuple<long long, long long, thrust::complex<double>>& r) const {
    return thrust::get<0>(l) == thrust::get<0>(r) ? thrust::get<1>(l) < thrust::get<1>(r) : thrust::get<0>(l) < thrust::get<0>(r);
  }
};

long long computeCooNNZ(long long Mb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[]) {
  return std::transform_reduce(ARows, &ARows[Mb], RowDims, 0ll, std::plus<long long>(), [&](const long long& begin, long long rows) { 
    return rows * std::transform_reduce(&ACols[begin], &ACols[(&begin)[1]], 0ll, std::plus<long long>(), [&](long long col) { return ColDims[col]; }); });
}

void genCsrEntries(long long CsrM, long long devRowIndx[], long long devColIndx[], std::complex<double> devVals[], long long Mb, long long Nb, const long long RowDims[], const long long ColDims[], const long long ARows[], const long long ACols[]) {
  long long lenA = ARows[Mb];
  thrust::device_vector<long long> ARowOffset(ARows, &ARows[Mb + 1]);
  thrust::device_vector<long long> ARowIndx(lenA, 0ll);
  thrust::device_vector<long long> AColIndx(ACols, &ACols[lenA]);
  thrust::device_vector<long long> AOffsets(lenA + 1, 0ll);

  thrust::device_vector<long long> devRowOffsets(Mb + 1);
  thrust::device_vector<long long> devColOffsets(Nb + 1);
  thrust::device_vector<long long> devADimM(lenA);
  thrust::device_vector<long long> devAIndY(lenA);
  thrust::device_vector<long long> devAIndX(lenA);

  long long keys_len = std::max(CsrM, std::max(lenA, Mb));
  thrust::device_vector<long long> keys(keys_len);
  thrust::device_vector<long long> counts(keys_len);
  
  auto one_iter = thrust::make_constant_iterator(1ll);
  auto ydim_iter = thrust::make_permutation_iterator(devRowOffsets.begin(), ARowIndx.begin());
  auto xdim_iter = thrust::make_permutation_iterator(devColOffsets.begin(), AColIndx.begin());

  thrust::copy(RowDims, &RowDims[Mb], devRowOffsets.begin());
  thrust::copy(ColDims, &ColDims[Nb], devColOffsets.begin());

  auto counts_end = thrust::reduce_by_key(ARowOffset.begin() + 1, ARowOffset.begin() + Mb, one_iter, keys.begin(), counts.begin()).second;
  thrust::scatter(counts.begin(), counts_end, keys.begin(), ARowIndx.begin()); 
  thrust::inclusive_scan(ARowIndx.begin(), ARowIndx.end(), ARowIndx.begin());
  thrust::transform(ydim_iter, ydim_iter + lenA, xdim_iter, AOffsets.begin(), thrust::multiplies<long long>());
  thrust::exclusive_scan(AOffsets.begin(), AOffsets.end(), AOffsets.begin(), 0ll);

  thrust::copy(ydim_iter, ydim_iter + lenA, devADimM.begin());
  thrust::exclusive_scan(devRowOffsets.begin(), devRowOffsets.end(), devRowOffsets.begin(), 0ll);
  thrust::exclusive_scan(devColOffsets.begin(), devColOffsets.end(), devColOffsets.begin(), 0ll);
  thrust::copy(ydim_iter, ydim_iter + lenA, devAIndY.begin());
  thrust::copy(xdim_iter, xdim_iter + lenA, devAIndX.begin());

  long long NNZ = AOffsets.back();
  thrust::device_vector<long long> Rows(NNZ, 0ll);
  thrust::device_ptr<long long> RowsPtr(devRowIndx);
  thrust::device_ptr<long long> ColsPtr(devColIndx);
  thrust::device_ptr<thrust::complex<double>> Vals(reinterpret_cast<thrust::complex<double>*>(devVals));

  auto ind_iter = thrust::make_zip_iterator(Rows.begin(), ColsPtr);
  auto inc_iter = thrust::make_counting_iterator(0ll);
  auto sort_iter = thrust::make_zip_iterator(Rows.begin(), ColsPtr, Vals);

  const long long* Mptr = thrust::raw_pointer_cast(devADimM.data());
  const long long* Aptr = thrust::raw_pointer_cast(AOffsets.data());
  const long long* Yptr = thrust::raw_pointer_cast(devAIndY.data());
  const long long* Xptr = thrust::raw_pointer_cast(devAIndX.data());

  counts_end = thrust::reduce_by_key(AOffsets.begin() + 1, AOffsets.begin() + lenA, one_iter, keys.begin(), counts.begin()).second;
  thrust::scatter(counts.begin(), counts_end, keys.begin(), Rows.begin());
  thrust::inclusive_scan(Rows.begin(), Rows.end(), Rows.begin());
  thrust::transform(inc_iter, inc_iter + NNZ, ind_iter, ind_iter, genXY(Mptr, Aptr, Yptr, Xptr));
  thrust::sort(sort_iter, sort_iter + NNZ, cmpXY());

  counts_end = thrust::reduce_by_key(Rows.begin(), Rows.end(), one_iter, keys.begin(), counts.begin()).second;
  thrust::fill(RowsPtr, &RowsPtr[CsrM + 1], 0ll);
  thrust::scatter(counts.begin(), counts_end, keys.begin(), RowsPtr);
  thrust::exclusive_scan(RowsPtr, &RowsPtr[CsrM + 1], RowsPtr, 0ll);
}

void createHostMatrix(hostMatrix_t* h, long long bdim, long long lenA) {
  long long block = bdim * bdim * sizeof(CUDA_CTYPE);
  h->lenA = lenA;
  cudaMallocHost(reinterpret_cast<void**>(&h->Adata), h->lenA * block);
}

void destroyHostMatrix(hostMatrix_t h) {
  cudaFreeHost(h.Adata);
}

void copyDataInMatrixDesc(deviceMatrixDesc_t desc, long long lenA, const STD_CTYPE* A, long long lenU, const STD_CTYPE* U, cudaStream_t stream) {
  long long block = desc.bdim * desc.bdim * sizeof(CUDA_CTYPE);
  cudaMemcpyAsync(desc.Adata, A, block * lenA, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(desc.Udata, U, block * lenU, cudaMemcpyHostToDevice, stream);
}

void copyDataOutMatrixDesc(deviceMatrixDesc_t desc, long long lenA, STD_CTYPE* A, long long lenV, STD_CTYPE* V, cudaStream_t stream) {
  long long block = desc.bdim * desc.bdim * sizeof(CUDA_CTYPE);
  cudaMemcpyAsync(A, desc.Adata, block * lenA, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(V, desc.Vdata, block * lenV, cudaMemcpyDeviceToHost, stream);
}

