
#include <factorize.cuh>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <mkl.h>

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
H2Factorize<DT>::H2Factorize(long long LD, long long lenA, long long lenQ, cudaStream_t stream) : maxA(lenA), maxQ(lenQ), bdim(LD), stream(stream) {
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

  DT** hostB;
  cudaMallocHost(reinterpret_cast<void**>(&hostB), psize * lenQ);
  for (long long i = 0; i < lenQ; i++)
    hostB[i] = &Bdata[i * LD * LD];

  cudaMemcpy(B, hostB, psize * lenQ, cudaMemcpyHostToDevice);
  cudaFreeHost(hostB);
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
}

template <typename DT> template <typename OT>
void H2Factorize<DT>::setData(long long rank, long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], const MatrixDataContainer<OT>& A, const MatrixDataContainer<OT>& Q) {
  long long block = bdim * bdim;
  lenD = M;
  lenA = std::min(maxA, A.nblocks());
  long long lenQ = std::min(maxQ, Q.nblocks());
  H2Factorize::rank = rank;

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

// needs explicit specialization due to cuBLAS calls
template <>
void H2Factorize<cuDoubleComplex>::compute() {
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
void H2Factorize<cuComplex>::compute() {
  long long N = bdim, S = rank, R = N - S;
  long long D = lenD;
  cuComplex one = make_cuComplex(1., 0.), zero = make_cuComplex(0., 0.), minus_one = make_cuComplex(-1., 0.);
  int info_host = 0;

  cublasCgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, U, N, A_SS, N, &zero, B, N, D);
  cublasCgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, U, N, B, N, &zero, A_SS, N, D);

  cublasCgetrfBatched(cublasH, R, A_RR, N, ipiv, info, D);
  cublasCgetrsBatched(cublasH, CUBLAS_OP_N, R, S, A_RR, N, ipiv, A_RS, N, &info_host, D);
  cublasCgetrsBatched(cublasH, CUBLAS_OP_N, R, N, A_RR, N, ipiv, V_R, N, &info_host, D);

  cublasCgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, S, R, &minus_one, A_SR, N, A_RS, N, &one, A_SS, N, D);

  for (int64_t i = D; i < lenA; i += maxQ) {
    int64_t len = std::min(lenA - i, maxQ);
    cublasCgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, &U[i], N, &A_SS[i], N, &zero, B, N, len);
    cublasCgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, &V[i], N, B, N, &zero, &A_SS[i], N, len);
  }

  cudaStreamSynchronize(stream);
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
void H2Factorize<float>::compute() {
  long long N = bdim, S = rank, R = N - S;
  long long D = lenD;
  float one = 1, zero = 0, minus_one = -1;
  int info_host = 0;

  //const auto COMP = CUBLAS_COMPUTE_32F_FAST_16F;
  const auto COMP = CUBLAS_COMPUTE_32F;
  const auto ALGO = CUBLAS_GEMM_DEFAULT;

  // gemm handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch count
  // gemmEx handle, transA, transB, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, batch count, computeType, algo
  //cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, U, N, A_SS, N, &zero, B, N, D);
  cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(U), CUDA_R_32F, N, reinterpret_cast<void**>(A_SS), CUDA_R_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(B), CUDA_R_32F, N, D, COMP, ALGO);
  //cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, U, N, A_SS, N, &zero, B, N, D);
  cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(U), CUDA_R_32F, N, reinterpret_cast<void**>(B), CUDA_R_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(A_SS), CUDA_R_32F, N, D, COMP, ALGO);

  cublasSgetrfBatched(cublasH, R, A_RR, N, ipiv, info, D);
  cublasSgetrsBatched(cublasH, CUBLAS_OP_N, R, S, A_RR, N, ipiv, A_RS, N, &info_host, D);
  cublasSgetrsBatched(cublasH, CUBLAS_OP_N, R, N, A_RR, N, ipiv, V_R, N, &info_host, D);

  //cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, S, R, &minus_one, A_SR, N, A_RS, N, &one, A_SS, N, D);
  cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, S, R, reinterpret_cast<void*>(&minus_one), reinterpret_cast<void**>(A_SR), CUDA_R_32F, N, reinterpret_cast<void**>(A_RS), CUDA_R_32F, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(A_SS), CUDA_R_32F, N, D, COMP, ALGO);

  for (int64_t i = D; i < lenA; i += maxQ) {
    int64_t len = std::min(lenA - i, maxQ);
    //cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, &U[i], N, &A_SS[i], N, &zero, B, N, len);
    cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(&U[i]), CUDA_R_32F, N, reinterpret_cast<void**>(&A_SS[i]), CUDA_R_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(B), CUDA_R_32F, N, len, COMP, ALGO);
    //cublasSgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, &V[i], N, B, N, &zero, &A_SS[i], N, len);
    cublasGemmBatchedEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, reinterpret_cast<void*>(&one), reinterpret_cast<void**>(&V[i]), CUDA_R_32F, N, reinterpret_cast<void**>(B), CUDA_R_32F, N, reinterpret_cast<void*>(&zero), reinterpret_cast<void**>(&A_SS[i]), CUDA_R_32F, N, len, COMP, ALGO);
  }

  cudaStreamSynchronize(stream);
}

template <typename DT> template <typename OT>
void H2Factorize<DT>::getResults(long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], MatrixDataContainer<OT>& A, int* ipvts) {
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

/* explicit template instantiation */
// complex double
template class H2Factorize<cuDoubleComplex>;
template void H2Factorize<cuDoubleComplex>::setData(long long, long long, long long, const long long[], const long long [], const long long[], const MatrixDataContainer<std::complex<double>>&, const MatrixDataContainer<std::complex<double>>&);
template void H2Factorize<cuDoubleComplex>::getResults(long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], MatrixDataContainer<std::complex<double>>& A, int* ipvts);
// complex float
template class H2Factorize<cuComplex>;
template void H2Factorize<cuComplex>::setData(long long, long long, long long, const long long[], const long long [], const long long[], const MatrixDataContainer<std::complex<float>>&, const MatrixDataContainer<std::complex<float>>&);
template void H2Factorize<cuComplex>::getResults(long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], MatrixDataContainer<std::complex<float>>& A, int* ipvts);
// double
template class H2Factorize<double>;
template void H2Factorize<double>::setData(long long, long long, long long, const long long[], const long long [], const long long[], const MatrixDataContainer<double>&, const MatrixDataContainer<double>&);
template void H2Factorize<double>::getResults(long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], MatrixDataContainer<double>& A, int* ipvts);
// float
template class H2Factorize<float>;
template void H2Factorize<float>::setData(long long, long long, long long, const long long[], const long long [], const long long[], const MatrixDataContainer<float>&, const MatrixDataContainer<float>&);
template void H2Factorize<float>::getResults(long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], MatrixDataContainer<float>& A, int* ipvts);