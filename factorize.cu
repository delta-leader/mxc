
#include <factorize.cuh>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <mkl.h>

H2Factorize::H2Factorize(long long LD, long long lenA, long long lenQ, cudaStream_t stream) : maxA(lenA), maxQ(lenQ), bdim(LD), stream(stream) {
  cublasH = nullptr;
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);

  long long bsize = LD * LD * sizeof(cuDoubleComplex);
  cudaMalloc(reinterpret_cast<void**>(&Adata), bsize * lenA);
  cudaMalloc(reinterpret_cast<void**>(&Bdata), bsize * lenQ);
  cudaMalloc(reinterpret_cast<void**>(&Udata), bsize * lenQ);
  cudaMalloc(reinterpret_cast<void**>(&Vdata), bsize * lenQ);

  long long psize = sizeof(cuDoubleComplex*);
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

  cuDoubleComplex** hostB;
  cudaMallocHost(reinterpret_cast<void**>(&hostB), psize * lenQ);
  for (long long i = 0; i < lenQ; i++)
    hostB[i] = &Bdata[i * LD * LD];

  cudaMemcpy(B, hostB, psize * lenQ, cudaMemcpyHostToDevice);
  cudaFreeHost(hostB);
}

H2Factorize::~H2Factorize() {
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

void H2Factorize::setData(long long rank, long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], const MatrixDataContainer<std::complex<double>>& A, const MatrixDataContainer<std::complex<double>>& Q) {
  long long block = bdim * bdim;
  lenD = M;
  lenA = std::min(maxA, A.nblocks());
  long long lenQ = std::min(maxQ, Q.nblocks());
  H2Factorize::rank = rank;

  cuDoubleComplex* hostU, *hostA;
  cudaMallocHost(reinterpret_cast<void**>(&hostU), block * lenQ * sizeof(cuDoubleComplex));
  std::fill(hostU, &hostU[block * lenQ], make_cuDoubleComplex(0., 0.));

  for (long long i = 0; i < lenQ; i++) {
    long long m = Dims[i];
    MKL_Zomatcopy('C', 'C', m, m, std::complex<double>(1., 0.), Q[i], m, reinterpret_cast<std::complex<double>*>(&hostU[i * block]), bdim);
  }
  cudaMemcpy(Udata, hostU, block * lenQ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  cudaMemcpy(Vdata, Udata, block * lenQ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
  
  cudaFreeHost(hostU);
  cudaMallocHost(reinterpret_cast<void**>(&hostA), block * lenA * sizeof(cuDoubleComplex));
  std::fill(hostA, &hostA[block * lenA], make_cuDoubleComplex(0., 0.));

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
    MKL_Zomatcopy('C', 'N', M, N, std::complex<double>(1., 0.), A[i], M, reinterpret_cast<std::complex<double>*>(&hostA[i * block]), bdim);
  }

  cudaMemcpy(Adata, hostA, block * lenA * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  cudaFreeHost(hostA);

  std::stable_partition(coo_list.begin(), coo_list.end(), 
    [](std::tuple<int64_t, int64_t, int64_t> i) { return std::get<0>(i) == std::get<1>(i); });

  cuDoubleComplex** hostAptrs, **hostUptrs, **hostVptrs;
  cudaMallocHost(reinterpret_cast<void**>(&hostAptrs), lenA * sizeof(cuDoubleComplex*));
  cudaMallocHost(reinterpret_cast<void**>(&hostUptrs), lenA * sizeof(cuDoubleComplex*));
  cudaMallocHost(reinterpret_cast<void**>(&hostVptrs), lenA * sizeof(cuDoubleComplex*));

  std::transform(coo_list.begin(), coo_list.end(), hostAptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Adata + std::get<2>(i) * block; });
  std::transform(coo_list.begin(), coo_list.end(), hostUptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Udata + std::get<1>(i) * block; });
  std::transform(coo_list.begin(), coo_list.end(), hostVptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Vdata + std::get<0>(i) * block; });

  cudaMemcpy(A_SS, hostAptrs, lenA * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
  cudaMemcpy(U, hostUptrs, lenA * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
  cudaMemcpy(V, hostVptrs, lenA * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);

  long long offset_SR = rank * bdim, offset_RS = rank, offset_RR = rank * (bdim + 1);
  std::transform(coo_list.begin(), coo_list.end(), hostAptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Adata + std::get<2>(i) * block + offset_SR; });
  std::transform(coo_list.begin(), coo_list.end(), hostUptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Adata + std::get<2>(i) * block + offset_RS; });
  std::transform(coo_list.begin(), coo_list.end(), hostVptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Adata + std::get<2>(i) * block + offset_RR; });

  cudaMemcpy(A_SR, hostAptrs, lenA * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
  cudaMemcpy(A_RS, hostUptrs, lenA * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
  cudaMemcpy(A_RR, hostVptrs, lenA * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);

  std::transform(coo_list.begin(), coo_list.begin() + M, hostVptrs, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Vdata + std::get<0>(i) * block + offset_RS; });
  cudaMemcpy(V_R, hostVptrs, M * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);

  cudaFreeHost(hostAptrs);
  cudaFreeHost(hostUptrs);
  cudaFreeHost(hostVptrs);
}

void H2Factorize::compute() {
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

void H2Factorize::getResults(long long D, long long M, const long long ARows[], const long long ACols[], const long long Dims[], MatrixDataContainer<std::complex<double>>& A, int* ipvts) {
  long long block = bdim * bdim;
  long long lenR = bdim - rank;

  cuDoubleComplex* hostA;
  int* hostI;
  cudaMallocHost(reinterpret_cast<void**>(&hostA), block * lenA * sizeof(cuDoubleComplex));
  cudaMallocHost(reinterpret_cast<void**>(&hostI), lenR * M * sizeof(int));
  cudaMemcpy(hostA, Adata, block * lenA * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
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
      MKL_Zomatcopy('C', 'N', m, n, std::complex<double>(1., 0.), reinterpret_cast<std::complex<double>*>(&hostA[ij * block]), bdim, A[ij], m);
    }
  }

  cudaFreeHost(hostA);
  cudaFreeHost(hostI);
}
