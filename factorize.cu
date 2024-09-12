
#include <factorize.cuh>
#include <algorithm>
#include <numeric>
#include <tuple>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/complex.h>

H2Factorize::H2Factorize(long long LD, long long lenA, long long lenQ, cudaStream_t stream) : maxA(lenA), maxQ(lenQ), stream(stream) {
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

  long long len = std::max(lenA, lenQ);
  cudaMallocHost(reinterpret_cast<void**>(&hostA), bsize * len);
  cudaMallocHost(reinterpret_cast<void**>(&hostP), psize * len);

  for (long long i = 0; i < lenQ; i++)
    hostP[i] = &Bdata[i * LD * LD];

  cudaMemcpy(B, hostP, psize * lenQ, cudaMemcpyHostToDevice);
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
  
  cudaFreeHost(hostA);
  cudaFreeHost(hostP);
}

void H2Factorize::setData(long long bdim, long long rank, long long D, long long M, const long long ARows[], const long long ACols[], const MatrixDataContainer<std::complex<double>>& A, const MatrixDataContainer<std::complex<double>>& Q) {
  long long block = bdim * bdim;
  lenD = M;
  lenA = std::min(maxA, A.nblocks());
  offsetD = D;
  long long lenQ = std::min(maxQ, Q.nblocks());
  H2Factorize::bdim = bdim;
  H2Factorize::rank = rank;

  std::copy(Q[0], Q[lenQ], reinterpret_cast<std::complex<double>*>(hostA));
  cudaMemcpy(Udata, hostA, block * lenQ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  std::copy(A[0], A[lenA], reinterpret_cast<std::complex<double>*>(hostA));
  cudaMemcpy(Adata, hostA, block * lenA * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  std::vector<std::tuple<long long, long long, long long>> coo_list(lenA);
  for (long long y = 0; y < M; y++) {
    long long begin = ARows[y];
    long long end = ARows[y + 1];
    std::transform(&ACols[begin], &ACols[end], coo_list.begin() + begin, 
      [&](const long long& x) { return std::make_tuple(y, x, std::distance(ACols, &x)); });
  }

  std::stable_partition(coo_list.begin(), coo_list.end(), 
    [=](std::tuple<int64_t, int64_t, int64_t> i) { return (std::get<0>(i) + D) == std::get<1>(i); });

  long long offset_SR = rank * bdim, offset_RS = rank, offset_RR = rank * (bdim + 1);

  std::transform(coo_list.begin(), coo_list.end(), hostP, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Adata + std::get<2>(i) * block; });
  cudaMemcpy(A_SS, hostP, lenA * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);

  std::transform(coo_list.begin(), coo_list.end(), hostP, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Udata + std::get<1>(i) * block; });
  cudaMemcpy(U, hostP, lenA * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);

  std::transform(coo_list.begin(), coo_list.end(), hostP, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Vdata + std::get<0>(i) * block; });
  cudaMemcpy(V, hostP, lenA * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);

  std::transform(coo_list.begin(), coo_list.end(), hostP, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Adata + std::get<2>(i) * block + offset_SR; });
  cudaMemcpy(A_SR, hostP, lenA * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);

  std::transform(coo_list.begin(), coo_list.end(), hostP, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Adata + std::get<2>(i) * block + offset_RS; });
  cudaMemcpy(A_RS, hostP, lenA * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);

  std::transform(coo_list.begin(), coo_list.end(), hostP, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Adata + std::get<2>(i) * block + offset_RR; });
  cudaMemcpy(A_RR, hostP, lenA * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);

  std::transform(coo_list.begin(), coo_list.begin() + M, hostP, [=](std::tuple<int64_t, int64_t, int64_t> i) { return Vdata + std::get<0>(i) * block + offset_RS; });
  cudaMemcpy(V_R, hostP, M * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
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

struct conjugateDouble {
  __host__ __device__ thrust::complex<double> operator()(const thrust::complex<double>& z) const {
    return thrust::conj(z);
  }
};

void H2Factorize::compute() {
  long long N = bdim, S = rank, R = N - S;
  long long D = lenD;
  cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.), minus_one = make_cuDoubleComplex(-1., 0.);
  int info_host = 0;

  long long ST = N * N, Dsize = ST * D;
  thrust::device_ptr<const thrust::complex<double>> u = thrust::device_ptr<const thrust::complex<double>>(reinterpret_cast<const thrust::complex<double>*>(&Udata[offsetD * ST]));
  thrust::device_ptr<thrust::complex<double>> v = thrust::device_ptr<thrust::complex<double>>(reinterpret_cast<thrust::complex<double>*>(Vdata));

  auto map = thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), swapXY(N, ST));
  thrust::gather(thrust::cuda::par.on(stream), map, map + Dsize, thrust::make_transform_iterator(u, conjugateDouble()), v);

  cublasZgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, N, N, N, &one, U, N, A_SS, N, &zero, B, N, D);
  cublasZgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, N, N, N, &one, U, N, B, N, &zero, A_SS, N, D);

  cublasZgetrfBatched(cublasH, R, A_RR, N, ipiv, info, D);
  cublasZgetrsBatched(cublasH, CUBLAS_OP_N, R, S, A_RR, N, ipiv, A_RS, N, &info_host, D);
  cublasZgetrsBatched(cublasH, CUBLAS_OP_N, R, N, A_RR, N, ipiv, V_R, N, &info_host, D);

  cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, S, R, &minus_one, A_SR, N, A_RS, N, &one, A_SS, N, D);

  for (int64_t i = D; i < lenA; i += maxQ) {
    int64_t len = std::min(lenA - i, maxQ);
    cublasZgemmBatched(cublasH, CUBLAS_OP_C, CUBLAS_OP_T, N, N, N, &one, &U[i], N, &A_SS[i], N, &zero, B, N, len);
    cublasZgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &one, &V[i], N, B, N, &zero, &A_SS[i], N, len);
  }
  cudaStreamSynchronize(stream);
}

void H2Factorize::getResults(long long D, long long M, MatrixDataContainer<std::complex<double>>& A, MatrixDataContainer<std::complex<double>>& Q) {
  long long block = bdim * bdim;

  cudaMemcpy(hostA, Adata, block * lenA * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  std::copy(reinterpret_cast<std::complex<double>*>(hostA), reinterpret_cast<std::complex<double>*>(&hostA[block * lenA]), A[0]);

  cudaMemcpy(hostA, Vdata, block * M * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  std::copy(reinterpret_cast<std::complex<double>*>(hostA), reinterpret_cast<std::complex<double>*>(&hostA[block * M]), Q[D]);
}
