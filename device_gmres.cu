
#include <device_csr_matrix.cuh>
#include <device_factorize.cuh>
#include <comm-mpi.hpp>

#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/inner_product.h>

#include <iostream>

struct conjugateFunc {
  __host__ __device__ thrust::complex<float> operator()(const thrust::complex<float>& z) const {
    return thrust::conj(z);
  }
  __host__ __device__ thrust::complex<double> operator()(const thrust::complex<double>& z) const {
    return thrust::conj(z);
  }
};

void levelCommSum(long long N, thrust::complex<double> X[], ncclComm_t AllComm, ncclComm_t DupComm, cudaStream_t stream) {
  if (AllComm)
    ncclAllReduce(const_cast<const thrust::complex<double>*>(X), X, N * 2, ncclDouble, ncclSum, AllComm, stream);
  if (DupComm)
    ncclBroadcast(const_cast<const thrust::complex<double>*>(X), X, N * 2, ncclDouble, 0, DupComm, stream);
}

void levelCommSum(long long N, thrust::complex<float> X[], ncclComm_t AllComm, ncclComm_t DupComm, cudaStream_t stream) {
  if (AllComm)
    ncclAllReduce(const_cast<const thrust::complex<float>*>(X), X, N * 2, ncclFloat, ncclSum, AllComm, stream);
  if (DupComm)
    ncclBroadcast(const_cast<const thrust::complex<float>*>(X), X, N * 2, ncclFloat, 0, DupComm, stream);
}

template <>
long long solveDeviceGMRES(deviceHandle_t handle, long long levels, CsrMatVecDesc_t<std::complex<double>> desc[], long long mlevels, deviceMatrixDesc_t<std::complex<double>> desc_m[], double tol, std::complex<double>* X, const std::complex<double>* B, long long inner_iters, long long outer_iters, double resid[], const ColCommMPI& comm, const ncclComms nccl_comms) {
  long long N = desc[levels]->X->lenX;
  long long ld = inner_iters + 1;

  thrust::device_vector<thrust::complex<double>> devB(B, &B[N]);
  thrust::device_vector<thrust::complex<double>> devR(devB.begin(), devB.end());
  thrust::device_vector<thrust::complex<double>> devX(N, thrust::complex<double>(0., 0.));
  thrust::device_vector<thrust::complex<double>> H(ld * inner_iters);
  thrust::device_vector<thrust::complex<double>> v(N * ld);
  thrust::device_vector<thrust::complex<double>> s(ld);
  thrust::device_vector<thrust::complex<double>*> ptr({thrust::raw_pointer_cast(H.data()), thrust::raw_pointer_cast(s.data())});

  cuDoubleComplex* Bdata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(devB.data()));
  cuDoubleComplex* Rdata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(devR.data()));
  cuDoubleComplex* Xdata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(devX.data()));
  cuDoubleComplex* Hdata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(H.data()));
  cuDoubleComplex* Vdata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(v.data()));
  cuDoubleComplex* Sdata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(s.data()));
  cuDoubleComplex** Pdata = reinterpret_cast<cuDoubleComplex**>(thrust::raw_pointer_cast(ptr.data()));

  ncclComm_t AllComm = findNcclComm(comm.AllReduceComm, nccl_comms);
  ncclComm_t DupComm = findNcclComm(comm.DupComm, nccl_comms);
  cudaStream_t stream = handle->compute_stream;
  cublasHandle_t cublasH = handle->cublasH;
  int* dev_info;
  cudaMalloc(reinterpret_cast<void**>(&dev_info), sizeof(int));

  auto conjR = thrust::make_transform_iterator(devR.begin(), conjugateFunc());
  thrust::complex<double> nsum = thrust::inner_product(thrust::cuda::par.on(stream), conjR, conjR + N, devR.begin(), 
    thrust::complex<double>(0., 0.), thrust::plus<thrust::complex<double>>(), thrust::multiplies<thrust::complex<double>>());
  comm.level_sum(reinterpret_cast<std::complex<double>*>(&nsum), 1);

  double normb = std::sqrt(nsum.real());
  if (normb == 0.)
    normb = 1.;
  resid[0] = 1.;
  long long iters = 0;

  thrust::complex<double> one(1., 0.), zero(0., 0.), minus_one(-1., 0.);

  while (iters < outer_iters && tol <= resid[iters]) {
    matSolvePreconditionDeviceH2(handle, mlevels, desc_m, reinterpret_cast<std::complex<double>*>(Rdata));
    nsum = thrust::inner_product(thrust::cuda::par.on(stream), conjR, conjR + N, devR.begin(), 
      thrust::complex<double>(0., 0.), thrust::plus<thrust::complex<double>>(), thrust::multiplies<thrust::complex<double>>());
    comm.level_sum(reinterpret_cast<std::complex<double>*>(&nsum), 1);

    double beta = std::sqrt(nsum.real());
    thrust::complex<double> inv_beta(1. / beta, 0.);
    thrust::fill(H.begin(), H.end(), thrust::complex<double>(0., 0.));
    thrust::fill(v.begin(), v.end(), thrust::complex<double>(0., 0.));
    thrust::fill(s.begin() + 1, s.end(), zero);
    s[0] = beta;
    cublasZaxpy(cublasH, N, reinterpret_cast<cuDoubleComplex*>(&inv_beta), Rdata, 1, Vdata, 1);
    
    for (long long i = 0; i < inner_iters; i++) {
      cublasZcopy(cublasH, N, &Vdata[i * N], 1, Rdata, 1);
      matVecDeviceH2(handle, levels, desc, reinterpret_cast<std::complex<double>*>(Rdata));
      matSolvePreconditionDeviceH2(handle, mlevels, desc_m, reinterpret_cast<std::complex<double>*>(Rdata));

      cublasZgemv(cublasH, CUBLAS_OP_C, N, i + 1, reinterpret_cast<cuDoubleComplex*>(&one), Vdata, N, Rdata, 1, reinterpret_cast<cuDoubleComplex*>(&zero), &Hdata[i * ld], 1);
      levelCommSum(i + 1, thrust::raw_pointer_cast(&H[i * ld]), AllComm, DupComm, stream);
      cublasZgemv(cublasH, CUBLAS_OP_N, N, i + 1, reinterpret_cast<cuDoubleComplex*>(&minus_one), Vdata, N, &Hdata[i * ld], 1, reinterpret_cast<cuDoubleComplex*>(&one), Rdata, 1);

      nsum = thrust::inner_product(thrust::cuda::par.on(stream), conjR, conjR + N, devR.begin(), 
        thrust::complex<double>(0., 0.), thrust::plus<thrust::complex<double>>(), thrust::multiplies<thrust::complex<double>>());
      comm.level_sum(reinterpret_cast<std::complex<double>*>(&nsum), 1);

      H[i * (ld + 1) + 1] = std::sqrt(nsum.real());
      thrust::complex<double> inv_beta(1. / std::sqrt(nsum.real()), 0.);
      cublasZaxpy(cublasH, N, reinterpret_cast<cuDoubleComplex*>(&inv_beta), Rdata, 1, &Vdata[N * (i + 1)], 1);
    }
    
    int info;
    cublasZgelsBatched(cublasH, CUBLAS_OP_N, ld, inner_iters, 1, Pdata, ld, &Pdata[1], ld, &info, dev_info, 1);
    cublasZgemv(cublasH, CUBLAS_OP_N, N, inner_iters, reinterpret_cast<cuDoubleComplex*>(&one), Vdata, N, Sdata, 1, reinterpret_cast<cuDoubleComplex*>(&one), Xdata, 1);

    cublasZgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, 1, reinterpret_cast<cuDoubleComplex*>(&minus_one), Xdata, N, reinterpret_cast<cuDoubleComplex*>(&zero), Rdata, N, Rdata, N);
    matVecDeviceH2(handle, levels, desc, reinterpret_cast<std::complex<double>*>(Rdata));
    cublasZaxpy(cublasH, N, reinterpret_cast<cuDoubleComplex*>(&one), Bdata, 1, Rdata, 1);

    nsum = thrust::inner_product(thrust::cuda::par.on(stream), conjR, conjR + N, devR.begin(), 
      thrust::complex<double>(0., 0.), thrust::plus<thrust::complex<double>>(), thrust::multiplies<thrust::complex<double>>());
    comm.level_sum(reinterpret_cast<std::complex<double>*>(&nsum), 1);
    resid[++iters] = std::sqrt(nsum.real()) / normb;
  }

  thrust::copy(devX.begin(), devX.end(), X);
  cudaFree(dev_info);
  return iters;
}

template <>
long long solveDeviceGMRES(deviceHandle_t handle, long long levels, CsrMatVecDesc_t<std::complex<double>> desc[], long long mlevels, deviceMatrixDesc_t<std::complex<float>> desc_m[], double tol, std::complex<double>* X, const std::complex<double>* B, long long inner_iters, long long outer_iters, double resid[], const ColCommMPI& comm, const ncclComms nccl_comms) {
  long long N = desc[levels]->X->lenX;
  long long ld = inner_iters + 1;

  thrust::device_vector<thrust::complex<double>> devB(B, &B[N]);
  thrust::device_vector<thrust::complex<double>> devR(devB.begin(), devB.end());
  thrust::device_vector<thrust::complex<float>> devR_low(N);
  thrust::device_vector<thrust::complex<double>> devX(N, thrust::complex<double>(0., 0.));
  thrust::device_vector<thrust::complex<double>> H(ld * inner_iters);
  thrust::device_vector<thrust::complex<double>> v(N * ld);
  thrust::device_vector<thrust::complex<double>> s(ld);
  thrust::device_vector<thrust::complex<double>*> ptr({thrust::raw_pointer_cast(H.data()), thrust::raw_pointer_cast(s.data())});

  cuDoubleComplex* Bdata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(devB.data()));
  cuDoubleComplex* Rdata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(devR.data()));
  cuComplex* Rdata_low = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(devR_low.data()));
  cuDoubleComplex* Xdata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(devX.data()));
  cuDoubleComplex* Hdata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(H.data()));
  cuDoubleComplex* Vdata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(v.data()));
  cuDoubleComplex* Sdata = reinterpret_cast<cuDoubleComplex*>(thrust::raw_pointer_cast(s.data()));
  cuDoubleComplex** Pdata = reinterpret_cast<cuDoubleComplex**>(thrust::raw_pointer_cast(ptr.data()));

  ncclComm_t AllComm = findNcclComm(comm.AllReduceComm, nccl_comms);
  ncclComm_t DupComm = findNcclComm(comm.DupComm, nccl_comms);
  cudaStream_t stream = handle->compute_stream;
  cublasHandle_t cublasH = handle->cublasH;
  int* dev_info;
  cudaMalloc(reinterpret_cast<void**>(&dev_info), sizeof(int));

  auto conjR = thrust::make_transform_iterator(devR.begin(), conjugateFunc());
  thrust::complex<double> nsum = thrust::inner_product(thrust::cuda::par.on(stream), conjR, conjR + N, devR.begin(), 
    thrust::complex<double>(0., 0.), thrust::plus<thrust::complex<double>>(), thrust::multiplies<thrust::complex<double>>());
  comm.level_sum(reinterpret_cast<std::complex<double>*>(&nsum), 1);

  double normb = std::sqrt(nsum.real());
  if (normb == 0.)
    normb = 1.;
  resid[0] = 1.;
  long long iters = 0;

  thrust::complex<double> one(1., 0.), zero(0., 0.), minus_one(-1., 0.);

  while (iters < outer_iters && tol <= resid[iters]) {
    devR_low = devR;
    matSolvePreconditionDeviceH2(handle, mlevels, desc_m, reinterpret_cast<std::complex<float>*>(Rdata_low));
    devR = devR_low;
    nsum = thrust::inner_product(thrust::cuda::par.on(stream), conjR, conjR + N, devR.begin(), 
      thrust::complex<double>(0., 0.), thrust::plus<thrust::complex<double>>(), thrust::multiplies<thrust::complex<double>>());
    comm.level_sum(reinterpret_cast<std::complex<double>*>(&nsum), 1);

    double beta = std::sqrt(nsum.real());
    thrust::complex<double> inv_beta(1. / beta, 0.);
    thrust::fill(H.begin(), H.end(), thrust::complex<double>(0., 0.));
    thrust::fill(v.begin(), v.end(), thrust::complex<double>(0., 0.));
    thrust::fill(s.begin() + 1, s.end(), zero);
    s[0] = beta;
    cublasZaxpy(cublasH, N, reinterpret_cast<cuDoubleComplex*>(&inv_beta), Rdata, 1, Vdata, 1);
    
    for (long long i = 0; i < inner_iters; i++) {
      cublasZcopy(cublasH, N, &Vdata[i * N], 1, Rdata, 1);
      matVecDeviceH2(handle, levels, desc, reinterpret_cast<std::complex<double>*>(Rdata));
      devR_low = devR;
      matSolvePreconditionDeviceH2(handle, mlevels, desc_m, reinterpret_cast<std::complex<float>*>(Rdata_low));
      devR = devR_low;

      cublasZgemv(cublasH, CUBLAS_OP_C, N, i + 1, reinterpret_cast<cuDoubleComplex*>(&one), Vdata, N, Rdata, 1, reinterpret_cast<cuDoubleComplex*>(&zero), &Hdata[i * ld], 1);
      levelCommSum(i + 1, thrust::raw_pointer_cast(&H[i * ld]), AllComm, DupComm, stream);
      cublasZgemv(cublasH, CUBLAS_OP_N, N, i + 1, reinterpret_cast<cuDoubleComplex*>(&minus_one), Vdata, N, &Hdata[i * ld], 1, reinterpret_cast<cuDoubleComplex*>(&one), Rdata, 1);

      nsum = thrust::inner_product(thrust::cuda::par.on(stream), conjR, conjR + N, devR.begin(), 
        thrust::complex<double>(0., 0.), thrust::plus<thrust::complex<double>>(), thrust::multiplies<thrust::complex<double>>());
      comm.level_sum(reinterpret_cast<std::complex<double>*>(&nsum), 1);

      H[i * (ld + 1) + 1] = std::sqrt(nsum.real());
      thrust::complex<double> inv_beta(1. / std::sqrt(nsum.real()), 0.);
      cublasZaxpy(cublasH, N, reinterpret_cast<cuDoubleComplex*>(&inv_beta), Rdata, 1, &Vdata[N * (i + 1)], 1);
    }
    
    int info;
    cublasZgelsBatched(cublasH, CUBLAS_OP_N, ld, inner_iters, 1, Pdata, ld, &Pdata[1], ld, &info, dev_info, 1);
    cublasZgemv(cublasH, CUBLAS_OP_N, N, inner_iters, reinterpret_cast<cuDoubleComplex*>(&one), Vdata, N, Sdata, 1, reinterpret_cast<cuDoubleComplex*>(&one), Xdata, 1);

    cublasZgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, 1, reinterpret_cast<cuDoubleComplex*>(&minus_one), Xdata, N, reinterpret_cast<cuDoubleComplex*>(&zero), Rdata, N, Rdata, N);
    matVecDeviceH2(handle, levels, desc, reinterpret_cast<std::complex<double>*>(Rdata));
    cublasZaxpy(cublasH, N, reinterpret_cast<cuDoubleComplex*>(&one), Bdata, 1, Rdata, 1);

    nsum = thrust::inner_product(thrust::cuda::par.on(stream), conjR, conjR + N, devR.begin(), 
      thrust::complex<double>(0., 0.), thrust::plus<thrust::complex<double>>(), thrust::multiplies<thrust::complex<double>>());
    comm.level_sum(reinterpret_cast<std::complex<double>*>(&nsum), 1);
    resid[++iters] = std::sqrt(nsum.real()) / normb;
  }

  thrust::copy(devX.begin(), devX.end(), X);
  cudaFree(dev_info);
  return iters;
}

template <>
long long solveDeviceGMRES(deviceHandle_t handle, long long levels, CsrMatVecDesc_t<std::complex<float>> desc[], long long mlevels, deviceMatrixDesc_t<std::complex<float>> desc_m[], double tol, std::complex<float>* X, const std::complex<float>* B, long long inner_iters, long long outer_iters, double resid[], const ColCommMPI& comm, const ncclComms nccl_comms) {
    long long N = desc[levels]->X->lenX;
  long long ld = inner_iters + 1;

  thrust::device_vector<thrust::complex<float>> devB(B, &B[N]);
  thrust::device_vector<thrust::complex<float>> devR(devB.begin(), devB.end());
  thrust::device_vector<thrust::complex<float>> devX(N, thrust::complex<float>(0., 0.));
  thrust::device_vector<thrust::complex<float>> H(ld * inner_iters);
  thrust::device_vector<thrust::complex<float>> v(N * ld);
  thrust::device_vector<thrust::complex<float>> s(ld);
  thrust::device_vector<thrust::complex<float>*> ptr({thrust::raw_pointer_cast(H.data()), thrust::raw_pointer_cast(s.data())});

  cuComplex* Bdata = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(devB.data()));
  cuComplex* Rdata = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(devR.data()));
  cuComplex* Xdata = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(devX.data()));
  cuComplex* Hdata = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(H.data()));
  cuComplex* Vdata = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(v.data()));
  cuComplex* Sdata = reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(s.data()));
  cuComplex** Pdata = reinterpret_cast<cuComplex**>(thrust::raw_pointer_cast(ptr.data()));

  ncclComm_t AllComm = findNcclComm(comm.AllReduceComm, nccl_comms);
  ncclComm_t DupComm = findNcclComm(comm.DupComm, nccl_comms);
  cudaStream_t stream = handle->compute_stream;
  cublasHandle_t cublasH = handle->cublasH;
  int* dev_info;
  cudaMalloc(reinterpret_cast<void**>(&dev_info), sizeof(int));

  auto conjR = thrust::make_transform_iterator(devR.begin(), conjugateFunc());
  thrust::complex<float> nsum = thrust::inner_product(thrust::cuda::par.on(stream), conjR, conjR + N, devR.begin(), 
    thrust::complex<float>(0., 0.), thrust::plus<thrust::complex<float>>(), thrust::multiplies<thrust::complex<float>>());
  comm.level_sum(reinterpret_cast<std::complex<float>*>(&nsum), 1);

  float normb = std::sqrt(nsum.real());
  if (normb == 0.)
    normb = 1.;
  resid[0] = 1.;
  long long iters = 0;

  thrust::complex<float> one(1., 0.), zero(0., 0.), minus_one(-1., 0.);

  while (iters < outer_iters && tol <= resid[iters]) {
    matSolvePreconditionDeviceH2(handle, mlevels, desc_m, reinterpret_cast<std::complex<float>*>(Rdata));
    nsum = thrust::inner_product(thrust::cuda::par.on(stream), conjR, conjR + N, devR.begin(), 
      thrust::complex<float>(0., 0.), thrust::plus<thrust::complex<float>>(), thrust::multiplies<thrust::complex<float>>());
    comm.level_sum(reinterpret_cast<std::complex<float>*>(&nsum), 1);

    float beta = std::sqrt(nsum.real());
    thrust::complex<float> inv_beta(1. / beta, 0.);
    thrust::fill(H.begin(), H.end(), thrust::complex<float>(0., 0.));
    thrust::fill(v.begin(), v.end(), thrust::complex<float>(0., 0.));
    thrust::fill(s.begin() + 1, s.end(), zero);
    s[0] = beta;
    cublasCaxpy(cublasH, N, reinterpret_cast<cuComplex*>(&inv_beta), Rdata, 1, Vdata, 1);
    
    for (long long i = 0; i < inner_iters; i++) {
      cublasCcopy(cublasH, N, &Vdata[i * N], 1, Rdata, 1);
      matVecDeviceH2(handle, levels, desc, reinterpret_cast<std::complex<float>*>(Rdata));
      matSolvePreconditionDeviceH2(handle, mlevels, desc_m, reinterpret_cast<std::complex<float>*>(Rdata));

      cublasCgemv(cublasH, CUBLAS_OP_C, N, i + 1, reinterpret_cast<cuComplex*>(&one), Vdata, N, Rdata, 1, reinterpret_cast<cuComplex*>(&zero), &Hdata[i * ld], 1);
      levelCommSum(i + 1, thrust::raw_pointer_cast(&H[i * ld]), AllComm, DupComm, stream);
      cublasCgemv(cublasH, CUBLAS_OP_N, N, i + 1, reinterpret_cast<cuComplex*>(&minus_one), Vdata, N, &Hdata[i * ld], 1, reinterpret_cast<cuComplex*>(&one), Rdata, 1);

      nsum = thrust::inner_product(thrust::cuda::par.on(stream), conjR, conjR + N, devR.begin(), 
        thrust::complex<float>(0., 0.), thrust::plus<thrust::complex<float>>(), thrust::multiplies<thrust::complex<float>>());
      comm.level_sum(reinterpret_cast<std::complex<float>*>(&nsum), 1);

      H[i * (ld + 1) + 1] = std::sqrt(nsum.real());
      thrust::complex<float> inv_beta(1. / std::sqrt(nsum.real()), 0.);
      cublasCaxpy(cublasH, N, reinterpret_cast<cuComplex*>(&inv_beta), Rdata, 1, &Vdata[N * (i + 1)], 1);
    }
    
    int info;
    cublasCgelsBatched(cublasH, CUBLAS_OP_N, ld, inner_iters, 1, Pdata, ld, &Pdata[1], ld, &info, dev_info, 1);
    cublasCgemv(cublasH, CUBLAS_OP_N, N, inner_iters, reinterpret_cast<cuComplex*>(&one), Vdata, N, Sdata, 1, reinterpret_cast<cuComplex*>(&one), Xdata, 1);

    cublasCgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, 1, reinterpret_cast<cuComplex*>(&minus_one), Xdata, N, reinterpret_cast<cuComplex*>(&zero), Rdata, N, Rdata, N);
    matVecDeviceH2(handle, levels, desc, reinterpret_cast<std::complex<float>*>(Rdata));
    cublasCaxpy(cublasH, N, reinterpret_cast<cuComplex*>(&one), Bdata, 1, Rdata, 1);

    nsum = thrust::inner_product(thrust::cuda::par.on(stream), conjR, conjR + N, devR.begin(), 
      thrust::complex<float>(0., 0.), thrust::plus<thrust::complex<float>>(), thrust::multiplies<thrust::complex<float>>());
    comm.level_sum(reinterpret_cast<std::complex<float>*>(&nsum), 1);
    resid[++iters] = std::sqrt(nsum.real()) / normb;
  }

  thrust::copy(devX.begin(), devX.end(), X);
  cudaFree(dev_info);
  return iters;
}

template <>
long long solveDeviceGMRES(deviceHandle_t handle, long long levels, CsrMatVecDesc_t<double> desc[], long long mlevels, deviceMatrixDesc_t<double> desc_m[], double tol, double* X, const double* B, long long inner_iters, long long outer_iters, double resid[], const ColCommMPI& comm, const ncclComms nccl_comms) {
  std::cout<<"Not implemented"<<std::endl;
}

template <>
long long solveDeviceGMRES(deviceHandle_t handle, long long levels, CsrMatVecDesc_t<float> desc[], long long mlevels, deviceMatrixDesc_t<float> desc_m[], double tol, float* X, const float* B, long long inner_iters, long long outer_iters, double resid[], const ColCommMPI& comm, const ncclComms nccl_comms) {
  std::cout<<"Not implemented"<<std::endl;
}
