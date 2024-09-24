#pragma once

#include <complex>
#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

void compute_factorize(cublasHandle_t cublasH, long long bdim, long long rank, long long D, long long M, long long N, const long long ARows[], const long long ACols[], std::complex<double>* A, std::complex<double>* R, const std::complex<double>* Q);
