
#pragma once

#include <cstdint>
#include <complex>

class Eval;

void gen_matrix(const Eval& eval, int64_t m, int64_t n, const double* bi, const double* bj, std::complex<double> Aij[], int64_t lda);

void compute_AallT(const Eval& eval, int64_t M, const double Xbodies[], int64_t Lfar, const int64_t Nfar[], const double* Fbodies[], std::complex<double> Aall[], int64_t LDA);

int64_t compute_basis(double epi, int64_t M, std::complex<double> A[], int64_t LDA, std::complex<double> R[], int64_t LDR, double Xbodies[]);

void mat_vec_reference(const Eval& eval, int64_t M, int64_t N, int64_t nrhs, std::complex<double> B[], int64_t ldB, const std::complex<double> X[], int64_t ldX, const double ibodies[], const double jbodies[]);

