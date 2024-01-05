
#pragma once

#include <cstdint>
#include <complex>

class Eval;

class Matrix {
public:
  std::complex<double>* A;
  int64_t M, N, LDA;
};

void gen_matrix(const Eval& eval, int64_t m, int64_t n, const double* bi, const double* bj, std::complex<double> Aij[], int64_t lda);

void mmult(char ta, char tb, const Matrix* A, const Matrix* B, Matrix* C, std::complex<double> alpha, std::complex<double> beta);

int64_t compute_basis(const Eval& eval, double epi, int64_t M, std::complex<double> A[], int64_t LDA, double Xbodies[], int64_t Lfar, int64_t Nfar[], const double* Fbodies[]);

void mat_vec_reference(const Eval& eval, int64_t M, int64_t N, std::complex<double> B[], const std::complex<double> X[], const double ibodies[], const double jbodies[]);

