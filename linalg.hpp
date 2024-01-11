
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

void compute_schur(const Eval& eval, int64_t M, int64_t N, int64_t K, std::complex<double> SijT[], int64_t LD, const double Ibodies[], const double Jbodies[], const double Kbodies[]);

int64_t compute_basis(const Eval& eval, double epi, int64_t M, std::complex<double> A[], int64_t LDA, double Xbodies[], int64_t Lfar, int64_t Nfar[], const double* Fbodies[]);

void mat_vec_reference(const Eval& eval, int64_t M, int64_t N, int64_t nrhs, std::complex<double> B[], int64_t ldB, const std::complex<double> X[], int64_t ldX, const double ibodies[], const double jbodies[]);

