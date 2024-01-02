
#pragma once

#include <cstdint>

class EvalDouble;

class Matrix {
public:
  double* A;
  int64_t M, N, LDA;
};

void gen_matrix(const EvalDouble& Eval, int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda);

void mmult(char ta, char tb, const Matrix* A, const Matrix* B, Matrix* C, double alpha, double beta);

int64_t compute_basis(const EvalDouble& eval, double epi, int64_t M, double* A, int64_t LDA, double Xbodies[], int64_t Lfar, int64_t Nfar[], const double* Fbodies[]);

void mat_vec_reference(const EvalDouble& eval, int64_t M, int64_t N, double B[], const double X[], const double ibodies[], const double jbodies[]);

