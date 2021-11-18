
#pragma once

#include <vector>
#include <cstdint>

namespace nbd {
    
  struct Matrix {
    std::vector<double> A;
    int64_t M;
    int64_t N;
  };

  struct Vector {
    std::vector<double> X;
    int64_t N;
  };

  void cMatrix(Matrix& mat, int64_t m, int64_t n);

  void cVector(Vector& vec, int64_t n);

  int64_t cpyFromMatrix(const Matrix& mat, double* m);

  int64_t cpyToMatrix(Matrix& mat, const double* m);

  int64_t cpyFromVector(const Vector& vec, double* v);

  int64_t cpyToVector(Vector& vec, const double* v);

  void cpyMatToMat(int64_t m, int64_t n, const Matrix& m1, Matrix& m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2);

  int64_t orthoBase(double repi, Matrix& A, Matrix& Us, Matrix& Uc);

  void zeroMatrix(Matrix& A);

  void mmult(char ta, char tb, const Matrix& A, const Matrix& B, Matrix& C, double alpha, double beta);

  void msample(char ta, int64_t lenR, const Matrix& A, const double* R, Matrix& C);

  void msample_m(char ta, const Matrix& A, const Matrix& B, Matrix& C);

  void minv(char ta, char lr, Matrix& A, Matrix& B);

  void lu_decomp(Matrix& A);

  void trsm_lowerA(Matrix& A, const Matrix& U);

  void trsm_upperA(Matrix& A, const Matrix& L);

  void utav(const Matrix& U, const Matrix& A, const Matrix& VT, Matrix& C);

  void lu_solve(Vector& X, const Matrix& A);

  void fw_solve(Vector& X, const Matrix& L);

  void bk_solve(Vector& X, const Matrix& U);

  void mvec(char ta, const Matrix& A, const Vector& X, Vector& B, double alpha, double beta);

  void pvc_fw(const Vector& X, const Matrix& Us, const Matrix& Uc, Vector& Xs, Vector& Xc);

  void pvc_bk(const Vector& Xs, const Vector& Xc, const Matrix& Us, const Matrix& Uc, Vector& X);

  struct CSC {
    int64_t M;
    int64_t N;
    int64_t NNZ;
    std::vector<int64_t> CSC_COLS;
    std::vector<int64_t> CSC_ROWS;
  };

  struct CSR {
    int64_t M;
    int64_t N;
    int64_t NNZ;
    std::vector<int64_t> CSR_ROWS;
    std::vector<int64_t> CSR_COLS;
  };
  
  typedef std::vector<Matrix> Matrices;
  typedef std::vector<Vector> Vectors;

  void lookupIJ(int64_t& ij, const CSC& rels, int64_t i, int64_t j);

  void toCSR(CSR& rels_csr, const CSC& rels_csc);

  void cVectors(Vectors& Xs, int64_t n, const int64_t* dims);

  int64_t ctoVectors(Vectors& Xs, const double* X);

  int64_t cbkVectors(double* X, const Vectors& Xs);

  void cpsVectors(char updwn, const Vectors& Xs, Vectors& Xt);

  void cMatrices(Matrices& Ms, const CSC& rels, const int64_t* Ydims, const int64_t* Xdims);

  int64_t ctoMatrices(Matrices& Ms, const double* M);

  int64_t cbkMatrices(double* M, const Matrices& Ms);

  void cpsMatrices(Matrices& Mup, const CSC& rels_up, const Matrices& Mlow, const CSC& rels_low);

};