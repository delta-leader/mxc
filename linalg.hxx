
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

  void cpyFromMatrix(char trans, const Matrix& A, double* v);

  void maxpy(Matrix& A, const double* v, double alpha);

  void cpyMatToMat(int64_t m, int64_t n, const Matrix& m1, Matrix& m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2);

  int64_t orthoBase(double repi, Matrix& A, Matrix& Us, Matrix& Uc);

  void zeroMatrix(Matrix& A);

  void mmult(char ta, char tb, const Matrix& A, const Matrix& B, Matrix& C, double alpha, double beta);

  void msample(char ta, int64_t lenR, const Matrix& A, const double* R, Matrix& C);

  void msample_m(char ta, const Matrix& A, const Matrix& B, Matrix& C);

  void minv(char ta, char lr, Matrix& A, Matrix& B);

  void chol_decomp(Matrix& A);

  void trsm_lowerA(Matrix& A, const Matrix& L);

  void utav(const Matrix& U, const Matrix& A, const Matrix& VT, Matrix& C);

  void axat(Matrix& A, Matrix& AT);

  void chol_solve(Vector& X, const Matrix& A);

  void fw_solve(Vector& X, const Matrix& L);

  void bk_solve(Vector& X, const Matrix& L);

  void mvec(char ta, const Matrix& A, const Vector& X, Vector& B, double alpha, double beta);

  void pvc_fw(const Vector& X, const Matrix& Us, const Matrix& Uc, Vector& Xs, Vector& Xc);

  void pvc_bk(const Vector& Xs, const Vector& Xc, const Matrix& Us, const Matrix& Uc, Vector& X);
  
  typedef std::vector<Matrix> Matrices;
  typedef std::vector<Vector> Vectors;

  //void cpsVectors(char updwn, const Vectors& Xs, Vectors& Xt);

  //void cpsMatrices(Matrices& Mup, const CSC& rels_up, const Matrices& Mlow, const CSC& rels_low);

};