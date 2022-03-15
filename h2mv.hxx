
#pragma once
#include "basis.hxx"

namespace nbd {

  struct MatVec {
    Vectors X;
    Vectors M;
    Vectors L;
    Vectors B;
  };

  void interTrans(char updn, MatVec& vx, const Matrices& basis, int64_t level);

  void horizontalPass(Vectors& B, const Vectors& X, EvalFunc ef, const Cell* cell, int64_t dim, int64_t level);

  void closeQuarter(Vectors& B, const Vectors& X, EvalFunc ef, const Cell* cell, int64_t dim, int64_t level);

  void permuteAndMerge(char fwbk, Vectors& px, Vectors& nx, int64_t nlevel);

  void allocMatVec(MatVec vx[], const Base base[], int64_t levels);

  void resetMatVec(MatVec vx[], const Vectors& X, int64_t levels);

  void h2MatVecLR(MatVec vx[], EvalFunc ef, const Cell* root, const Base basis[], int64_t dim, const Vectors& X, int64_t levels, int64_t mpi_rank, int64_t mpi_size);

  void h2MatVecAll(MatVec vx[], EvalFunc ef, const Cell* root, const Base basis[], int64_t dim, const Vectors& X, int64_t levels, int64_t mpi_rank, int64_t mpi_size);

  void zeroC(const Cell* cell, double* c, int64_t lmin, int64_t lmax);

  void multiplyC(EvalFunc ef, const Cells& cells, int64_t dim, int64_t level, const Matrices& base, double* m, double* l);

  int64_t C2X(const Cells& cells, int64_t level, const double* c, double* x);

  int64_t X2C(const Cells& cells, int64_t level, const double* x, double* c);

  void h2inv(EvalFunc ef, const Cells& cells, int64_t dim, const Matrices& base, const Matrices d[], const CSC rels[], double* x);

}
