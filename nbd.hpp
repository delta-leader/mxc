
#pragma once

#include "mpi.h"

#include <vector>
#include <utility>
#include <cstdint>
#include <cstddef>

#include <comm.hpp>
#include <basis.hpp>
#include <sparse_row.hpp>

struct Matrix { double* A; int64_t M, N, LDA; };

class Cell {
public:
  int64_t Child[2], Body[2];
  double R[3], C[3];
};

struct Node {
  int64_t lenA, lenS;
  struct Matrix *A, *S;
  double* A_ptr, *X_ptr;
};

struct EvalDouble;

void gen_matrix(const EvalDouble& Eval, int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda);

void mmult(char ta, char tb, const struct Matrix* A, const struct Matrix* B, struct Matrix* C, double alpha, double beta);

void mul_AS(const struct Matrix* RU, const struct Matrix* RV, struct Matrix* A);

int64_t compute_basis(const EvalDouble& eval, double epi, int64_t M, double* A, int64_t LDA, double Xbodies[], int64_t Nfar, const double Fbodies[]);

void mat_vec_reference(const EvalDouble& eval, int64_t begin, int64_t end, double B[], int64_t nbodies, const double* bodies, const double Xbodies[]);

void buildTree(struct Cell* cells, double* bodies, int64_t nbodies, int64_t levels);

void buildTreeBuckets(struct Cell* cells, const double* bodies, const int64_t buckets[], int64_t levels);

void traverse(char NoF, CSR* rels, int64_t ncells, const struct Cell* cells, double theta);

void countMaxIJ(int64_t* max_i, int64_t* max_j, const CSR* rels);

void loadX(double* X, int64_t seg, const double Xbodies[], int64_t Xbegin, int64_t ncells, const struct Cell cells[]);

void evalD(const EvalDouble& eval, struct Matrix* D, const CSR* rels, const struct Cell* cells, const double* bodies, const struct CellComm* comm);

void evalS(const EvalDouble& eval, struct Matrix* S, const struct Base* basis, const CSR* rels, const struct CellComm* comm);

void allocNodes(struct Node A[], const struct Base basis[], const CSR rels_near[], const CSR rels_far[], const struct CellComm comm[], int64_t levels);

void node_free(struct Node* node);

void matVecA(const struct Node A[], const struct Base basis[], const CSR rels_near[], double* X, const struct CellComm comm[], int64_t levels);

void solveRelErr(double* err_out, const double* X, const double* ref, int64_t lenX);


