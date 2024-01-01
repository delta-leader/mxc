
#pragma once

#include <cstdint>

class Matrix;
class CSR;
class Base;
class CellComm;

class Node {
public:
  int64_t lenA, lenS;
  Matrix *A, *S;
  double* A_ptr, *X_ptr;
};

void allocNodes(Node A[], const Base basis[], const CSR rels_near[], const CSR rels_far[], const CellComm comm[], int64_t levels);

void node_free(Node* node);

void matVecA(const Node A[], const Base basis[], const CSR* rels_near, double* X, const CellComm comm[], int64_t levels);

void solveRelErr(double* err_out, const double* X, const double* ref, int64_t lenX);

