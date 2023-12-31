
#pragma once

#include <cstdint>

class EvalDouble;
class CSR;
class CellComm;
class Base;
class Matrix;

class Cell {
public:
  int64_t Child[2], Body[2];
  double R[3], C[3];
};


void buildTree(Cell* cells, double* bodies, int64_t nbodies, int64_t levels);

void buildTreeBuckets(Cell* cells, const double* bodies, const int64_t buckets[], int64_t levels);

void traverse(char NoF, CSR* rels, int64_t ncells, const Cell* cells, double theta);

void loadX(double* X, int64_t seg, const double Xbodies[], int64_t Xbegin, int64_t ncells, const Cell cells[]);

void evalD(const EvalDouble& eval, Matrix* D, const CSR* rels, const Cell* cells, const double* bodies, const CellComm* comm);

void evalS(const EvalDouble& eval, Matrix* S, const Base* basis, const CSR* rels, const CellComm* comm);


