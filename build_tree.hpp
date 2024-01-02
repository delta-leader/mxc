
#pragma once

#include <cstdint>
#include <vector>

class EvalDouble;

class CellComm;
class Base;
class Matrix;

class Cell {
public:
  int64_t Child[2], Body[2];
  double R[3], C[3];
};

class CSR {
public:
  std::vector<int64_t> RowIndex;
  std::vector<int64_t> ColIndex;

  CSR(char NoF, int64_t ncells, const Cell* cells, double theta);
};

void buildTree(Cell* cells, double* bodies, int64_t nbodies, int64_t levels);

void buildTreeBuckets(Cell* cells, const double* bodies, const int64_t buckets[], int64_t levels);


