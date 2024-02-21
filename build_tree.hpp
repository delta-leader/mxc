
#pragma once

#include <vector>

class Cell {
public:
  long long Child[2], ParentSeq, Body[2];
  double R[3], C[3];

  Cell();
};

class CSR {
public:
  std::vector<long long> RowIndex;
  std::vector<long long> ColIndex;

  CSR(char NoF, long long ncells, const Cell* cells, double theta);
};

void buildTree(Cell* cells, double* bodies, long long nbodies, long long levels);

void buildTreeBuckets(Cell* cells, const double* bodies, const long long buckets[], long long levels);
