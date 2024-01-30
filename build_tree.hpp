
#pragma once

#include <cstdint>
#include <vector>

class Cell {
public:
  int64_t Child[2], Body[2];
  double R[3], C[3];

  Cell();
};

class CSR {
public:
  std::vector<int64_t> RowIndex;
  std::vector<int64_t> ColIndex;

  CSR(char NoF, int64_t ncells, const Cell* cells, double theta);
  CSR(const CSR& A, const CSR& B);
};

void buildTree(Cell* cells, double* bodies, int64_t nbodies, int64_t levels);

void buildTreeBuckets(Cell* cells, const double* bodies, const int64_t buckets[], int64_t levels);

std::vector<int64_t> getLevelOffsets(const Cell cells[], int64_t ncells);

std::vector<std::pair<int64_t, int64_t>> getProcessMapping(int64_t mpi_size, const Cell cells[], int64_t ncells);

void getLocalRange(int64_t& level_begin, int64_t& level_end, int64_t mpi_rank, const std::vector<std::pair<int64_t, int64_t>>& mapping);
