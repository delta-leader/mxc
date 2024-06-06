
#pragma once

#include <vector>
#include <array>

class Cell {
public:
  std::array<long long, 2> Child;
  std::array<long long, 2> Body;
  std::array<double, 3> R;
  std::array<double, 3> C;

  Cell() : Child(std::array<long long, 2>{ -1, -1 }), Body(std::array<long long, 2>{ -1, -1 }), R(std::array<double, 3>{ 0., 0., 0. }), C(std::array<double, 3>{ 0., 0., 0. }) {}
};

class CSR {
public:
  std::vector<long long> RowIndex;
  std::vector<long long> ColIndex;

  CSR(char NoF, const std::vector<Cell>& ci, const std::vector<Cell>& cj, double theta);
  CSR(const CSR& A, const CSR& B);
};

void buildBinaryTree(Cell* cells, double* bodies, long long nbodies, long long levels);
