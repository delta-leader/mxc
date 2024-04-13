
#pragma once

#include <vector>
#include <array>

class Cell {
public:
  long long Parent;
  std::array<long long, 2> Child;
  std::array<long long, 2> Body;
  std::array<double, 3> R;
  std::array<double, 3> C;

  Cell() : Parent(-1), Child(std::array<long long, 2>{ -1, -1 }), Body(std::array<long long, 2>{ -1, -1 }), R(std::array<double, 3>{ 0., 0., 0. }), C(std::array<double, 3>{ 0., 0., 0. }) {}
};

typedef std::vector<Cell> Cells;

class CSR {
public:
  std::vector<long long> RowIndex;
  std::vector<long long> ColIndex;

  CSR(char NoF, const Cells& ci, const Cells& cj, double theta);
};

void buildBinaryTree(Cell* cells, double* bodies, long long nbodies, long long levels);
