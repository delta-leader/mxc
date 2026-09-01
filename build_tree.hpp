
#pragma once

#include <vector>
#include <array>

#include <include/elast3d.hpp>


class Cell {
public:
  std::array<long long, 2> Child;
  std::array<long long, 2> Body;
  std::array<double, 3> R;
  std::array<double, 3> C;
  bool nodes;

  Cell() : Child(std::array<long long, 2>{ -1, -1 }), Body(std::array<long long, 2>{ -1, -1 }), R(std::array<double, 3>{ 0., 0., 0. }), C(std::array<double, 3>{ 0., 0., 0. }), nodes(true) {}
};

class CSR {
public:
  std::vector<long long> RowIndex;
  std::vector<long long> ColIndex;

  CSR(char NoF, const std::vector<Cell>& ci, const std::vector<Cell>& cj, double theta);
  long long lookupIJ(long long i, long long j) const;
};

void buildBinaryTreeElemsOnly(Cell* cells, const elastWave3d::element* elems, long long* indices, long long num_elems, long long levels);
