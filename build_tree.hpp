
#pragma once

#include <vector>
#include <array>

class Cell {
public:
  std::array<long long, 2> Child;
  std::array<long long, 2> Body;
  // radius
  std::array<double, 3> R;
  // center
  std::array<double, 3> C;

  Cell() : Child(std::array<long long, 2>{ -1, -1 }), Body(std::array<long long, 2>{ -1, -1 }), R(std::array<double, 3>{ 0., 0., 0. }), C(std::array<double, 3>{ 0., 0., 0. }) {}
};

//compressed row storage
class CSR {
public:
  // Starting index for each cell in ColIndex
  std::vector<long long> RowIndex;
  // indices of j in the i x j block cluster tree
  std::vector<long long> ColIndex;

  /*
  NoF: either 'F' for far field or 'N' for near field
  ci: node i from the cluster tree
  cj:  node j from the cluster tree
  theta: admisibility
  */
  CSR(char NoF, const std::vector<Cell>& ci, const std::vector<Cell>& cj, double theta);
  // combines the content of two sparse jmatrices
  CSR(const CSR& A, const CSR& B);
};

/*
cells: the cell array (output)
bodies: the points
nbodies: N (number of points)
levels: number of levels
*/
void buildBinaryTree(Cell* cells, double* bodies, long long nbodies, long long levels);
