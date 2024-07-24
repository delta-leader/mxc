
#pragma once

#include <vector>
#include <array>

class Cell {
public:
  // Start and End indices of the children cells
  // within the cells array
  std::array<long long, 2> Child;
  // Start and End indices of the points contained within the cell
  // with respect to the bodies array
  std::array<long long, 2> Body;
  // radius
  std::array<double, 3> R;
  // center
  std::array<double, 3> C;

  Cell() : Child(std::array<long long, 2>{ -1, -1 }), Body(std::array<long long, 2>{ -1, -1 }), R(std::array<double, 3>{ 0., 0., 0. }), C(std::array<double, 3>{ 0., 0., 0. }) {}
};

// compressed row storage
// a non-zero element i x j in the stored block cluster tree I x J
// is given by ColIndex[RowIndex[i] + j]
class CSR {
public:
  // Starting index for each cell in ColIndex
  std::vector<long long> RowIndex;
  // indices of j in the i x j block cluster tree
  std::vector<long long> ColIndex;

  /*
  NoF: either 'F' for far field or 'N' for near field
  cells: the cluster tree
  theta: admisibility
  */
  CSR(const char NoF, const std::vector<Cell>& cells, const double theta);
  // combines the content of two sparse matrices
  CSR(const CSR& A, const CSR& B);
};

/*
In: 
  nlevels: the number of levels
  nbodies: the number of points
InOut:
  bodies: the points (output is sorted)
Out:
  cells: the nodes in the cluster tree
*/
void buildBinaryTree(long long nlevels, const long long nbodies, double* const bodies, Cell* const cells);
