
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
  long long lookupIJ(long long i, long long j) const;
};

class MatrixDesc { // Coordinates:[location in CSR], Index:[N'th child in upper]
public:
  long long Y;
  long long M;
  long long NZA;
  long long NZC;

  std::vector<long long> ARowOffsets;
  std::vector<long long> ACoordinatesY;
  std::vector<long long> ACoordinatesX;
  std::vector<long long> AUpperCoordinates;
  std::vector<long long> AUpperIndexY;
  std::vector<long long> AUpperIndexX;

  std::vector<long long> CRowOffsets;
  std::vector<long long> CCoordinatesY;
  std::vector<long long> CCoordinatesX;
  std::vector<long long> CUpperCoordinates;
  std::vector<long long> CUpperIndexY;
  std::vector<long long> CUpperIndexX;

  std::vector<long long> XUpperCoordinatesY;

  MatrixDesc(long long lbegin, long long lend, long long ubegin, long long uend, const std::pair<long long, long long> Tree[], const CSR& Near, const CSR& Far);
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
void buildBinaryTree(const long long nlevels, const long long nbodies, double* const bodies, Cell* const cells);
