
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

void buildBinaryTree(Cell* cells, double* bodies, long long nbodies, long long levels);
