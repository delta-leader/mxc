
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

void buildBinaryTree(Cell* cells, double* bodies, long long nbodies, long long levels);
void buildBinaryTree(Cell* cells, double* bodies, long long* indices, long long nbodies, long long levels);
void buildBinaryTree(Cell* cells, double* bodies, long long* indices, long long nbodies, long long levels, long long start, long long bodies_offset);
void buildBinaryTree2(Cell* cells, double* bodies, long long* indices, long long nbodies, long long levels);
void buildBinaryTree3(Cell* cells, double* bodies, long long* indices, long long nbodies, long long levels, long long start, long long bodies_offset);
void buildBinaryTreeNodes(Cell* cells, elastWave3d::nodal_point* nodes, long long num_nodes, long long levels, long long first_cell_idx);
void buildBinaryTreeElems(Cell* cells, elastWave3d::element* nodes, long long num_nodes, long long levels, long long first_cell_idx);
void buildBinaryTreeNodes(Cell* cells, elastWave3d::nodal_point* nodes, long long* indices, long long num_nodes, long long levels, long long first_cell_idx);
void buildBinaryTreeElems(Cell* cells, elastWave3d::element* nodes, long long* indices, long long num_nodes, long long levels, long long first_cell_idx);
