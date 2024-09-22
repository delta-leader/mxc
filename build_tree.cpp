
#include <build_tree.hpp>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <set>
#include <iostream>

/*
In:
  bodies: the points within the cell
  nbodies: the number of points within the cell
Out:
  radius: the radius of the cell
  center: the center point of the cell
*/
void get_bounds(const double* const bodies, const long long nbodies, double radius[], double center[]) { 
  const std::array<double, 3>* bodies3 = reinterpret_cast<const std::array<double, 3>*>(&bodies[0]);
  const std::array<double, 3>* bodies3_end = reinterpret_cast<const std::array<double, 3>*>(&bodies[nbodies * 3]);

  // find the minimum and maximum values in each dimension
  double dim_min[3], dim_max[3];
  for (int d = 0; d < 3; d++) {
    auto minmax = std::minmax_element(bodies3, bodies3_end, 
      [=](const std::array<double, 3>& x, const std::array<double, 3>& y) { return x[d] < y[d]; });
    dim_min[d] = (*minmax.first)[d];
    dim_max[d] = (*minmax.second)[d];
  }

  // calculate the center as (min + max) / 2
  std::transform(dim_min, &dim_min[3], dim_max, center, [](double min, double max) { return (min + max) * 0.5; });
  // calculate the radius as (max - min ) / 2 + border_offset
  std::transform(dim_min, &dim_min[3], dim_max, radius, [](double min, double max) { return (min == max && min == 0.) ? 0. : ((max - min) * 0.5 + 1.e-8); });
}

/*
Recursive function to calculate the indices of the near/far field
In:
  NoF: 'N' for near and 'F' for far field
  cells: the cell array
  ci: index of cell i
  cj: indx of cell cj
  theta: admisibility
Out:
  blocks: The index pairs of the near/far field (vector of pairs)
*/
void getList(const char NoF, const Cell cells[], const long long ci, const long long cj, const double theta, std::vector<std::pair<long long, long long>>& blocks) {
  // squared distance between (the centers of) ci and cj
  double distance = std::transform_reduce(cells[ci].C.begin(), cells[ci].C.end(), cells[cj].C.begin(), (double)0., std::plus<double>(), [](double x, double y) { return (x - y) * (x - y); });
  // squared diameter of ci
  double diam_ci = std::transform_reduce(cells[ci].R.begin(), cells[ci].R.end(), cells[ci].R.begin(), (double)0., std::plus<double>(), std::multiplies<double>());
  // squared diameter of cj
  double diam_cj = std::transform_reduce(cells[cj].R.begin(), cells[cj].R.end(), cells[cj].R.begin(), (double)0., std::plus<double>(), std::multiplies<double>());

  // TODO confirm admissibility condition,
  // usually it is either min or max of the diameters
  bool admis = distance > (theta * (diam_ci + diam_cj));
  bool write_far = NoF == 'F' || NoF == 'f';
  bool write_near = NoF == 'N' || NoF == 'n';
  // store the index pair in blocks if it is in the near/far field
  if (admis ? write_far : write_near)
    blocks.emplace_back(ci, cj);
  
  // if a block is not admissible and both cells have children,
  // TODO since the cluster tree is split to the leaf level, only nodes on the same level can have no children, correct?
  // then call the function recursively for the cross product of each
  // cells children
  if (!admis && cells[ci].Child[0] >= 0 && cells[cj].Child[0] >= 0)
    for (long long i = cells[ci].Child[0]; i < cells[ci].Child[1]; i++)
      for (long long j = cells[cj].Child[0]; j < cells[cj].Child[1]; j++)
        getList(NoF, cells, i, j, theta, blocks);
}

CSR::CSR(const char NoF, const std::vector<Cell>& cells, const double theta) {
  long long ncells = cells.size();
  std::vector<std::pair<long long, long long>> blocks;
  getList(NoF, &cells[0], 0, 0, theta, blocks);
  // sorts the blocks according to i first, then according to j
  std::sort(blocks.begin(), blocks.end());

  // number of elements
  long long nblocks = blocks.size();
  RowIndex.resize(ncells + 1);
  ColIndex.resize(nblocks);
  // write the column indices in the CSR
  std::transform(blocks.begin(), blocks.end(), ColIndex.begin(), 
    [](const std::pair<long long, long long>& i) { return i.second; });

  RowIndex[0] = 0;
  // write the offests
  // basically counts the number of elements until the row index i changes
  for (long long n = 1; n <= ncells; n++)
    RowIndex[n] = std::distance(blocks.begin(), 
      std::find_if(blocks.begin() + RowIndex[n - 1], blocks.end(), 
        [=](const std::pair<long long, long long>& i) { return n <= i.first; }));
}

CSR::CSR(const CSR& A, const CSR& B) {
  long long M = A.RowIndex.size() - 1;
  RowIndex.resize(M + 1);
  ColIndex.clear();
  RowIndex[0] = 0;

  for (long long y = 0; y < M; y++) {
    std::set<long long> cols;
    cols.insert(A.ColIndex.begin() + A.RowIndex[y], A.ColIndex.begin() + A.RowIndex[y + 1]);
    cols.insert(B.ColIndex.begin() + B.RowIndex[y], B.ColIndex.begin() + B.RowIndex[y + 1]);

    for (std::set<long long>::iterator i = cols.begin(); i != cols.end(); i = std::next(i))
      ColIndex.push_back(*i);
    RowIndex[y + 1] = (long long)ColIndex.size();
  }
}

long long CSR::lookupIJ(long long i, long long j) const {
  if (i < 0 || RowIndex.size() <= (1ull + i))
    return -1;
  long long k = std::distance(ColIndex.begin(), std::find(ColIndex.begin() + RowIndex[i], ColIndex.begin() + RowIndex[i + 1], j));
  return (k < RowIndex[i + 1]) ? k : -1;
}

MatrixDesc::MatrixDesc(long long lbegin, long long lend, long long ubegin, long long uend, const std::pair<long long, long long> Tree[], const CSR& Near, const CSR& Far) : 
Y(lbegin),
M(lend - lbegin),
NZA(Near.RowIndex[lend] - Near.RowIndex[lbegin]),
NZC(Far.RowIndex[lend] - Far.RowIndex[lbegin]),
ARowOffsets(&Near.RowIndex[lbegin], &Near.RowIndex[lend + 1]),
ACoordinatesY(NZA),
ACoordinatesX(&Near.ColIndex[Near.RowIndex[lbegin]], &Near.ColIndex[Near.RowIndex[lend]]),
AUpperCoordinates(NZA, -1),
AUpperIndexY(NZA),
AUpperIndexX(NZA),
CRowOffsets(&Far.RowIndex[lbegin], &Far.RowIndex[lend + 1]),
CCoordinatesY(NZC),
CCoordinatesX(&Far.ColIndex[Far.RowIndex[lbegin]], &Far.ColIndex[Far.RowIndex[lend]]),
CUpperCoordinates(NZC, -1),
CUpperIndexY(NZC),
CUpperIndexX(NZC),
XUpperCoordinatesY(uend - ubegin, -1) {
  
  long long offsetA = ARowOffsets[0], offsetC = CRowOffsets[0];
  std::for_each(ARowOffsets.begin(), ARowOffsets.end(), [=](long long& i) { i = i - offsetA; });
  std::for_each(CRowOffsets.begin(), CRowOffsets.end(), [=](long long& i) { i = i - offsetC; });

  for (long long i = 0; i < M; i++) {
    long long gi = i + lbegin;
    std::fill(&ACoordinatesY[ARowOffsets[i]], &ACoordinatesY[ARowOffsets[i + 1]], gi);
    std::fill(&CCoordinatesY[CRowOffsets[i]], &CCoordinatesY[CRowOffsets[i + 1]], gi);
  }

  long long offsetU = (0 <= ubegin && ubegin < uend) ? Near.RowIndex[ubegin] : 0;
  for (long long i = ubegin; i < uend; i++)
    for (long long ci = Tree[i].first; ci < Tree[i].second; ci++)
      for (long long ij = Near.RowIndex[i]; ij < Near.RowIndex[i + 1]; ij++) {
        long long j = Near.ColIndex[ij];
        for (long long cj = Tree[j].first; cj < Tree[j].second; cj++) {
          long long iA = Near.lookupIJ(ci, cj) - offsetA;
          long long iC = Far.lookupIJ(ci, cj) - offsetC;
          if (0 <= iA && iA < NZA) {
            AUpperCoordinates[iA] = ij - offsetU;
            AUpperIndexY[iA] = ci - Tree[i].first;
            AUpperIndexX[iA] = cj - Tree[j].first;
          }
          if (0 <= iC && iC < NZC) {
            CUpperCoordinates[iC] = ij - offsetU;
            CUpperIndexY[iC] = ci - Tree[i].first;
            CUpperIndexX[iC] = cj - Tree[j].first;
          }
        }
      }
  
  if (0 <= ubegin && ubegin < uend)
    std::transform(&Tree[ubegin], &Tree[uend], XUpperCoordinatesY.begin(), [](const std::pair<long long, long long>& c) { return c.first; });
}

void buildBinaryTree(const long long nlevels, const long long nbodies, double* const bodies, Cell* const cells) {
  // the root node contains all points
  cells[0].Body[0] = 0;
  cells[0].Body[1] = nbodies;
  // calculate boundaries for the root
  get_bounds(bodies, nbodies, cells[0].R.data(), cells[0].C.data());

  // the number of non leaf nodes
  long long inner_nodes = ((long long)1 << nlevels) - 1;
  // iterate over all non-leaf nodes (in ascending order)
  for (long long i = 0; i < inner_nodes; i++) {
    Cell& parent = cells[i];
    // split along the dimension with the longest radius
    long long splitting_dim = std::distance(parent.R.begin(), std::max_element(parent.R.begin(), parent.R.end()));
    // start and end indices of points from the parent
    long long bodies_start = parent.Body[0];
    long long bodies_end = parent.Body[1];

    // sort the bodies according to the splitting dimension
    std::array<double, 3>* bodies3 = reinterpret_cast<std::array<double, 3>*>(&bodies[bodies_start * 3]);
    std::array<double, 3>* bodies3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[bodies_end * 3]);
    std::sort(bodies3, bodies3_end, 
      [=](std::array<double, 3>& i, std::array<double, 3>& j) { return i[splitting_dim] < j[splitting_dim]; });

    // starting index of the children cells within the cell array
    long long cells_start = (i << 1) + 1;
    // create the children cells
    Cell& child0 = cells[cells_start];
    Cell& child1 = cells[cells_start + 1];
    // set the children indices in the parent (end is non inclusive)
    parent.Child[0] = cells_start;
    parent.Child[1] = cells_start + 2;

    // calculate the splitting index
    long long splitting_idx = bodies_start + (bodies_end - bodies_start) / 2;
    child0.Body[0] = bodies_start;
    child0.Body[1] = splitting_idx;
    child1.Body[0] = splitting_idx;
    child1.Body[1] = bodies_end;

    // calculate boundaries for the children
    get_bounds(&bodies[bodies_start * 3], splitting_idx - bodies_start, child0.R.data(), child0.C.data());
    get_bounds(&bodies[splitting_idx * 3], bodies_end - splitting_idx, child1.R.data(), child1.C.data());
  }
}
