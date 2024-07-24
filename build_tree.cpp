
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

void getList(char NoF, std::vector<std::pair<long long, long long>>& rels, const Cell ci[], long long i, const Cell cj[], long long j, double theta) {
  double dC = std::transform_reduce(ci[i].C.begin(), ci[i].C.end(), cj[j].C.begin(), (double)0., std::plus<double>(), [](double x, double y) { return (x - y) * (x - y); });
  double dR1 = std::transform_reduce(ci[i].R.begin(), ci[i].R.end(), ci[i].R.begin(), (double)0., std::plus<double>(), std::multiplies<double>());
  double dR2 = std::transform_reduce(cj[j].R.begin(), cj[j].R.end(), cj[j].R.begin(), (double)0., std::plus<double>(), std::multiplies<double>());

  bool admis = dC > (theta * (dR1 + dR2));
  bool write_far = NoF == 'F' || NoF == 'f';
  bool write_near = NoF == 'N' || NoF == 'n';
  if (admis ? write_far : write_near)
    rels.emplace_back(i, j);
  
  if (!admis && ci[i].Child[0] >= 0 && cj[j].Child[0] >= 0)
    for (long long k = ci[i].Child[0]; k < ci[i].Child[1]; k++)
      for (long long l = cj[j].Child[0]; l < cj[j].Child[1]; l++)
        getList(NoF, rels, ci, k, cj, l, theta);
}

CSR::CSR(char NoF, const std::vector<Cell>& ci, const std::vector<Cell>& cj, double theta) {
  long long ncells = ci.size();
  std::vector<std::pair<long long, long long>> LIL;
  getList(NoF, LIL, &ci[0], 0, &cj[0], 0, theta);
  std::sort(LIL.begin(), LIL.end());

  long long len = LIL.size();
  RowIndex.resize(ncells + 1);
  ColIndex.resize(len);
  std::transform(LIL.begin(), LIL.end(), ColIndex.begin(), 
    [](const std::pair<long long, long long>& i) { return i.second; });

  RowIndex[0] = 0;
  for (long long n = 1; n <= ncells; n++)
    RowIndex[n] = std::distance(LIL.begin(), 
      std::find_if(LIL.begin() + RowIndex[n - 1], LIL.end(), 
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
