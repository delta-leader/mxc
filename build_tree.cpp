#include <build_tree.hpp>

#include <numeric>
#include <cstring>


void get_bounds(const elastWave3d::element* elems, long long num_elems, double R[], double C[]) {
  double Xmin[3], Xmax[3];
  for (int i = 0; i < 3; i++) {
    auto minmax = std::minmax_element(&elems[0], &elems[num_elems], 
      [=](const elastWave3d::element& x, const elastWave3d::element& y) { return x.xc[i] < y.xc[i]; });
    Xmin[i] = (*minmax.first).xc[i];
    Xmax[i] = (*minmax.second).xc[i];
  }
  std::transform(Xmin, &Xmin[3], Xmax, C, [](double min, double max) { return (min + max) * 0.5; });
  std::transform(Xmin, &Xmin[3], Xmax, R, [](double min, double max) { return (min == max && min == 0.) ? 0. : ((max - min) * 0.5 + 1.e-8); });
}

void get_bounds(const elastWave3d::element* elems, long long * indices, long long num_elems, double R[], double C[]) {
  double Xmin[3], Xmax[3];
  for (int i = 0; i < 3; i++) {
    auto minmax = std::minmax_element(&indices[0], &indices[num_elems], 
      [=](const long long x, const long long y) { return elems[x].xc[i] < elems[y].xc[i]; });
    Xmin[i] = elems[*minmax.first].xc[i];
    Xmax[i] = elems[*minmax.second].xc[i];
  }
  std::transform(Xmin, &Xmin[3], Xmax, C, [](double min, double max) { return (min + max) * 0.5; });
  std::transform(Xmin, &Xmin[3], Xmax, R, [](double min, double max) { return (min == max && min == 0.) ? 0. : ((max - min) * 0.5 + 1.e-8); });
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

long long CSR::lookupIJ(long long i, long long j) const {
  if (i < 0 || RowIndex.size() <= (1ull + i))
    return -1;
  long long k = std::distance(ColIndex.begin(), std::find(ColIndex.begin() + RowIndex[i], ColIndex.begin() + RowIndex[i + 1], j));
  return (k < RowIndex[i + 1]) ? k : -1;
}

void buildBinaryTreeElemsOnly(Cell* cells, const elastWave3d::element* elems, long long* indices, long long num_elems, long long levels) {
  cells[0].Body[0] = 0;
  cells[0].Body[1] = num_elems;
  cells[0].nodes = false;
  get_bounds(elems, num_elems, cells[0].R.data(), cells[0].C.data());

  long long nleaf = (long long)1 << levels;
  for (long long i = 0; i < nleaf - 1; ++i) {
    Cell& ci = cells[i];
    long long sdim = std::distance(ci.R.begin(), std::max_element(ci.R.begin(), ci.R.end()));
    long long elems_begin = ci.Body[0];
    long long elems_end = ci.Body[1];

    std::vector<long long> sort_idx(elems_end - elems_begin);
    std::iota(sort_idx.begin(), sort_idx.end(), 0);
    std::sort(sort_idx.begin(), sort_idx.end(), 
      [&](size_t i, size_t j) { return elems[indices[elems_begin + i]].xc[sdim] < elems[indices[elems_begin + j]].xc[sdim]; });

    std::vector<long long> indices_copy(elems_end - elems_begin);
    std::memcpy(indices_copy.data(), &indices[elems_begin], sizeof(long long) * indices_copy.size());
    for (size_t i = 0; i < indices_copy.size(); ++i) {
      indices[elems_begin + i] = indices_copy[sort_idx[i]];
    }
    long long len = (i << 1) + 1;
    Cell& child0 = cells[len];
    Cell& child1 = cells[len + 1];
    ci.Child[0] = len;
    ci.Child[1] = len + 2;

    long long elems_mid = elems_begin + (elems_end - elems_begin) / 2;
    child0.Body[0] = elems_begin;
    child0.Body[1] = elems_mid;
    child0.nodes = false;
    child1.Body[0] = elems_mid;
    child1.Body[1] = elems_end;
    child1.nodes = false;

    get_bounds(elems, &indices[elems_begin], elems_mid - elems_begin, child0.R.data(), child0.C.data());
    get_bounds(elems, &indices[elems_mid], elems_end - elems_mid, child1.R.data(), child1.C.data());
  }
}