
#include <build_tree.hpp>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <set>
#include <cstring>

#include<iostream>

void get_bounds(const double* bodies, long long nbodies, double R[], double C[]) {
  const std::array<double, 3>* bodies3 = reinterpret_cast<const std::array<double, 3>*>(&bodies[0]);
  const std::array<double, 3>* bodies3_end = reinterpret_cast<const std::array<double, 3>*>(&bodies[nbodies * 3]);

  double Xmin[3], Xmax[3];
  for (int i = 0; i < 3; i++) {
    auto minmax = std::minmax_element(bodies3, bodies3_end, 
      [=](const std::array<double, 3>& x, const std::array<double, 3>& y) { return x[i] < y[i]; });
    Xmin[i] = (*minmax.first)[i];
    Xmax[i] = (*minmax.second)[i];
  }

  std::transform(Xmin, &Xmin[3], Xmax, C, [](double min, double max) { return (min + max) * 0.5; });
  std::transform(Xmin, &Xmin[3], Xmax, R, [](double min, double max) { return (min == max && min == 0.) ? 0. : ((max - min) * 0.5 + 1.e-8); });
}

void get_bounds2(const double* bodies, long long nbodies, double R[], double C[]) {
  const std::array<double, 2>* bodies3 = reinterpret_cast<const std::array<double, 2>*>(&bodies[0]);
  const std::array<double, 2>* bodies3_end = reinterpret_cast<const std::array<double, 2>*>(&bodies[nbodies * 2]);

  double Xmin[2], Xmax[2];
  for (int i = 0; i < 2; i++) {
    auto minmax = std::minmax_element(bodies3, bodies3_end, 
      [=](const std::array<double, 2>& x, const std::array<double, 2>& y) { return x[i] < y[i]; });
    Xmin[i] = (*minmax.first)[i];
    Xmax[i] = (*minmax.second)[i];
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

void buildBinaryTree(Cell* cells, double* bodies, long long nbodies, long long levels) {
  cells[0].Body[0] = 0;
  cells[0].Body[1] = nbodies;
  get_bounds(bodies, nbodies, cells[0].R.data(), cells[0].C.data());

  long long nleaf = (long long)1 << levels;
  for (long long i = 0; i < nleaf - 1; i++) {
    Cell& ci = cells[i];
    long long sdim = std::distance(ci.R.begin(), std::max_element(ci.R.begin(), ci.R.end()));
    long long i_begin = ci.Body[0];
    long long i_end = ci.Body[1];

    std::array<double, 3>* bodies3 = reinterpret_cast<std::array<double, 3>*>(&bodies[i_begin * 3]);
    std::array<double, 3>* bodies3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[i_end * 3]);
    std::sort(bodies3, bodies3_end, 
      [=](std::array<double, 3>& i, std::array<double, 3>& j) { return i[sdim] < j[sdim]; });

    long long len = (i << 1) + 1;
    Cell& c0 = cells[len];
    Cell& c1 = cells[len + 1];
    ci.Child[0] = len;
    ci.Child[1] = len + 2;

    long long loc = i_begin + (i_end - i_begin) / 2;
    c0.Body[0] = i_begin;
    c0.Body[1] = loc;
    c1.Body[0] = loc;
    c1.Body[1] = i_end;

    get_bounds(&bodies[i_begin * 3], loc - i_begin, c0.R.data(), c0.C.data());
    get_bounds(&bodies[loc * 3], i_end - loc, c1.R.data(), c1.C.data());
  }
}

void buildBinaryTree(Cell* cells, double* bodies, long long* indices, long long nbodies, long long levels) {
  cells[0].Body[0] = 0;
  cells[0].Body[1] = nbodies;
  get_bounds(bodies, nbodies, cells[0].R.data(), cells[0].C.data());

  long long nleaf = (long long)1 << levels;
  for (long long i = 0; i < nleaf - 1; i++) {
    Cell& ci = cells[i];
    long long sdim = std::distance(ci.R.begin(), std::max_element(ci.R.begin(), ci.R.end()));
    long long i_begin = ci.Body[0];
    long long i_end = ci.Body[1];

    long long i_num = i_end - i_begin;
    std::vector<long long> sort_idx(i_num);
    std::iota(sort_idx.begin(), sort_idx.end(), 0);
    std::array<double, 3>* bodies3 = reinterpret_cast<std::array<double, 3>*>(&bodies[i_begin * 3]);
    std::array<double, 3>* bodies3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[i_end * 3]);
    std::sort(sort_idx.begin(), sort_idx.end(), 
      [&](size_t i, size_t j) { return bodies3[i][sdim] < bodies3[j][sdim]; });

    std::vector<double> bodies_copy(i_num * 3);
    std::vector<long long> indices_copy(i_num);
    std::memcpy(bodies_copy.data(), &bodies[i_begin * 3], sizeof(double) * i_num * 3);
    std::memcpy(indices_copy.data(), &indices[i_begin], sizeof(long long) * i_num);
    for (long long i = 0; i < i_num; ++i) {
      for (long long j = 0; j < 3; ++j)
        bodies[(i_begin + i) * 3 + j] = bodies_copy[sort_idx[i] * 3 + j];
      indices[i_begin + i] = indices_copy[sort_idx[i]];
    }

    long long len = (i << 1) + 1;
    Cell& c0 = cells[len];
    Cell& c1 = cells[len + 1];
    ci.Child[0] = len;
    ci.Child[1] = len + 2;

    long long loc = i_begin + (i_end - i_begin) / 2;
    c0.Body[0] = i_begin;
    c0.Body[1] = loc;
    c1.Body[0] = loc;
    c1.Body[1] = i_end;

    get_bounds(&bodies[i_begin * 3], loc - i_begin, c0.R.data(), c0.C.data());
    get_bounds(&bodies[loc * 3], i_end - loc, c1.R.data(), c1.C.data());
  }
}

void buildBinaryTree(Cell* cells, double* bodies, long long* indices, long long nbodies, long long levels, long long start, long long bodies_offset) {
  cells[start].Body[0] = bodies_offset;
  cells[start].Body[1] = bodies_offset + nbodies;
  get_bounds(bodies, nbodies, cells[start].R.data(), cells[start].C.data());

  long long nleaf = (long long)1 << levels;
  for (long long level = 0; level < levels; ++ level) {
    //std::cout<<"Level "<<level<<std::endl;
    for (long long i = 0; i < (1 << level); i++) {
      //std::cout<<"i "<<i<<std::endl;
      long long offset = start << level;
      //std::cout<<"cell "<<offset+i<<std::endl;
      Cell& ci = cells[offset + i];
      long long sdim = std::distance(ci.R.begin(), std::max_element(ci.R.begin(), ci.R.end()));
      long long i_begin = ci.Body[0] - bodies_offset;
      long long i_end = ci.Body[1] - bodies_offset;

      long long i_num = i_end - i_begin;
      std::vector<long long> sort_idx(i_num);
      std::iota(sort_idx.begin(), sort_idx.end(), 0);
      std::array<double, 3>* bodies3 = reinterpret_cast<std::array<double, 3>*>(&bodies[i_begin * 3]);
      std::array<double, 3>* bodies3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[i_end * 3]);
      std::sort(sort_idx.begin(), sort_idx.end(), 
        [&](size_t i, size_t j) { return bodies3[i][sdim] < bodies3[j][sdim]; });

      std::vector<double> bodies_copy(i_num * 3);
      std::vector<long long> indices_copy(i_num);
      std::memcpy(bodies_copy.data(), &bodies[i_begin * 3], sizeof(double) * i_num * 3);
      std::memcpy(indices_copy.data(), &indices[i_begin], sizeof(long long) * i_num);
      for (long long i = 0; i < i_num; ++i) {
        for (long long j = 0; j < 3; ++j)
          bodies[(i_begin + i) * 3 + j] = bodies_copy[sort_idx[i] * 3 + j];
        indices[i_begin + i] = indices_copy[sort_idx[i]];
      }
      long long len = (offset << 1) + (i << 1);
      //std::cout<<"Child 0 "<<len<<std::endl;
      Cell& c0 = cells[len];
      Cell& c1 = cells[len + 1];
      ci.Child[0] = len;
      ci.Child[1] = len + 2;

      long long loc = i_begin + (i_end - i_begin) / 2;
      c0.Body[0] = i_begin + bodies_offset;
      c0.Body[1] = loc + bodies_offset;
      c1.Body[0] = loc + bodies_offset;
      c1.Body[1] = i_end + bodies_offset;

      get_bounds(&bodies[i_begin * 3], loc - i_begin, c0.R.data(), c0.C.data());
      get_bounds(&bodies[loc * 3], i_end - loc, c1.R.data(), c1.C.data());
    }
  }
}


void buildBinaryTree2(Cell* cells, double* bodies, long long* indices, long long nbodies, long long levels) {
  cells[0].Body[0] = 0;
  cells[0].Body[1] = nbodies;
  get_bounds2(bodies, nbodies, cells[0].R.data(), cells[0].C.data());

  long long nleaf = (long long)1 << levels;
  for (long long i = 0; i < nleaf - 1; i++) {
    Cell& ci = cells[i];
    long long sdim = std::distance(ci.R.begin(), std::max_element(ci.R.begin(), ci.R.end()));
    long long i_begin = ci.Body[0];
    long long i_end = ci.Body[1];

    long long i_num = i_end - i_begin;
    std::vector<long long> sort_idx(i_num);
    std::iota(sort_idx.begin(), sort_idx.end(), 0);
    std::array<double, 2>* bodies3 = reinterpret_cast<std::array<double, 2>*>(&bodies[i_begin * 2]);
    std::array<double, 2>* bodies3_end = reinterpret_cast<std::array<double, 2>*>(&bodies[i_end * 2]);
    std::sort(sort_idx.begin(), sort_idx.end(), 
      [&](size_t i, size_t j) { return bodies3[i][sdim] < bodies3[j][sdim]; });

    std::vector<double> bodies_copy(i_num * 2);
    std::vector<long long> indices_copy(i_num);
    std::memcpy(bodies_copy.data(), &bodies[i_begin * 2], sizeof(double) * i_num * 2);
    std::memcpy(indices_copy.data(), &indices[i_begin], sizeof(long long) * i_num);
    for (long long i = 0; i < i_num; ++i) {
      for (long long j = 0; j < 2; ++j)
        bodies[(i_begin + i) * 2 + j] = bodies_copy[sort_idx[i] * 2 + j];
      indices[i_begin + i] = indices_copy[sort_idx[i]];
    }

    long long len = (i << 1) + 1;
    Cell& c0 = cells[len];
    Cell& c1 = cells[len + 1];
    ci.Child[0] = len;
    ci.Child[1] = len + 2;

    long long loc = i_begin + (i_end - i_begin) / 2;
    c0.Body[0] = i_begin;
    c0.Body[1] = loc;
    c1.Body[0] = loc;
    c1.Body[1] = i_end;

    get_bounds2(&bodies[i_begin * 2], loc - i_begin, c0.R.data(), c0.C.data());
    get_bounds2(&bodies[loc * 2], i_end - loc, c1.R.data(), c1.C.data());
  }
}

void buildBinaryTree3(Cell* cells, double* bodies, long long* indices, long long nbodies, long long levels, long long start, long long bodies_offset) {
  cells[start].Body[0] = bodies_offset;
  cells[start].Body[1] = bodies_offset + nbodies;
  get_bounds(bodies, nbodies, cells[start].R.data(), cells[start].C.data());

  long long nleaf = (long long)1 << levels;
  long long offset = 0;
  for (long long level = 0; level < levels; ++ level) {
    //std::cout<<"level "<<level<<std::endl;
    offset = level ? offset * 2 + 2 : start;
    for (long long i = 0; i < (1 << level); i++) {
      //std::cout<<"i "<<i<<std::endl;
      //long long offset = start << level;
      //std::cout<<"cell "<<offset+i<<std::endl;
      Cell& ci = cells[offset + i];
      long long sdim = std::distance(ci.R.begin(), std::max_element(ci.R.begin(), ci.R.end()));
      long long i_begin = ci.Body[0] - bodies_offset;
      long long i_end = ci.Body[1] - bodies_offset;

      long long i_num = i_end - i_begin;
      std::vector<long long> sort_idx(i_num);
      std::iota(sort_idx.begin(), sort_idx.end(), 0);
      std::array<double, 3>* bodies3 = reinterpret_cast<std::array<double, 3>*>(&bodies[i_begin * 3]);
      std::array<double, 3>* bodies3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[i_end * 3]);
      std::sort(sort_idx.begin(), sort_idx.end(), 
        [&](size_t i, size_t j) { return bodies3[i][sdim] < bodies3[j][sdim]; });

      std::vector<double> bodies_copy(i_num * 3);
      std::vector<long long> indices_copy(i_num);
      std::memcpy(bodies_copy.data(), &bodies[i_begin * 3], sizeof(double) * i_num * 3);
      std::memcpy(indices_copy.data(), &indices[i_begin], sizeof(long long) * i_num);
      for (long long i = 0; i < i_num; ++i) {
        for (long long j = 0; j < 3; ++j)
          bodies[(i_begin + i) * 3 + j] = bodies_copy[sort_idx[i] * 3 + j];
        indices[i_begin + i] = indices_copy[sort_idx[i]];
      }
      long long len = (offset + i) * 2 + 2;
      //std::cout<<"Child 0 "<<len<<std::endl;
      Cell& c0 = cells[len];
      Cell& c1 = cells[len + 1];
      ci.Child[0] = len;
      ci.Child[1] = len + 2;

      long long loc = i_begin + (i_end - i_begin) / 2;
      c0.Body[0] = i_begin + bodies_offset;
      c0.Body[1] = loc + bodies_offset;
      c1.Body[0] = loc + bodies_offset;
      c1.Body[1] = i_end + bodies_offset;

      get_bounds(&bodies[i_begin * 3], loc - i_begin, c0.R.data(), c0.C.data());
      get_bounds(&bodies[loc * 3], i_end - loc, c1.R.data(), c1.C.data());
    }
  }
}