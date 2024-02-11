
#include <build_tree.hpp>
#include <basis.hpp>
#include <comm.hpp>

#include <cmath>
#include <algorithm>
#include <array>
#include <set>

void get_bounds(const double* bodies, int64_t nbodies, double R[], double C[]) {
  double Xmin[3];
  double Xmax[3];
  Xmin[0] = Xmax[0] = bodies[0];
  Xmin[1] = Xmax[1] = bodies[1];
  Xmin[2] = Xmax[2] = bodies[2];

  for (int64_t i = 1; i < nbodies; i++) {
    const double* x_bi = &bodies[i * 3];
    Xmin[0] = fmin(x_bi[0], Xmin[0]);
    Xmin[1] = fmin(x_bi[1], Xmin[1]);
    Xmin[2] = fmin(x_bi[2], Xmin[2]);

    Xmax[0] = fmax(x_bi[0], Xmax[0]);
    Xmax[1] = fmax(x_bi[1], Xmax[1]);
    Xmax[2] = fmax(x_bi[2], Xmax[2]);
  }

  C[0] = (Xmin[0] + Xmax[0]) / 2.;
  C[1] = (Xmin[1] + Xmax[1]) / 2.;
  C[2] = (Xmin[2] + Xmax[2]) / 2.;

  double d0 = Xmax[0] - Xmin[0];
  double d1 = Xmax[1] - Xmin[1];
  double d2 = Xmax[2] - Xmin[2];

  R[0] = (d0 == 0. && Xmin[0] == 0.) ? 0. : (1.e-8 + d0 / 2.);
  R[1] = (d1 == 0. && Xmin[1] == 0.) ? 0. : (1.e-8 + d1 / 2.);
  R[2] = (d2 == 0. && Xmin[2] == 0.) ? 0. : (1.e-8 + d2 / 2.);
}

void sort_bodies(double* bodies, int64_t nbodies, int64_t sdim) {
  std::array<double, 3>* bodies3 = reinterpret_cast<std::array<double, 3>*>(bodies);
  std::array<double, 3>* bodies3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[3 * nbodies]);
  std::sort(bodies3, bodies3_end, [=](std::array<double, 3>& i, std::array<double, 3>& j)->bool {
    double x = i[sdim];
    double y = j[sdim];
    return x < y;
  });
}

int admis_check(double theta, const double C1[], const double C2[], const double R1[], const double R2[]) {
  double dCi[3];
  dCi[0] = C1[0] - C2[0];
  dCi[1] = C1[1] - C2[1];
  dCi[2] = C1[2] - C2[2];

  dCi[0] = dCi[0] * dCi[0];
  dCi[1] = dCi[1] * dCi[1];
  dCi[2] = dCi[2] * dCi[2];

  double dRi[3];
  dRi[0] = R1[0] * R1[0];
  dRi[1] = R1[1] * R1[1];
  dRi[2] = R1[2] * R1[2];

  double dRj[3];
  dRj[0] = R2[0] * R2[0];
  dRj[1] = R2[1] * R2[1];
  dRj[2] = R2[2] * R2[2];

  double dC = dCi[0] + dCi[1] + dCi[2];
  double dR = (dRi[0] + dRi[1] + dRi[2] + dRj[0] + dRj[1] + dRj[2]) * theta;
  return (int)(dC > dR);
}

void buildTree(Cell* cells, double* bodies, int64_t nbodies, int64_t levels) {
  Cell* root = &cells[0];
  root->Body[0] = 0;
  root->Body[1] = nbodies;
  get_bounds(bodies, nbodies, root->R, root->C);

  int64_t nleaf = (int64_t)1 << levels;
  int64_t len = 1;
  for (int64_t i = 0; i < nleaf - 1; i++) {
    Cell* ci = &cells[i];
    ci->Child[0] = -1;
    ci->Child[1] = -1;

    int64_t sdim = 0;
    double maxR = ci->R[0];
    if (ci->R[1] > maxR)
    { sdim = 1; maxR = ci->R[1]; }
    if (ci->R[2] > maxR)
    { sdim = 2; maxR = ci->R[2]; }

    int64_t i_begin = ci->Body[0];
    int64_t i_end = ci->Body[1];
    int64_t nbody_i = i_end - i_begin;
    sort_bodies(&bodies[i_begin * 3], nbody_i, sdim);
    int64_t loc = i_begin + nbody_i / 2;

    Cell* c0 = &cells[len];
    Cell* c1 = &cells[len + 1];
    ci->Child[0] = len;
    ci->Child[1] = len + 2;
    len = len + 2;

    c0->Body[0] = i_begin;
    c0->Body[1] = loc;
    c1->Body[0] = loc;
    c1->Body[1] = i_end;

    get_bounds(&bodies[i_begin * 3], loc - i_begin, c0->R, c0->C);
    get_bounds(&bodies[loc * 3], i_end - loc, c1->R, c1->C);
  }
}

void buildTreeBuckets(Cell* cells, const double* bodies, const int64_t buckets[], int64_t levels) {
  int64_t nleaf = (int64_t)1 << levels;
  int64_t count = 0;
  for (int64_t i = 0; i < nleaf; i++) {
    int64_t ci = i + nleaf - 1;
    cells[ci].Child[0] = -1;
    cells[ci].Child[1] = -1;
    cells[ci].Body[0] = count;
    cells[ci].Body[1] = count + buckets[i];
    get_bounds(&bodies[count * 3], buckets[i], cells[ci].R, cells[ci].C);
    count = count + buckets[i];
  }

  for (int64_t i = nleaf - 2; i >= 0; i--) {
    int64_t c0 = (i << 1) + 1;
    int64_t c1 = (i << 1) + 2;
    int64_t begin = cells[c0].Body[0];
    int64_t len = cells[c1].Body[1] - begin;
    cells[i].Child[0] = c0;
    cells[i].Child[1] = c0 + 2;
    cells[i].Body[0] = begin;
    cells[i].Body[1] = begin + len;
    get_bounds(&bodies[begin * 3], len, cells[i].R, cells[i].C);
  }
}

void getList(char NoF, std::vector<std::pair<int64_t, int64_t>>& rels, const Cell cells[], int64_t i, int64_t j, double theta) {
  int admis = admis_check(theta, cells[i].C, cells[j].C, cells[i].R, cells[j].R);
  int write_far = NoF == 'F' || NoF == 'f';
  int write_near = NoF == 'N' || NoF == 'n';
  if (admis ? write_far : write_near)
    rels.emplace_back(i, j);
  
  if (!admis && cells[i].Child[0] >= 0 && cells[j].Child[0] >= 0)
    for (int64_t k = cells[i].Child[0]; k < cells[i].Child[1]; k++)
      for (int64_t l = cells[j].Child[0]; l < cells[j].Child[1]; l++)
        getList(NoF, rels, cells, k, l, theta);
}

Cell::Cell() {
  Child[0] = Child[1] = -1;
  Body[0] = Body[1] = -1;
  C[0] = C[1] = C[2] = 0.;
  R[0] = R[1] = R[2] = 0.;
}

CSR::CSR(char NoF, int64_t ncells, const Cell* cells, double theta) {
  std::vector<std::pair<int64_t, int64_t>> LIL;
  getList(NoF, LIL, cells, 0, 0, theta);
  std::sort(LIL.begin(), LIL.end());

  int64_t len = LIL.size();
  RowIndex.resize(ncells + 1);
  ColIndex.resize(len);
  std::transform(LIL.begin(), LIL.end(), ColIndex.begin(), 
    [](const std::pair<int64_t, int64_t>& i) { return i.second; });

  RowIndex[0] = 0;
  for (int64_t n = 1; n <= ncells; n++)
    RowIndex[n] = std::distance(LIL.begin(), 
      std::find_if(LIL.begin() + RowIndex[n - 1], LIL.end(), 
        [=](const std::pair<int64_t, int64_t>& i) { return n <= i.first; }));
}

CSR::CSR(const CSR& A, const CSR& B) {
  int64_t M = A.RowIndex.size() - 1;
  RowIndex.resize(M + 1);
  ColIndex.clear();
  RowIndex[0] = 0;

  for (int64_t y = 0; y < M; y++) {
    std::set<int64_t> cols;
    std::for_each(&A.ColIndex[A.RowIndex[y]], &A.ColIndex[A.RowIndex[y + 1]], [&](int64_t x) {
      std::for_each(&B.ColIndex[B.RowIndex[x]], &B.ColIndex[B.RowIndex[x + 1]], [&](int64_t z) {
        cols.insert(z);
      });
    });

    for (std::set<int64_t>::iterator i = cols.begin(); i != cols.end(); i = std::next(i))
      ColIndex.push_back(*i);
    RowIndex[y + 1] = (int64_t)ColIndex.size();
  }
}
