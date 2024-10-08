
#include <build_tree.hpp>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <set>

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
