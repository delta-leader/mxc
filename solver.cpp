
#include <solver.hpp>
#include <basis.hpp>
#include <build_tree.hpp>
#include <comm.hpp>
#include <kernel.hpp>

#include <algorithm>
#include <numeric>
#include <set>

Solver::Solver(const int64_t dims[], const CSR& Near, const CellComm& comm) {
  int64_t xlen = comm.lenNeighbors();
  Dims = std::vector<int64_t>(xlen);
  DimsLr = std::vector<int64_t>(xlen, 0);
  std::copy(dims, &dims[xlen], Dims.begin());

  std::vector<int64_t> global_y(xlen);
  std::for_each(global_y.begin(), global_y.end(), 
    [&](int64_t& y) { int64_t i = std::distance(&global_y[0], &y); y = comm.iGlobal(i); });

  Q = std::vector<std::complex<double>*>(xlen);
  R = std::vector<std::complex<double>*>(xlen);

  std::vector<int64_t> alen(xlen), offset_a(xlen + 1);
  std::transform(dims, &dims[xlen], alen.begin(), [&](int64_t d) { return d * d; });
  std::inclusive_scan(alen.begin(), alen.end(), offset_a.begin() + 1);
  offset_a[0] = 0;

  Qdata = std::vector<std::complex<double>>(offset_a[xlen], std::complex<double>(0., 0.));
  Rdata = std::vector<std::complex<double>>(offset_a[xlen], std::complex<double>(0., 0.));
  std::transform(offset_a.begin(), offset_a.begin() + xlen, Q.begin(), [&](int64_t i) { return &Qdata[i]; });
  std::transform(offset_a.begin(), offset_a.begin() + xlen, R.begin(), [&](int64_t i) { return &Rdata[i]; });

  std::vector<int64_t> ylen(xlen), offset_y(xlen + 1);
  std::transform(global_y.begin(), global_y.end(), ylen.begin(), 
    [&](int64_t y) { return Near.RowIndex[y + 1] - Near.RowIndex[y]; });
  std::inclusive_scan(ylen.begin(), ylen.end(), offset_y.begin() + 1);
  offset_y[0] = 0;

  A = std::vector<std::complex<double>*>(offset_y[xlen]);
  X = std::vector<int64_t>(offset_y[xlen]);
  Y = std::vector<int64_t>(offset_y[xlen]);
  std::set<std::pair<int64_t, int64_t>> fill_locs;

  for (int64_t i = 0; i < xlen; i++) {
    int64_t y = global_y[i];
    std::fill(&Y[offset_y[i]], &Y[offset_y[i + 1]], i);
    std::transform(&Near.ColIndex[Near.RowIndex[y]], &Near.ColIndex[Near.RowIndex[y + 1]], &X[offset_y[i]],
      [&](int64_t x) { return comm.iLocal(x); });
    std::for_each(&Near.ColIndex[Near.RowIndex[y]], &Near.ColIndex[Near.RowIndex[y + 1]], [&](int64_t x1) {
      std::for_each(&Near.ColIndex[Near.RowIndex[y]], &Near.ColIndex[Near.RowIndex[y + 1]], [&](int64_t x2) {
        if(std::find(&Near.ColIndex[Near.RowIndex[x1]], &Near.ColIndex[Near.RowIndex[x1 + 1]], x2) == &Near.ColIndex[Near.RowIndex[x1 + 1]])
          fill_locs.insert(std::make_pair(x1, x2));
      });
    });
  }

  alen.resize(offset_y[xlen]);
  offset_a.resize(offset_y[xlen] + 1);
  std::transform(Y.begin(), Y.end(), X.begin(), alen.begin(), [&](int64_t y, int64_t x) { return Dims[y] * Dims[x]; });
  std::inclusive_scan(alen.begin(), alen.end(), offset_a.begin() + 1);
  offset_a[0] = 0;

  Adata = std::vector<std::complex<double>>(offset_a[offset_y[xlen]], std::complex<double>(0., 0.));
  std::transform(offset_a.begin(), offset_a.begin() + offset_y[xlen], A.begin(), [&](int64_t i) { return &Adata[i]; });

  F = std::vector<std::complex<double>*>(fill_locs.size());
  FX = std::vector<int64_t>(fill_locs.size());
  FY = std::vector<int64_t>(fill_locs.size());
  std::transform(fill_locs.begin(), fill_locs.end(), FX.begin(), [](std::pair<int64_t, int64_t> p) { return p.first; });
  std::transform(fill_locs.begin(), fill_locs.end(), FY.begin(), [](std::pair<int64_t, int64_t> p) { return p.second; });

  alen.resize(fill_locs.size());
  offset_a.resize(fill_locs.size() + 1);
  std::transform(FY.begin(), FY.end(), FX.begin(), alen.begin(), [&](int64_t y, int64_t x) { return Dims[y] * Dims[x]; });
  std::inclusive_scan(alen.begin(), alen.end(), offset_a.begin() + 1);
  offset_a[0] = 0;

  Fdata = std::vector<std::complex<double>>(offset_a[fill_locs.size()], std::complex<double>(0., 0.));
  std::transform(offset_a.begin(), offset_a.begin() + fill_locs.size(), F.begin(), [&](int64_t i) { return &Fdata[i]; });
}

void Solver::setData_leaf(const Eval& eval, const Cell cells[], const double bodies[], const CellComm& comm) {
  for (int64_t i = 0; i < (int64_t)A.size(); i++) {
    int64_t y = Y[i];
    int64_t x = X[i];
    gen_matrix(eval, Dims[y], Dims[x], bodies + 3 * cells[comm.iGlobal(y)].Body[0], bodies + 3 * cells[comm.iGlobal(x)].Body[0], A[i], Dims[y]);
  }
}

void Solver::setData_far(const Eval& eval, const MatVecBasis& basis, const CellComm& comm) {
  
}
