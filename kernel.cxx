
#include "kernel.hxx"
#include "build_tree.hxx"

#include <minblas.h>
#include <cmath>
#include <random>

using namespace nbd;

EvalFunc nbd::r2() {
  EvalFunc ef;
  ef.r2f = [](double& r2, double singularity, double alpha) -> void {};
  ef.singularity = 0.;
  ef.alpha = 1.;
  return ef;
}

EvalFunc nbd::l2d() {
  EvalFunc ef;
  ef.r2f = [](double& r2, double singularity, double alpha) -> void {
    r2 = r2 == 0 ? singularity : std::log(std::sqrt(r2));
  };
  ef.singularity = 1.e6;
  ef.alpha = 1.;
  return ef;
}

EvalFunc nbd::l3d() {
  EvalFunc ef;
  ef.r2f = [](double& r2, double singularity, double alpha) -> void {
    r2 = r2 == 0 ? singularity : 1. / std::sqrt(r2);
  };
  ef.singularity = 1.e6;
  ef.alpha = 1.;
  return ef;
}


void nbd::eval(EvalFunc ef, const Body* bi, const Body* bj, int64_t dim, double* out) {
  double& r2 = *out;
  r2 = 0.;
  for (int64_t i = 0; i < dim; i++) {
    double dX = bi->X[i] - bj->X[i];
    r2 += dX * dX;
  }
  ef.r2f(r2, ef.singularity, ef.alpha);
}


void nbd::P2P(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, const Vector& X, Vector& B) {
  int64_t m = ci->NBODY;
  int64_t n = cj->NBODY;
  const double* x = &X.X[0];
  double* b = &B.X[0];

  for (int64_t i = 0; i < m; i++) {
    double sum = b[i];
    for (int64_t j = 0; j < n; j++) {
      double r2;
      eval(ef, ci->BODY + i, cj->BODY + j, dim, &r2);
      sum += r2 * x[j];
    }
    b[i] = sum;
  }
}


void nbd::P2Pmat(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, Matrix& a) {
  int64_t m = ci->NBODY, n = cj->NBODY;
  a.A.resize(m * n);
  a.M = m;
  a.N = n;

  for (int64_t i = 0; i < m * n; i++) {
    int64_t x = i / m;
    int64_t y = i - x * m;
    double r2;
    eval(ef, ci->BODY + y, cj->BODY + x, dim, &r2);
    a.A[x * a.M + y] = r2;
  }
}

void nbd::M2L(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, const double m[], double l[]) {
  const std::vector<int64_t>& mi = ci->Multipole;
  const std::vector<int64_t>& mj = cj->Multipole;
  int64_t y = mi.size();
  int64_t x = mj.size();

  for (int64_t i = 0; i < y; i++) {
    int64_t bi = mi[i];
    double sum = l[i];
    for (int64_t j = 0; j < x; j++) {
      double r2;
      int64_t bj = mj[j];
      eval(ef, ci->BODY + bi, cj->BODY + bj, dim, &r2);
      sum += r2 * m[j];
    }
    l[i] = sum;
  }
}

void nbd::M2Lc(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, const Vector& M, Vector& L) {
  int64_t off_m = 0;
  for (int64_t j = 0; j < cj->NCHILD; j++) {
    const Cell* ccj = cj->CHILD + j;
    int64_t off_l = 0;
    for (int64_t i = 0; i < ci->NCHILD; i++) {
      const Cell* cci = ci->CHILD + i;
      int64_t len_far = cci->listFar.size();
      for (int64_t k = 0; k < len_far; k++)
        if (cci->listFar[k] == ccj)
          M2L(ef, cci, ccj, dim, &M.X[off_m], &L.X[off_l]);

      off_l = off_l + cci->Multipole.size();
    }
    off_m = off_m + ccj->Multipole.size();
  }
}

void nbd::M2Lmat(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, Matrix& a) {
  const std::vector<int64_t>& mi = ci->Multipole;
  const std::vector<int64_t>& mj = cj->Multipole;
  int64_t m = mi.size();
  int64_t n = mj.size();
  a.A.resize(m * n);
  a.M = m;
  a.N = n;

  for (int64_t i = 0; i < m * n; i++) {
    int64_t x = i / m;
    int64_t bx = mj[x];
    int64_t y = i - x * m;
    int64_t by = mi[y];

    double r2;
    eval(ef, ci->BODY + by, cj->BODY + bx, dim, &r2);
    a.A[x * a.M + y] = r2;
  }
}

void childMultipoles(Bodies& multipole, Cell& cell, int64_t dim) {
  if (cell.NCHILD > 0) {
    int64_t size = 0;
    int64_t count = 0;
    for (int64_t i = 0; i < cell.NCHILD; i++)
      size += cell.CHILD[i].Multipole.size();
    multipole.resize(size);
    std::vector<int64_t>& cellm = cell.cMultipoles;
    cellm.resize(size);

    for (int64_t i = 0; i < cell.NCHILD; i++) {
      const Cell& c = cell.CHILD[i];
      int64_t loc = c.BODY - cell.BODY;
      int64_t len = c.Multipole.size();
      for (int64_t n = 0; n < len; n++) {
        int64_t nloc = loc + c.Multipole[n];
        for (int64_t d = 0; d < dim; d++)
          multipole[count].X[d] = cell.BODY[nloc].X[d];
        multipole[count].B = cell.BODY[nloc].B;
        cellm[count] = nloc;
        count += 1;
      }
    }
  }
  else {
    multipole.resize(cell.NBODY);
    for (int64_t i = 0; i < cell.NBODY; i++) {
      for (int64_t d = 0; d < dim; d++)
        multipole[i].X[d] = cell.BODY[i].X[d];
      multipole[i].B = cell.BODY[i].B;
    }
  }
}

void selectMultipole(Cell& cell, const int64_t arows[], int64_t rank) {
  if (cell.Multipole.size() != rank)
    cell.Multipole.resize(rank);
  if (cell.NCHILD > 0)
    for (int64_t i = 0; i < rank; i++) {
      int64_t ai = arows[i];
      cell.Multipole[i] = cell.cMultipoles[ai];
    }
  else
    for (int64_t i = 0; i < rank; i++)
      cell.Multipole[i] = arows[i];
}

void nbd::P2Mmat(EvalFunc ef, Cell* ci, const Body rm[], int64_t n, int64_t dim, Matrix& u, double epi, int64_t rank) {
  Bodies src;
  childMultipoles(src, *ci, dim);

  int64_t m = src.size();
  if (m > 0 && n > 0) {
    Matrix a;
    std::vector<int64_t> pa(rank);
    cMatrix(a, m, n);
    cMatrix(u, m, rank);

    for (int64_t i = 0; i < m * n; i++) {
      int64_t x = i / m;
      int64_t y = i - x * m;
      double r2;
      eval(ef, &src[y], &rm[x], dim, &r2);
      a.A[y + x * m] = r2;
    }

    int64_t iters;
    didrow(epi, m, n, rank, &a.A[0], m, &u.A[0], m, &pa[0], &iters);
    selectMultipole(*ci, &pa[0], iters);

    if (iters != rank)
      cMatrix(u, m, iters);
  }
  else {
    ci->Multipole.resize(ci->cMultipoles.size());
    std::copy(ci->cMultipoles.begin(), ci->cMultipoles.end(), ci->Multipole.begin());
  }
}

void nbd::invBasis(const Matrix& u, Matrix& uinv) {
  int64_t m = u.M;
  int64_t n = u.N;
  if (m > 0 && n > 0) {
    std::vector<double> a(n * n);
    std::vector<double> q(n * n);

    uinv.A.resize(n * m);
    uinv.M = n;
    uinv.N = m;

    dgemm('T', 'N', n, n, m, 1., &u.A[0], m, &u.A[0], m, 0., &a[0], n);
    dorth('F', n, n, &a[0], n, &q[0], n);
    dgemm('T', 'T', n, m, n, 1., &q[0], n, &u.A[0], m, 0., &uinv.A[0], n);
    dtrsmr_left(n, m, &a[0], n, &uinv.A[0], n);
  }
}

void nbd::D2C(const Matrix& d, const Matrix& u, const Matrix& v, Matrix& c, int64_t y, int64_t x) {
  int64_t m = d.M;
  int64_t n = d.N;
  int64_t rm = u.M;
  int64_t rn = v.M;
  int64_t ldc = c.M;
  double* ptc = &c.A[y + x * ldc];

  if (rm > 0 && rn > 0) {
    std::vector<double> work(m * rn);
    dgemm('N', 'T', m, rn, n, 1., &d.A[0], m, &v.A[0], rn, 0., &work[0], m);
    dgemm('N', 'N', rm, rn, m, 1., &u.A[0], rm, &work[0], m, 0., ptc, ldc);
  }
  else
    for (int64_t i = 0; i < n; i++)
      dcopy(m, &d.A[i * m], 1, &ptc[i * ldc], 1);
}

void nbd::L2C(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, Matrix& c, int64_t y, int64_t x) {
  const std::vector<int64_t>& mi = ci->Multipole;
  const std::vector<int64_t>& mj = cj->Multipole;
  int64_t m = mi.size();
  int64_t n = mj.size();
  int64_t ldc = c.M;
  double* ptc = &c.A[y + x * ldc];

  for (int64_t i = 0; i < m * n; i++) {
    int64_t x = i / m;
    int64_t bx = mj[x];
    int64_t y = i - x * m;
    int64_t by = mi[y];

    double r2;
    eval(ef, ci->BODY + by, cj->BODY + bx, dim, &r2);
    ptc[y + x * ldc] = r2;
  }
}

