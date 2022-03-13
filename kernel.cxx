
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


void nbd::P2P(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, const double x[], double b[]) {
  int64_t m = ci->NBODY;
  int64_t n = cj->NBODY;

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


void nbd::M2L(EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, const double M[], double L[]) {
  const std::vector<int64_t>& mi = ci->Multipole;
  const std::vector<int64_t>& mj = cj->Multipole;
  int64_t m = mi.size();
  int64_t n = mj.size();
  const double* mm = &M[cj->MPOS];
  double* ll = &L[ci->MPOS];

  for (int64_t i = 0; i < m; i++) {
    int64_t bi = mi[i];
    double sum = ll[i];
    for (int64_t j = 0; j < n; j++) {
      double r2;
      int64_t bj = mj[j];
      eval(ef, ci->BODY + bi, cj->BODY + bj, dim, &r2);
      sum += r2 * mm[j];
    }
    ll[i] = sum;
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
    cell.Multipole.resize(size);

    for (int64_t i = 0; i < cell.NCHILD; i++) {
      const Cell& c = cell.CHILD[i];
      int64_t loc = c.BODY - cell.BODY;
      int64_t len = c.Multipole.size();
      for (int64_t n = 0; n < len; n++) {
        int64_t nloc = loc + c.Multipole[n];
        for (int64_t d = 0; d < dim; d++)
          multipole[count].X[d] = cell.BODY[nloc].X[d];
        cell.Multipole[count] = nloc;
        count += 1;
      }
    }
  }
  else {
    multipole.resize(cell.NBODY);
    for (int64_t i = 0; i < cell.NBODY; i++)
      for (int64_t d = 0; d < dim; d++)
        multipole[i].X[d] = cell.BODY[i].X[d];
  }
}

void selectMultipole(Cell& cell, const int64_t arows[], int64_t rank) {
  std::vector<int64_t> mc(rank);
  if (cell.NCHILD > 0)
    for (int64_t i = 0; i < rank; i++) {
      int64_t ai = arows[i];
      mc[i] = cell.Multipole[ai];
    }
  else
    for (int64_t i = 0; i < rank; i++)
      mc[i] = arows[i];
  if (cell.Multipole.size() != rank)
    cell.Multipole.resize(rank);
  std::copy(mc.begin(), mc.end(), cell.Multipole.begin());
}

void nbd::P2Mmat(EvalFunc ef, Cell* ci, const Bodies& rm, int64_t dim, Matrix& u, double epi, int64_t rank) {
  Bodies src;
  childMultipoles(src, *ci, dim);

  int64_t m = src.size();
  int64_t n = rm.size();
  if (m > 0 && n > 0) {
    std::vector<double> a(m * n);
    std::vector<double> ax(m * rank);
    std::vector<int64_t> pa(rank);

    for (int64_t i = 0; i < m * n; i++) {
      int64_t x = i / m;
      int64_t y = i - x * m;
      double r2;
      eval(ef, &src[y], &rm[x], dim, &r2);
      a[y + x * m] = r2;
    }

    int64_t iters;
    didrow(epi, m, n, rank, &a[0], m, &ax[0], m, &pa[0], &iters);
    selectMultipole(*ci, &pa[0], iters);

    u.A.resize(m * iters);
    u.M = m;
    u.N = iters;
    for (int64_t i = 0; i < m; i++)
      for (int64_t j = 0; j < iters; j++)
        u.A[i + j * m] = ax[i + j * m];
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

void nbd::P2M(const Cell* cell, const Matrix& ba, const double x[], double m[]) {
  int64_t mm = cell->NBODY;
  int64_t n = cell->Multipole.size();
  int64_t m_off = cell->MPOS;
  dgemv('T', mm, n, 1., &ba.A[0], mm, x, 1, 0., &m[m_off], 1);
}

void nbd::M2M(const Cell* cell, const Matrix& ba, double m[]) {
  int64_t n = cell->Multipole.size();
  double* m_parent = &m[cell->MPOS];
  int64_t u_off = 0;
  if (ba.M > 0)
    for (int64_t j = 0; j < cell->NCHILD; j++) {
      const Cell* cj = cell->CHILD + j;
      int64_t jj = cj->ZID;
      int64_t mj = cj->Multipole.size();
      int64_t m_off = cj->MPOS;
      dgemv('T', mj, n, 1., &ba.A[u_off], ba.M, &m[m_off], 1, 1., m_parent, 1);
      u_off += mj;
    }
}

void nbd::L2L(const Cell* cell, const Matrix& ba, double l[]) {
  int64_t n = cell->Multipole.size();
  const double* l_parent = &l[cell->MPOS];
  int64_t u_off = 0;
  if (ba.M > 0)
    for (int64_t j = 0; j < cell->NCHILD; j++) {
      const Cell* cj = cell->CHILD + j;
      int64_t m = cj->Multipole.size();
      int64_t l_off = cj->MPOS;
      dgemv('N', m, n, 1., &ba.A[u_off], ba.M, l_parent, 1, 1., &l[l_off], 1);
      u_off += m;
    }
}

void nbd::L2P(const Cell* cell, const Matrix& ba, const double l[], double b[]) {
  int64_t m = cell->NBODY;
  int64_t n = cell->Multipole.size();
  int64_t l_off = cell->MPOS;
  dgemv('N', m, n, 1., &ba.A[0], m, &l[l_off], 1, 0., b, 1);
}

void nbd::M2X(const Cell* cell, const double m[], double x[]) {
  int64_t n = cell->Multipole.size();
  int64_t m_off = cell->MPOS;
  dcopy(n, &m[m_off], 1, x, 1);
}

void nbd::X2M(const Cell* cell, const double x[], double m[]) {
  int64_t n = cell->Multipole.size();
  int64_t m_off = cell->MPOS;
  dcopy(n, x, 1, &m[m_off], 1);
}

void nbd::factorD(Matrix& a) {
  int64_t n = a.M;
  dpotrf(n, &a.A[0], n);
}

void nbd::invD(const Matrix& a, double* x) {
  int64_t n = a.M;
  dtrsml_left(n, 1, &a.A[0], n, x, n);
  dtrsmlt_left(n, 1, &a.A[0], n, x, n);
}
