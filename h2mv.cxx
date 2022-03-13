
#include "h2mv.hxx"

#include <cmath>

using namespace nbd;

void nbd::upwardPassLeaf(const Cells& cells, const Matrices& base, const double* x, double* m) {
  int64_t len = cells.size();
  int64_t x_off = 0;
  for (int64_t i = 0; i < len; i++) {
    const Cell& c = cells[i];
    if (c.NCHILD == 0) {
      P2M(&c, base[i], &x[x_off], m);
      x_off += c.NBODY;
    }
  }
}

void nbd::downwardPassLeaf(const Cells& cells, const Matrices& base, const double* l, double* b) {
  int64_t len = cells.size();
  int64_t b_off = 0;
  for (int64_t i = 0; i < len; i++) {
    const Cell& c = cells[i];
    if (c.NCHILD == 0) {
      L2P(&c, base[i], l, &b[b_off]);
      b_off += c.NBODY;
    }
  }
}


void nbd::horizontalPass(EvalFunc ef, const Cells& cells, int64_t dim, int64_t level, const double* m, double* l) {
  int64_t range = (int64_t)1 << level;
  std::vector<int64_t> offs(range);
  range = 0;
  findCellsAtLevel(&offs[0], &range, &cells[0], &cells[0], level);

  for (int64_t i = 0; i < range; i++) {
    int64_t ii = offs[i];
    const Cell& ci = cells[ii];
    int64_t lislen = ci.listFar.size();
    for (int64_t j = 0; j < lislen; j++) {
      const Cell* cj = ci.listFar[j];
      M2L(ef, &ci, cj, dim, m, l);
    }
  }
}

void nbd::upwardPass(const Cells& cells, int64_t level, const Matrices& base, double* m) {
  int64_t range = (int64_t)1 << level;
  std::vector<int64_t> offs(range);
  range = 0;
  findCellsAtLevel(&offs[0], &range, &cells[0], &cells[0], level);

  for (int64_t i = 0; i < range; i++) {
    int64_t ii = offs[i];
    M2M(&cells[ii], base[ii], m);
  }
}

void nbd::downwardPass(const Cells& cells, int64_t level, const Matrices& base, double* l) {
  int64_t range = (int64_t)1 << level;
  std::vector<int64_t> offs(range);
  range = 0;
  findCellsAtLevel(&offs[0], &range, &cells[0], &cells[0], level);

  for (int64_t i = 0; i < range; i++) {
    int64_t ii = offs[i];
    L2L(&cells[ii], base[ii], l);
  }
}

void nbd::closeQuarter(EvalFunc ef, const Cells& cells, int64_t dim, const double* x, double* b) {
  const Body* begin = cells[0].BODY;
  int64_t len = cells.size();
  for (int64_t y = 0; y < len; y++) {
    const Cell& ci = cells[y];
    int64_t yi = ci.BODY - begin;
    if (ci.NCHILD == 0) {
      int64_t len_ci = ci.listNear.size();
      for (int64_t j = 0; j < len_ci; j++) {
        const Cell* cj = ci.listNear[j];
        int64_t xi = cj->BODY - begin;
        P2P(ef, &ci, cj, dim, &x[xi], &b[yi]);
      }
    }
  }
}

void nbd::zeroC(const Cell* cell, double* c, int64_t lmin, int64_t lmax) {
  if (cell->LEVEL <= lmax) {
    double* begin = &c[cell->MPOS];
    int64_t len = cell->Multipole.size();
    if (cell->LEVEL >= lmin)
      std::fill(begin, begin + len, 0.);

    for (int64_t i = 0; i < cell->NCHILD; i++)
      zeroC(cell->CHILD + i, c, lmin, lmax);
  }
}

void nbd::multiplyC(EvalFunc ef, const Cells& cells, int64_t dim, int64_t level, const Matrices& base, double* m, double* l) {
  zeroC(&cells[0], m, 0, level - 1);
  zeroC(&cells[0], l, 0, level);
  horizontalPass(ef, cells, dim, level, m, l);

  for (int64_t i = level - 1; i >= 0; i--) {
    upwardPass(cells, i, base, m);
    horizontalPass(ef, cells, dim, i, m, l);
  }

  for (int64_t i = 0; i < level; i++)
    downwardPass(cells, i, base, l);
}


void nbd::h2mv_complete(EvalFunc ef, const Cells& cells, int64_t dim, const Matrices& base, const double* x, double* b) {
  int64_t len = cells.size();
  int64_t level = 0;
  int64_t lm = 0;
  for (int64_t i = 0; i < len; i++) {
    level = level < cells[i].LEVEL ? cells[i].LEVEL : level;
    lm = lm + cells[i].Multipole.size();
  }

  std::vector<double> m(lm);
  std::vector<double> l(lm);

  upwardPassLeaf(cells, base, x, &m[0]);
  multiplyC(ef, cells, dim, level, base, &m[0], &l[0]);
  downwardPassLeaf(cells, base, &l[0], b);
  closeQuarter(ef, cells, dim, x, b);
}

int64_t nbd::C2X(const Cells& cells, int64_t level, const double* c, double* x) {
  int64_t range = (int64_t)1 << level;
  std::vector<int64_t> offs(range);
  range = 0;
  findCellsAtLevel(&offs[0], &range, &cells[0], &cells[0], level);

  int64_t len = 0;
  for (int64_t i = 0; i < range; i++) {
    int64_t ii = offs[i];
    const Cell* ci = &cells[ii];
    M2X(ci, c, &x[len]);
    int64_t n = ci->Multipole.size();
    len = len + n;
  }
  return len;
}

int64_t nbd::X2C(const Cells& cells, int64_t level, const double* x, double* c) {
  int64_t range = (int64_t)1 << level;
  std::vector<int64_t> offs(range);
  range = 0;
  findCellsAtLevel(&offs[0], &range, &cells[0], &cells[0], level);

  int64_t len = 0;
  for (int64_t i = 0; i < range; i++) {
    int64_t ii = offs[i];
    const Cell* ci = &cells[ii];
    X2M(ci, &x[len], c);
    int64_t n = ci->Multipole.size();
    len = len + n;
  }
  return len;
}

void woodburyP1(EvalFunc ef, const Cells& cells, int64_t dim, int64_t level, const Matrices& base, double* x1, double* x2, double* c1, double* c2) {
  int64_t len = cells.size();
  int64_t level_m = (int64_t)std::log2(len);

  if (level == level_m)
    upwardPassLeaf(cells, base, x1, c1);
  else {
    X2C(cells, level + 1, x1, c1);
    zeroC(&cells[0], c1, level, level);
    upwardPass(cells, level, base, c1);
  }

  multiplyC(ef, cells, dim, level, base, c1, c2);
  C2X(cells, level, c1, x1);
  C2X(cells, level, c2, x2);
}

void woodburyP2(EvalFunc ef, const Cells& cells, int64_t dim, int64_t level, const Matrices& base, double* x1, double* x2, double* c1, double* c2) {
  int64_t len = cells.size();
  int64_t level_m = (int64_t)std::log2(len);

  X2C(cells, level, x1, c1);
  multiplyC(ef, cells, dim, level, base, c1, c2);

  int64_t lenx;
  if (level == level_m) {
    downwardPassLeaf(cells, base, c2, x1);
    lenx = cells[0].NBODY;
  }
  else {
    zeroC(&cells[0], c2, level + 1, level + 1);
    downwardPass(cells, level, base, c2);
    lenx = C2X(cells, level + 1, c2, x1);
  }
  for (int64_t i = 0; i < lenx; i++)
    x2[i] = x2[i] - x1[i];
}


void h2inv_step(EvalFunc ef, const Cells& cells, int64_t dim, int64_t level, const Matrices& base, const Matrices d[], const CSC rels[], double* x1, double* x2, double* c1, double* c2) {
  /*Matrix d_dense;
  convertSparse2Dense(rels[level], d[level], d_dense);
  factorD(d_dense);

  int64_t range = (int64_t)1 << level;
  std::vector<int64_t> offs(range);
  range = 0;
  findCellsAtLevel(&offs[0], &range, &cells[0], &cells[0], level);
  int64_t len = 0;
  for (int64_t i = 0; i < range; i++) {
    int64_t ii = offs[i];
    len = len + cells[ii].Multipole.size();
  }

  std::vector<double> x3(len);
  std::vector<double> x4(len);

  invD(d_dense, x2);
  woodburyP1(ef, cells, dim, level, base, x2, &x3[0], c1, c2);
  std::copy(x3.begin(), x3.end(), x4.begin());

  if (level > 1)
    h2inv_step(ef, cells, dim, level - 1, base, d, rels, &x3[0], &x4[0], c1, c2);
  else {
    Matrix last;
    convertSparse2Dense(rels[0], d[0], last);
    factorD(last);
    invD(last, &x3[0]);
  }

  for (int64_t i = 0; i < len; i++)
    x2[i] = x2[i] - x3[i];
  woodburyP2(ef, cells, dim, level, base, x2, x1, c1, c2);
  invD(d_dense, x1);*/
}

void nbd::h2inv(EvalFunc ef, const Cells& cells, int64_t dim, const Matrices& base, const Matrices d[], const CSC rels[], double* x) {
  int64_t nbodies = cells[0].NBODY;
  std::vector<double> x2(nbodies);
  std::copy(x, x + nbodies, x2.begin());

  int64_t lm = cells.back().MPOS + cells.back().Multipole.size();
  std::vector<double> c1(lm, 0);
  std::vector<double> c2(lm, 0);

  int64_t len = cells.size();
  int64_t level_m = (int64_t)std::log2(len);
  h2inv_step(ef, cells, dim, level_m, base, d, rels, x, &x2[0], &c1[0], &c2[0]);
}