
#include "build_tree.hxx"

#include <cmath>
#include <iterator>
#include <random>
#include <numeric>
#include <algorithm>
#include <cstdio>

using namespace nbd;


void nbd::Z_index(int64_t i, int64_t dim, int64_t X[]) {
  std::fill(X, X + dim, 0);
  int64_t iter = 0;
  while(i > 0) {
    for (int64_t d = 0; d < dim; d++) {
      int64_t bit = i & 1;
      i = i >> 1;
      X[d] = X[d] | (bit << iter);
    }
    iter = iter + 1;
  }
}


void nbd::Z_index_i(const int64_t X[], int64_t dim, int64_t& i) {
  std::vector<int64_t> Xi(dim);
  std::copy(X, X + dim, Xi.data());
  std::vector<int64_t>::iterator it = std::find_if(Xi.begin(), Xi.end(), [](int64_t& x){ return x < 0; });
  i = -1;
  if (it == Xi.end()) { 
    i = 0;
    bool run = true;
    int64_t iter = 0;
    while (run) {
      run = false;
      for (int64_t d = 0; d < dim; d++) {
        int64_t bit = Xi[d] & 1;
        Xi[d] = Xi[d] >> 1;
        i = i | (bit << (iter * dim + d));
        run = run || (Xi[d] > 0);
      }
      iter = iter + 1;
    }
  }
}


int64_t nbd::Z_neighbors(int64_t N[], int64_t i, int64_t dim, int64_t max_i, int64_t theta) {
  std::vector<int64_t> Xi(dim);
  std::vector<int64_t> Xj(dim);
  Z_index(i, dim, Xi.data());

  int64_t len = (int64_t)std::floor(std::sqrt(theta));
  int64_t width = len * 2 + 1;
  int64_t ncom = (int64_t)std::pow(width, dim);
  std::vector<int64_t> ks(std::min(ncom, max_i));
  int64_t nk = 0;

  for (int64_t j = 0; j < ncom; j++) {
    Xj[0] = j;
    for (int64_t d = 1; d < dim; d++) {
      Xj[d] = Xj[d - 1] / width;
      Xj[d - 1] = Xj[d - 1] - Xj[d] * width;
    }
    int64_t r2 = 0;
    for (int64_t d = 0; d < dim; d++) {
      Xj[d] = Xj[d] - len;
      r2 = r2 + Xj[d] * Xj[d];
    }
    
    if (r2 <= theta) {
      for (int64_t d = 0; d < dim; d++)
        Xj[d] = Xj[d] + Xi[d];
      int64_t k;
      Z_index_i(Xj.data(), dim, k);
      if (k >= 0 && k < max_i)
        ks[nk++] = k;
    }
  }

  std::sort(ks.begin(), ks.begin() + nk);
  std::copy(ks.begin(), ks.begin() + nk, N);
  return nk;
}

void nbd::Z_neighbors_level(std::vector<int64_t>& ranks, int64_t& self_i, int64_t level, const LocalDomain& domain) {
  if (level >= 0) {
    int64_t lvl_diff = 0;
    int64_t size = (int64_t)1 << domain.MY_LEVEL;
    if (level < domain.MY_LEVEL) {
      lvl_diff = domain.MY_LEVEL - level;
      size = (int64_t)1 << level;
    }
    int64_t my_rank = domain.RANK >> lvl_diff;
    std::vector<int64_t> work(size);
    int64_t len = Z_neighbors(work.data(), my_rank, domain.DIM, size, domain.THETA);
    ranks.resize(len);
    std::copy(work.begin(), work.begin() + len, ranks.begin());
    self_i = std::distance(work.begin(), std::find(work.begin(), work.begin() + len, my_rank));
  }
}

void nbd::slices_level(int64_t slices[], int64_t lbegin, int64_t lend, int64_t dim) {
  std::fill(slices, slices + dim, 1);
  int64_t sdim = lbegin % dim;
  for (int64_t i = lbegin; i < lend; i++) {
    slices[sdim] <<= 1;
    sdim = (sdim == dim - 1) ? 0 : sdim + 1;
  }
}

void nbd::Global_Partition(GlobalDomain& goDomain, int64_t Nbodies, int64_t Ncrit, int64_t dim, double min, double max) {
  goDomain.DIM = dim;
  goDomain.NBODY = Nbodies;

  goDomain.LEVELS = (int64_t)std::floor(std::log2(Nbodies / Ncrit));
  goDomain.Xmin.resize(dim, min);
  goDomain.Xmax.resize(dim, max);
}


void nbd::Local_Partition(LocalDomain& loDomain, const GlobalDomain& goDomain, int64_t rank, int64_t size, int64_t theta) {
  loDomain.RANK = rank;
  loDomain.MY_LEVEL = (int64_t)std::floor(std::log2(size));
  loDomain.DIM = goDomain.DIM;
  loDomain.THETA = theta;

  loDomain.LOCAL_LEVELS = goDomain.LEVELS - loDomain.MY_LEVEL;
  int64_t boxes_local = (int64_t)1 << loDomain.LOCAL_LEVELS;
  size = (int64_t)1 << loDomain.MY_LEVEL;
}


void nbd::Bucket_sort(double* bodies, int64_t* lens, int64_t* offsets, int64_t nbodies, const GlobalDomain& goDomain, const LocalDomain& loDomain) {
  int64_t dim = goDomain.DIM;
  std::vector<int64_t> slices(dim);
  slices_level(slices.data(), 0, goDomain.LEVELS, dim);
  
  std::vector<double> box_dim(dim);
  for (int64_t d = 0; d < dim; d++)
    box_dim[d] = (goDomain.Xmax[d] - goDomain.Xmin[d]) / slices[d];

  int64_t nboxes = (int64_t)1 << loDomain.LOCAL_LEVELS;
  std::fill(lens, lens + nboxes, 0);

  std::vector<double*> bodies_p(nbodies);
  std::vector<int64_t> bodies_i(nbodies);
  std::vector<int64_t> Xi(dim);
  int64_t lbegin = loDomain.RANK * nboxes;

  for (int64_t i = 0; i < nbodies; i++) {
    double* p = &bodies[i * dim];
    int64_t ind;
    for (int64_t d = 0; d < dim; d++)
      Xi[d] = (int64_t)std::floor((p[d] - goDomain.Xmin[d]) / box_dim[d]);
    Z_index_i(Xi.data(), dim, ind);
    ind = ind - lbegin;
    bodies_p[i] = p;
    bodies_i[i] = ind;
    lens[ind]++;
  }

  int64_t old_offset = offsets[0];
  offsets[0] = 0;
  for(int64_t i = 1; i < nboxes; i++)
    offsets[i] = offsets[i - 1] + lens[i - 1];
  std::vector<double> bodies_cpy(nbodies * dim);

  for (int64_t i = 0; i < nbodies; i++) {
    int64_t bi = bodies_i[i];
    const double* src = bodies_p[i];
    int64_t offset_bi = offsets[bi];
    double* tar = &bodies_cpy[offset_bi * dim];
    for (int64_t d = 0; d < dim; d++)
      tar[d] = src[d];
    offsets[bi] = offset_bi + 1;
  }
  
  std::copy(&bodies_cpy[0], &bodies_cpy[nbodies * dim], bodies);
  offsets[0] = old_offset;
  for(int64_t i = 1; i < nboxes; i++)
    offsets[i] = offsets[i - 1] + lens[i - 1];
}


void nbd::N_bodies_box(int64_t Nbodies, int64_t i, int64_t box_lvl, int64_t& bodies_box) {
  if (box_lvl <= 0)
    bodies_box = Nbodies;
  else {
    int64_t bodies_parent;
    N_bodies_box(Nbodies, i >> 1, box_lvl - 1, bodies_parent);
    int64_t a = (i & 1) & (bodies_parent & 1);
    bodies_box = bodies_parent / 2 + a;
  }
}

void nbd::Local_bounds(double* Xmin, double* Xmax, const GlobalDomain& goDomain, const LocalDomain& loDomain) {
  std::vector<int64_t> Xi(goDomain.DIM);
  std::vector<int64_t> Slice(goDomain.DIM);
  Z_index(loDomain.RANK << loDomain.LOCAL_LEVELS, goDomain.DIM, Xi.data());
  slices_level(Slice.data(), loDomain.LOCAL_LEVELS, goDomain.LEVELS, goDomain.DIM);

  int64_t d = 0;
  for (int64_t i = 0; i < loDomain.LOCAL_LEVELS; i++) {
    Xi[d] >>= 1;
    d = (d == goDomain.DIM - 1) ? 0 : d + 1;
  }

  for (d = 0; d < goDomain.DIM; d++) {
    double bo_len = (goDomain.Xmax[d] - goDomain.Xmin[d]) / Slice[d];
    Xmin[d] = goDomain.Xmin[d] + Xi[d] * bo_len;
    Xmax[d] = Xmin[d] + bo_len;
  }
}

void nbd::Alloc_bodies(LocalBodies& bodies, const GlobalDomain& goDomain, const LocalDomain& loDomain) {
  bodies.BOXES = (int64_t)1 << loDomain.LOCAL_LEVELS;
  bodies.DIM = loDomain.DIM;
  Z_neighbors_level(bodies.RANKS, bodies.SELF_I, loDomain.MY_LEVEL, loDomain);

  int64_t nodes = bodies.RANKS.size();
  bodies.NBODIES.resize(nodes);
  bodies.LENS.resize(nodes * bodies.BOXES);
  bodies.OFFSETS.resize(nodes * bodies.BOXES);

  int64_t tot_local = 0;
  for (int64_t i = 0; i < nodes; i++) {
    N_bodies_box(goDomain.NBODY, bodies.RANKS[i], loDomain.MY_LEVEL, bodies.NBODIES[i]);
    bodies.OFFSETS[i * bodies.BOXES] = tot_local;
    tot_local = tot_local + bodies.NBODIES[i];
  }
  
  bodies.BODIES.resize(tot_local * bodies.DIM);
}


void nbd::Random_bodies(LocalBodies& bodies, const GlobalDomain& goDomain, const LocalDomain& loDomain, unsigned int seed) {
  if (seed)
    std::srand(seed);

  std::vector<double> Xmin_box(goDomain.DIM);
  std::vector<double> Xmax_box(goDomain.DIM);
  Local_bounds(Xmin_box.data(), Xmax_box.data(), goDomain, loDomain);
  Alloc_bodies(bodies, goDomain, loDomain);

  int64_t ind = bodies.SELF_I;
  double* bodies_begin = &bodies.BODIES[bodies.OFFSETS[ind]];
  int64_t d = 0;
  for (int64_t i = 0; i < bodies.NBODIES[ind] * goDomain.DIM; i++) {
    double min = Xmin_box[d];
    double max = Xmax_box[d];
    double r = min + (max - min) * ((double)std::rand() / RAND_MAX);
    bodies_begin[i] = r;
    d = (d == goDomain.DIM - 1) ? 0 : d + 1;
  }

  Bucket_sort(bodies_begin, &bodies.LENS[ind], &bodies.OFFSETS[ind], bodies.NBODIES[ind], goDomain, loDomain);
}


void nbd::Interactions(CSC& rels, int64_t y, int64_t xbegin, int64_t xend, int64_t dim, int64_t theta) {
  rels.M = y;
  rels.N = xend - xbegin;
  rels.CSC_COLS.resize(rels.N + 1);
  rels.CSC_ROWS.clear();
  rels.CSC_COLS[0] = 0;

  std::vector<int64_t> work(y);
  for (int64_t j = 0; j < rels.N; j++) {
    int64_t len = Z_neighbors(work.data(), j + xbegin, dim, y, theta);
    rels.CSC_ROWS.resize(rels.CSC_ROWS.size() + len);
    std::copy(work.begin(), work.begin() + len, rels.CSC_ROWS.begin() + rels.CSC_COLS[j]);
    rels.CSC_COLS[j + 1] = rels.CSC_ROWS.size();
  }
  rels.NNZ = rels.CSC_COLS[rels.N];
}


void nbd::Local_Interactions(CSC& rels, int64_t level, const LocalDomain& loDomain) {
  int64_t dim = loDomain.DIM;
  int64_t y = (int64_t)1 << level;
  int64_t xbegin, xend;
  if (level > loDomain.MY_LEVEL) {
    int64_t local_levels = level - loDomain.MY_LEVEL;
    xbegin = loDomain.RANK << local_levels;
    xend = (loDomain.RANK + 1) << local_levels;
  }
  else {
    int64_t merged_levels = loDomain.MY_LEVEL - level;
    xbegin = loDomain.RANK >> merged_levels;
    xend = (loDomain.RANK + 1) >> merged_levels;
  }
  Interactions(rels, y, xbegin, xend, dim, loDomain.THETA);
}


void nbd::BlockCSC(Matrices& A, CSC& rels, EvalFunc ef, const LocalDomain& loDomain, const LocalBodies& bodies) {
  int64_t dim = bodies.DIM;
  int64_t nboxes = bodies.BOXES;
  Local_Interactions(rels, loDomain.MY_LEVEL + loDomain.LOCAL_LEVELS, loDomain);
  A.resize(rels.NNZ);

  for (int64_t j = 0; j < rels.N; j++) {
    int64_t box_j = bodies.SELF_I * nboxes + j;
    int64_t nbodies_j = bodies.LENS[box_j];
    int64_t offset_j = bodies.OFFSETS[box_j] * dim;

    for (int64_t ij = rels.CSC_COLS[j]; ij < rels.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels.CSC_ROWS[ij];
      int64_t box_rank = i / nboxes;
      int64_t i_rank = i - box_rank * nboxes;

      int64_t box_ind = std::distance(bodies.RANKS.begin(), std::find(bodies.RANKS.begin(), bodies.RANKS.end(), box_rank));
      int64_t box_i = box_ind * nboxes + i_rank;
      int64_t nbodies_i = bodies.LENS[box_i];
      int64_t offset_i = bodies.OFFSETS[box_i] * dim;

      Matrix& A_ij = A[ij];
      cMatrix(A_ij, nbodies_i, nbodies_j);
      matrix_kernel(ef, nbodies_i, nbodies_j, &bodies.BODIES[offset_i], &bodies.BODIES[offset_j], dim, A_ij.A.data(), A_ij.M);
    }
  }
}


void nbd::printLocal(const LocalDomain& loDomain, const LocalBodies& loBodies) {
  printf("-- Process: %ld out of %ld --\n", loDomain.RANK, (int64_t)1 << loDomain.MY_LEVEL);
  printf(" Level: %ld\n", loDomain.MY_LEVEL);
  printf(" Local Tree Levels: %ld\n", loDomain.LOCAL_LEVELS);
  printf(" Boxes: %ld\n", loBodies.BOXES);

  printf(" Ranks: ");
  for (int64_t i = 0; i < loBodies.RANKS.size(); i++)
    printf("%ld::%ld, ", loBodies.RANKS[i], loBodies.NBODIES[i]);
  printf("\n");
}
