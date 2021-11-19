
#include "bodies.hxx"

#include <random>
#include <algorithm>
#include <cstdio>

using namespace nbd;


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


void nbd::Alloc_bodies(LocalBodies& bodies, const GlobalDomain& goDomain, const LocalDomain& loDomain) {
  const GlobalIndex& gi_leaf = loDomain.MY_IDS.back();
  int64_t nboxes = gi_leaf.BOXES;
  bodies.DIM = loDomain.DIM;

  int64_t nodes = gi_leaf.NGB_RNKS.size();
  bodies.NBODIES.resize(nodes);
  bodies.LENS.resize(nodes * nboxes);
  bodies.OFFSETS.resize(nodes * nboxes);

  int64_t tot_local = 0;
  for (int64_t i = 0; i < nodes; i++) {
    N_bodies_box(goDomain.NBODY, gi_leaf.NGB_RNKS[i], loDomain.MY_LEVEL, bodies.NBODIES[i]);
    bodies.OFFSETS[i * nboxes] = tot_local;
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

  int64_t ind = loDomain.MY_IDS.back().SELF_I;
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



void nbd::BlockCSC(Matrices& A, EvalFunc ef, const LocalDomain& loDomain, const LocalBodies& bodies) {
  const GlobalIndex& gi_leaf = loDomain.MY_IDS.back();
  const CSC& rels = gi_leaf.RELS;
  int64_t dim = bodies.DIM;
  int64_t nboxes = gi_leaf.BOXES;
  A.resize(rels.NNZ);

  for (int64_t j = 0; j < rels.N; j++) {
    int64_t box_j = gi_leaf.SELF_I * nboxes + j;
    int64_t nbodies_j = bodies.LENS[box_j];
    int64_t offset_j = bodies.OFFSETS[box_j] * dim;

    for (int64_t ij = rels.CSC_COLS[j]; ij < rels.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels.CSC_ROWS[ij];
      int64_t box_i;
      Lookup_GlobalI(box_i, gi_leaf, i);
      int64_t nbodies_i = bodies.LENS[box_i];
      int64_t offset_i = bodies.OFFSETS[box_i] * dim;

      Matrix& A_ij = A[ij];
      cMatrix(A_ij, nbodies_i, nbodies_j);
      matrix_kernel(ef, nbodies_i, nbodies_j, &bodies.BODIES[offset_i], &bodies.BODIES[offset_j], dim, A_ij.A.data(), A_ij.M);
    }
  }
}


