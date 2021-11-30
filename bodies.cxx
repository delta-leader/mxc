
#include "bodies.hxx"
#include "dist.hxx"

#include <random>
#include <algorithm>

using namespace nbd;


void nbd::Bucket_sort(double* bodies, int64_t* lens, int64_t* offsets, int64_t nbodies, int64_t nboxes, const GlobalDomain& goDomain) {
  int64_t dim = goDomain.DIM;
  std::vector<int64_t> slices(dim);
  slices_level(slices.data(), 0, goDomain.LEVELS, dim);
  
  std::vector<double> box_dim(dim);
  for (int64_t d = 0; d < dim; d++)
    box_dim[d] = (goDomain.Xmax[d] - goDomain.Xmin[d]) / slices[d];

  std::fill(lens, lens + nboxes, 0);

  std::vector<int64_t> bodies_i(nbodies);
  std::vector<int64_t> Xi(dim);
  int64_t lbegin = goDomain.MY_RANK * nboxes;

  for (int64_t i = 0; i < nbodies; i++) {
    double* p = &bodies[i * dim];
    int64_t ind;
    for (int64_t d = 0; d < dim; d++)
      Xi[d] = (int64_t)std::floor((p[d] - goDomain.Xmin[d]) / box_dim[d]);
    Z_index_i(Xi.data(), dim, ind);
    ind = ind - lbegin;
    bodies_i[i] = ind;
    lens[ind] = lens[ind] + 1;
  }

  int64_t old_offset = offsets[0];
  offsets[0] = 0;
  for(int64_t i = 1; i < nboxes; i++)
    offsets[i] = offsets[i - 1] + lens[i - 1];
  std::vector<double> bodies_cpy(nbodies * dim);

  for (int64_t i = 0; i < nbodies; i++) {
    int64_t bi = bodies_i[i];
    const double* src = &bodies[i * dim];
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


void nbd::Alloc_bodies(LocalBodies& bodies, const GlobalDomain& goDomain, const GlobalIndex& gi) {
  int64_t nboxes = gi.BOXES;
  int64_t nodes = gi.NGB_RNKS.size();
  bodies.DIM = goDomain.DIM;
  bodies.NBODIES.resize(nodes);
  bodies.LENS.resize(nodes * nboxes);
  bodies.OFFSETS.resize(nodes * nboxes);

  int64_t tot_local = 0;
  for (int64_t i = 0; i < nodes; i++) {
    N_bodies_box(goDomain.NBODY, gi.NGB_RNKS[i], goDomain.MY_LEVEL, bodies.NBODIES[i]);
    bodies.OFFSETS[i * nboxes] = tot_local;
    tot_local = tot_local + bodies.NBODIES[i];
  }
  
  bodies.BODIES.resize(tot_local * bodies.DIM);
}


void nbd::Random_bodies(LocalBodies& bodies, const GlobalDomain& goDomain, const GlobalIndex& gi, unsigned int seed) {
  if (seed)
    std::srand(seed);

  std::vector<double> Xmin_box(goDomain.DIM);
  std::vector<double> Xmax_box(goDomain.DIM);
  Local_bounds(Xmin_box.data(), Xmax_box.data(), goDomain);
  Alloc_bodies(bodies, goDomain, gi);

  int64_t nboxes = gi.BOXES;
  int64_t ind = gi.SELF_I;
  int64_t offset = bodies.OFFSETS[ind * nboxes] * goDomain.DIM;
  int64_t nbody = bodies.NBODIES[ind];
  double* bodies_begin = &bodies.BODIES[offset];
  int64_t d = 0;
  for (int64_t i = 0; i < nbody * goDomain.DIM; i++) {
    double min = Xmin_box[d];
    double max = Xmax_box[d];
    double r = min + (max - min) * ((double)std::rand() / RAND_MAX);
    bodies_begin[i] = r;
    d = (d == goDomain.DIM - 1) ? 0 : d + 1;
  }

  int64_t* lens = &bodies.LENS[ind * nboxes];
  int64_t* offsets = &bodies.OFFSETS[ind * nboxes];
  Bucket_sort(bodies_begin, lens, offsets, nbody, nboxes, goDomain);
  DistributeBodies(bodies, gi);
}


void nbd::localBodiesDim(int64_t* dims, const GlobalIndex& gi, const LocalBodies& bodies) {
  int64_t lbegin = gi.SELF_I * gi.BOXES;
  int64_t lend = lbegin + gi.BOXES;
  std::copy(&bodies.LENS[lbegin], &bodies.LENS[lend], dims);
}


void nbd::BlockCSC(Matrices& A, EvalFunc ef, const GlobalIndex& gi, const LocalBodies& bodies) {
  const CSC& rels = gi.RELS;
  int64_t dim = bodies.DIM;
  int64_t lbegin = gi.SELF_I * gi.BOXES;
  A.resize(rels.NNZ);

  for (int64_t j = 0; j < rels.N; j++) {
    int64_t box_j = lbegin + j;
    int64_t nbodies_j = bodies.LENS[box_j];
    int64_t offset_j = bodies.OFFSETS[box_j] * dim;
    const double* bodies_j = &bodies.BODIES[offset_j];

    for (int64_t ij = rels.CSC_COLS[j]; ij < rels.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels.CSC_ROWS[ij];
      int64_t box_i;
      Lookup_GlobalI(box_i, gi, i);
      int64_t nbodies_i = bodies.LENS[box_i];
      int64_t offset_i = bodies.OFFSETS[box_i] * dim;
      const double* bodies_i = &bodies.BODIES[offset_i];

      Matrix& A_ij = A[ij];
      cMatrix(A_ij, nbodies_i, nbodies_j);
      matrix_kernel(ef, nbodies_i, nbodies_j, bodies_i, bodies_j, dim, A_ij.A.data(), A_ij.M);
    }
  }
}

Vector* nbd::randomVectors(Vectors& B, const GlobalIndex& gi, const LocalBodies& bodies, double min, double max, unsigned int seed) {
  if (seed)
    std::srand(seed);
  
  B.resize(bodies.LENS.size());
  for (int64_t i = 0; i < B.size(); i++)
    cVector(B[i], bodies.LENS[i]);
  
  std::vector<double> R(bodies.NBODIES[gi.SELF_I]);
  for (int64_t i = 0; i < R.size(); i++)
    R[i] = min + (max - min) * ((double)std::rand() / RAND_MAX);

  int64_t lbegin = gi.SELF_I * gi.BOXES;
  int64_t lend = lbegin + gi.BOXES;
  int64_t offset = 0;
  for (int64_t i = lbegin; i < lend; i++) {
    Vector& vi = B[i];
    vaxpby(vi, &R[offset], 1., 0.);
    offset = offset + bodies.LENS[i];
  }

  DistributeVectorsList(B, gi);
  return &B[lbegin];
}

void nbd::blockAxEb(Vector* B, EvalFunc ef, const Vectors& X, const GlobalIndex& gi, const LocalBodies& bodies) {
  int64_t lbegin = gi.SELF_I * gi.BOXES;
  int64_t dim = bodies.DIM;
  for (int64_t i = 0; i < gi.BOXES; i++) {
    cVector(B[i], bodies.LENS[i + lbegin]);
    zeroVector(B[i]);
  }
  
  const CSC& rels = gi.RELS;
  for (int64_t j = 0; j < rels.N; j++) {
    int64_t box_j = lbegin + j;
    int64_t nbodies_j = bodies.LENS[box_j];
    int64_t offset_j = bodies.OFFSETS[box_j] * dim;
    const double* bodies_j = &bodies.BODIES[offset_j];

    for (int64_t ij = rels.CSC_COLS[j]; ij < rels.CSC_COLS[j + 1]; ij++) {
      int64_t i = rels.CSC_ROWS[ij];
      int64_t box_i;
      Lookup_GlobalI(box_i, gi, i);
      int64_t nbodies_i = bodies.LENS[box_i];
      int64_t offset_i = bodies.OFFSETS[box_i] * dim;
      const double* bodies_i = &bodies.BODIES[offset_i];
      mvec_kernel(ef, nbodies_j, nbodies_i, bodies_j, bodies_i, dim, X[box_i].X.data(), B[j].X.data());
    }
  }
}
