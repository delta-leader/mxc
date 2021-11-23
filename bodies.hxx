
#pragma once

#include "kernel.hxx"
#include "linalg.hxx"
#include "domain.hxx"

namespace nbd {

  struct LocalBodies {
    int64_t DIM;
    std::vector<int64_t> NBODIES;

    std::vector<double> BODIES;
    std::vector<int64_t> LENS;
    std::vector<int64_t> OFFSETS;
  };

  void Bucket_sort(double* bodies, int64_t* lens, int64_t* offsets, int64_t nbodies, int64_t nboxes, const GlobalDomain& goDomain);

  void N_bodies_box(int64_t Nbodies, int64_t i, int64_t box_lvl, int64_t& bodies_box);

  void Alloc_bodies(LocalBodies& bodies, const GlobalDomain& goDomain, const GlobalIndex& gi);

  void localBodiesDim(int64_t* dims, const GlobalIndex& gi, const LocalBodies& bodies);

  void Random_bodies(LocalBodies& bodies, const GlobalDomain& goDomain, const GlobalIndex& gi, unsigned int seed);

  void BlockCSC(Matrices& A, EvalFunc ef, const GlobalIndex& gi, const LocalBodies& bodies);

  Vector* randomVectors(Vectors& B, const GlobalIndex& gi, const LocalBodies& bodies, double min, double max, unsigned int seed);

  void blockAxEb(Vectors& B, EvalFunc ef, const Vectors& X, const GlobalIndex& gi, const LocalBodies& bodies);

  void checkBodies(int64_t my_rank, const GlobalDomain& goDomain, const GlobalIndex& gi, const LocalBodies& bodies);

};

