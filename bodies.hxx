
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

  void Bucket_sort(double* bodies, int64_t* lens, int64_t* offsets, int64_t nbodies, const GlobalDomain& goDomain, const LocalDomain& loDomain);

  void N_bodies_box(int64_t Nbodies, int64_t i, int64_t box_lvl, int64_t& bodies_box);

  void Alloc_bodies(LocalBodies& bodies, const GlobalDomain& goDomain, const LocalDomain& loDomain);

  void Random_bodies(LocalBodies& bodies, const GlobalDomain& goDomain, const LocalDomain& loDomain, unsigned int seed);

  void BlockCSC(Matrices& A, EvalFunc ef, const LocalDomain& loDomain, const LocalBodies& bodies);

  void DistributeBodies(LocalBodies& bodies, const GlobalIndex& gi);

  void checkBodies(const GlobalDomain& goDomain, const LocalDomain& loDomain, const LocalBodies& bodies);

};

