
#pragma once

#include "kernel.hxx"
#include "linalg.hxx"

namespace nbd {

  struct GlobalDomain {
    int64_t NBODY;
    int64_t LEVELS;
    int64_t DIM;
    std::vector<double> Xmin;
    std::vector<double> Xmax;
  };

  struct LocalDomain {
    int64_t RANK;
    int64_t MY_LEVEL;
    int64_t LOCAL_LEVELS;
    int64_t DIM;
    int64_t THETA;
  };

  struct LocalBodies {
    int64_t BOXES;
    int64_t DIM;
    int64_t SELF_I;
    std::vector<int64_t> RANKS;
    std::vector<int64_t> NBODIES;

    std::vector<double> BODIES;
    std::vector<int64_t> LENS;
    std::vector<int64_t> OFFSETS;
  };

  void Z_index(int64_t i, int64_t dim, int64_t X[]);

  void Z_index_i(const int64_t X[], int64_t dim, int64_t& i);

  int64_t Z_neighbors(int64_t N[], int64_t i, int64_t dim, int64_t max_i, int64_t theta);

  void Z_neighbors_level(std::vector<int64_t>& ranks, int64_t& self_i, int64_t level, const LocalDomain& domain);

  void slices_level(int64_t slices[], int64_t lbegin, int64_t lend, int64_t dim);

  void Global_Partition(GlobalDomain& goDomain, int64_t Nbodies, int64_t Ncrit, int64_t dim, double min, double max);

  void Local_Partition(LocalDomain& loDomain, const GlobalDomain& goDomain, int64_t rank, int64_t size, int64_t theta);

  void Bucket_sort(double* bodies, int64_t* lens, int64_t* offsets, int64_t nbodies, const GlobalDomain& goDomain, const LocalDomain& loDomain);

  void N_bodies_box(int64_t Nbodies, int64_t i, int64_t box_lvl, int64_t& bodies_box);

  void Local_bounds(double* Xmin, double* Xmax, const GlobalDomain& goDomain, const LocalDomain& loDomain);

  void Alloc_bodies(LocalBodies& bodies, const GlobalDomain& goDomain, const LocalDomain& loDomain);

  void Random_bodies(LocalBodies& bodies, const GlobalDomain& goDomain, const LocalDomain& loDomain, unsigned int seed);

  void Interactions(CSC& rels, int64_t y, int64_t xbegin, int64_t xend, int64_t dim, int64_t theta);

  void Local_Interactions(CSC& rels, int64_t level, const LocalDomain& loDomain);

  void BlockCSC(Matrices& A, CSC& rels, EvalFunc ef, const LocalDomain& loDomain, const LocalBodies& bodies);

  void printLocal(const LocalDomain& loDomain, const LocalBodies& loBodies);

  void DistributeBodies(LocalBodies& bodies);


};

