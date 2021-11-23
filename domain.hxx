
#pragma once

#include <vector>
#include <cstdint>

namespace nbd {

  struct GlobalDomain {
    int64_t NBODY;
    int64_t LEVELS;
    int64_t DIM;
    int64_t MY_RANK;
    int64_t MY_LEVEL;
    std::vector<double> Xmin;
    std::vector<double> Xmax;
  };

  struct CSC {
    int64_t M;
    int64_t N;
    int64_t NNZ;
    std::vector<int64_t> CSC_COLS;
    std::vector<int64_t> CSC_ROWS;
  };

  struct GlobalIndex {
    int64_t BOXES;
    int64_t SELF_I;
    int64_t GBEGIN;
    std::vector<int64_t> NGB_RNKS;
    std::vector<int64_t> COMM_RNKS;
    CSC RELS;
  };

  typedef std::vector<GlobalIndex> LocalDomain;

  void Z_index(int64_t i, int64_t dim, int64_t X[]);

  void Z_index_i(const int64_t X[], int64_t dim, int64_t& i);

  int64_t Z_neighbors(int64_t N[], int64_t i, int64_t dim, int64_t max_i, int64_t theta);

  void slices_level(int64_t slices[], int64_t lbegin, int64_t lend, int64_t dim);

  void Global_Partition(GlobalDomain& goDomain, int64_t rank, int64_t size, int64_t Nbodies, int64_t Ncrit, int64_t dim, double min, double max);

  void Interactions(CSC& rels, int64_t y, int64_t xbegin, int64_t xend, int64_t dim, int64_t theta);

  GlobalIndex* Local_Partition(LocalDomain& loDomain, const GlobalDomain& goDomain, int64_t theta);

  void Local_bounds(double* Xmin, double* Xmax, const GlobalDomain& goDomain);

  void lookupIJ(int64_t& ij, const CSC& rels, int64_t i, int64_t j);

  void Lookup_GlobalI(int64_t& ilocal, const GlobalIndex& gi, int64_t iglobal);

  void printGlobalI(const GlobalIndex& gi);

};