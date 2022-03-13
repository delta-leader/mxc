
#pragma once
#include "kernel.hxx"

namespace nbd {

  struct Body {
    double X[3];
    double B;
  };

  typedef std::vector<Body> Bodies;

  struct Cell {
    int64_t NCHILD;
    int64_t NBODY;
    Cell* CHILD;
    Body* BODY;
    int64_t ZID;
    int64_t ZX[3];
    int64_t LEVEL;

    std::vector<Cell*> listFar;
    std::vector<Cell*> listNear;
    std::vector<int64_t> Multipole;
    int64_t MPOS;
  };

  typedef std::vector<Cell> Cells;

  struct CSC {
    int64_t M;
    int64_t N;
    int64_t NNZ;
    int64_t CBGN;
    std::vector<int64_t> CSC_COLS;
    std::vector<int64_t> CSC_ROWS;
  };

  void randomBodies(Bodies& bodies, int64_t nbody, const double dmin[], const double dmax[], int64_t dim, int seed);

  int64_t getIndex(const int64_t iX[], int64_t dim);

  void getIX(int64_t iX[], int64_t index, int64_t dim);

  void bucketSort(Bodies& bodies, int64_t buckets[], int64_t slices[], const double dmin[], const double dmax[], int64_t dim);

  void buildTree(Cells& cells, Bodies& bodies, int64_t ncrit, const double dmin[], const double dmax[], int64_t dim);

  void getList(Cell* Ci, Cell* Cj, int64_t dim, int64_t theta);

  void findCellsAtLevel(int64_t off[], int64_t* len, const Cell* cell, const Cell* root, int64_t level);

  void remoteBodies(Bodies& remote, int64_t size, const Cell& cell, const Bodies& bodies, int64_t dim);

  void evaluateBasis(Matrices& basis, EvalFunc ef, Cells& cells, Cell* c, const Bodies& bodies, int64_t sp_pts, int64_t rank, int64_t dim);

  void traverse(EvalFunc ef, Cells& cells, int64_t dim, Matrices& base, const Bodies& bodies, int64_t sp_pts, int64_t theta, int64_t rank);

  void invAllBase(const Matrices& base, Matrices& binv);

  void relationsNear(CSC rels[], const Cells& cells, int64_t mpi_rank, int64_t mpi_size);

  void evaluateLeafNear(Matrices& d, EvalFunc ef, const Cell* cell, int64_t dim, const CSC& csc);

  void lookupIJ(int64_t& ij, const CSC& rels, int64_t i, int64_t j);

  void evaluateNear(Matrices d[], EvalFunc ef, const Cells& cells, int64_t dim, const Matrices& uinv, const CSC rels[], int64_t levels);

}

