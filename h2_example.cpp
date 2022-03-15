
#include "build_tree.hxx"
#include "kernel.hxx"
#include "h2mv.hxx"
#include "basis.hxx"
#include "dist.hxx"

#include <cstdio>
#include <cstdlib>
#include <random>

int main(int argc, char* argv[]) {

  using namespace nbd;

  int64_t dim = 2;
  int64_t m = 4000;
  int64_t leaf = 200;
  int64_t rank = 200;
  int64_t theta = 1;

  std::vector<double> my_min(dim + 1, 0.);
  std::vector<double> my_max(dim + 1, 1.);
  auto fun = dim == 2 ? l2d() : l3d();

  Bodies body(m);
  randomBodies(body, m, &my_min[0], &my_max[0], dim, 1234);
  Cells cell;
  int64_t levels = buildTree(cell, body, leaf, &my_min[0], &my_max[0], dim);

  int64_t mpi_rank = 0;
  int64_t mpi_size = 1;
  initComm(&argc, &argv, &mpi_rank, &mpi_size);

  traverse(cell, levels, dim, theta, mpi_rank, mpi_size);
  evaluateBasis(fun, cell, &cell[0], body, 2000, rank, dim);

  const Cell* local = &cell[0];
  Basis basis;
  allocBasis(basis, levels);
  for (int64_t i = 0; i <= levels; i++) {
    local = findLocalAtLevel(local, i, mpi_rank, mpi_size);
    fillDimsFromCell(basis[i], local, i);
  }

  std::vector<CSC> cscs(levels + 1);
  relationsNear(&cscs[0], cell, mpi_rank, mpi_size);
  std::vector<Matrices> d(levels + 1);
  evaluateNear(&d[0], fun, cell, dim, &cscs[0], levels);

  std::vector<MatVec> vx(levels + 1);
  allocMatVec(&vx[0], &basis[0], levels);

  Vectors X;
  loadX(X, local, levels);

  h2MatVecAll(&vx[0], fun, &cell[0], &basis[0], dim, X, levels, mpi_rank, mpi_size);

  /*zeroVector(b_ref);
  P2P(fun, &cell[0], &cell[0], dim, x, b_ref);

  printf("H2-vec vs direct m-vec err %e\n", rel2err(&b.X[0], &b_ref.X[0], m, 1, m, m));*/

  closeComm();
  return 0;
}
