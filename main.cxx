

#include "solver.hxx"
#include "dist.hxx"
#include "h2mv.hxx"
#include "minblas.h"

#include <random>
#include <cstdio>
#include <cmath>

using namespace nbd;

int main(int argc, char* argv[]) {

  int64_t mpi_rank = 0;
  int64_t mpi_size = 1;
  initComm(&argc, &argv, &mpi_rank, &mpi_size);

  int64_t Nbody = 40000;
  int64_t Ncrit = 100;
  int64_t theta = 1;
  int64_t dim = 2;
  EvalFunc ef = dim == 2 ? l2d() : l3d();

  std::srand(100);
  std::vector<double> R(1 << 16);
  for (int64_t i = 0; i < R.size(); i++)
    R[i] = -1. + 2. * ((double)std::rand() / RAND_MAX);

  double my_min[]{ 0., 0., -1.e2 };
  double my_max[]{ 1., 1., 1.e2 };

  Bodies body(Nbody);
  randomBodies(body, Nbody, my_min, my_max, dim, 1234);
  Cells cell;
  int64_t levels = buildTree(cell, body, Ncrit, my_min, my_max, dim);

  std::vector<Cell*> locals(levels + 1);
  traverse(cell, &locals[0], levels, dim, theta, mpi_rank, mpi_size);

  std::vector<CSC> rels(levels + 1);
  relationsNear(&rels[0], cell, mpi_rank, mpi_size);

  Nodes nodes;
  allocNodes(nodes, &rels[0], levels);
  Matrices& A = nodes.back().A;
  evaluateLeafNear(A, ef, &cell[0], dim, rels[levels]);

  Basis basis;
  allocBasis(basis, levels);
  fillDimsFromCell(basis.back(), &cell[0], levels);

  factorA(&nodes[0], &basis[0], &rels[0], levels, 1.e-6, R.data(), R.size());

  Vectors X;
  loadX(X, locals[levels], levels);

  RHSS rhs;
  allocRightHandSides(rhs, &basis[0], levels);
  closeQuarter(rhs[levels].X, X, ef, locals[levels], dim, levels);

  solveA(&rhs[0], &nodes[0], &basis[0], &rels[0], levels);

  double err;
  solveRelErr(&err, rhs[levels].X, X, levels);
  printf("%d ERR: %e\n", mpi_rank, err);

  int64_t* flops = getFLOPS();
  double gf = flops[0] * 1.e-9;
  printf("%d GFLOPS: %f\n", mpi_rank, gf);
  closeComm();
  return 0;
}
