
#include <geometry.hpp>
#include <kernel.hpp>
#include <nbd.hpp>
#include <profile.hpp>

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int64_t Nbody = argc > 1 ? atol(argv[1]) : 8192;
  double theta = argc > 2 ? atof(argv[2]) : 1e0;
  int64_t leaf_size = argc > 3 ? atol(argv[3]) : 256;
  double epi = argc > 4 ? atof(argv[4]) : 1e-10;
  int64_t rank_max = argc > 5 ? atol(argv[5]) : 100;
  int64_t sp_pts = argc > 6 ? atol(argv[6]) : 2000;
  const char* fname = argc > 7 ? argv[7] : NULL;

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  int64_t levels = (int64_t)log2((double)Nbody / leaf_size);
  int64_t Nleaf = (int64_t)1 << levels;
  int64_t ncells = Nleaf + Nleaf - 1;
  
  Laplace3D eval(1.e-6);
  //Yukawa3D eval(1.e-6, 1.);
  //Gaussian eval(0.2);
  
  std::vector<double> body(Nbody * 3);
  std::vector<double> Xbody(Nbody);
  struct Cell* cell = (struct Cell*)calloc(ncells, sizeof(struct Cell));
  CSR cellNear, cellFar;
  std::vector<CSR> rels_far(levels + 1), rels_near(levels + 1);

  struct CellComm* cell_comm = (struct CellComm*)calloc(levels + 1, sizeof(struct CellComm));
  struct Base* basis = (struct Base*)calloc(levels + 1, sizeof(struct Base));
  struct Node* nodes = (struct Node*)malloc(sizeof(struct Node) * (levels + 1));

  if (fname == NULL) {
    mesh_unit_sphere(&body[0], Nbody);
    //mesh_unit_cube(&body[0], Nbody);
    //uniform_unit_cube(&body[0], Nbody, 1);
    double c[3] = { 0, 0, 0 };
    double r[3] = { 1, 1, 1 };
    magnify_reloc(&body[0], Nbody, c, c, r, sqrt(Nbody));
    buildTree(&ncells, cell, &body[0], Nbody, levels);
  }
  else {
    int64_t* buckets = (int64_t*)malloc(sizeof(int64_t) * Nleaf);
    read_sorted_bodies(&Nbody, Nleaf, &body[0], buckets, fname);
    //buildTreeBuckets(cell, body, buckets, levels);
    buildTree(&ncells, cell, &body[0], Nbody, levels);
    free(buckets);
  }
  body_neutral_charge(&Xbody[0], Nbody, 1., 999);

  traverse('N', &cellNear, ncells, cell, theta);
  traverse('F', &cellFar, ncells, cell, theta);

  struct CommTimer timer;
  buildComm(cell_comm, ncells, cell, &cellFar, &cellNear, levels);
  for (int64_t i = 0; i <= levels; i++) {
    cell_comm[i].timer = &timer;
  }
  relations(&rels_near[0], &cellNear, levels, cell_comm);
  relations(&rels_far[0], &cellFar, levels, cell_comm);

  int64_t lbegin = 0, llen = 0;
  content_length(&llen, NULL, &lbegin, &cell_comm[levels]);
  int64_t gbegin = cell_comm[levels].iGlobal(lbegin);

  MPI_Barrier(MPI_COMM_WORLD);
  double construct_time = MPI_Wtime(), construct_comm_time;
  buildBasis(eval, basis, cell, &cellNear, levels, cell_comm, &body[0], Nbody, epi, rank_max, sp_pts, 4);

  MPI_Barrier(MPI_COMM_WORLD);
  construct_time = MPI_Wtime() - construct_time;
  construct_comm_time = timer.get_comm_timing();

  allocNodes(nodes, basis, &rels_near[0], &rels_far[0], cell_comm, levels);

  evalD(eval, nodes[levels].A, &cellNear, cell, &body[0], &cell_comm[levels]);
  for (int64_t i = 0; i <= levels; i++)
    evalS(eval, nodes[i].S, &basis[i], &rels_far[i], &cell_comm[i]);

  int64_t lenX = rels_near[levels].N * basis[levels].dimN;
  std::vector<double> X1(lenX, 0);
  std::vector<double> X2(lenX, 0);

  loadX(&X1[0], basis[levels].dimN, &Xbody[0], 0, llen, &cell[gbegin]);
  double matvec_time = MPI_Wtime(), matvec_comm_time;
  matVecA(nodes, basis, &rels_near[0], &X1[0], cell_comm, levels);

  matvec_time = MPI_Wtime() - matvec_time;
  matvec_comm_time = timer.get_comm_timing();

  double cerr = 0.;
  int64_t body_local[2] = { cell[gbegin].Body[0], cell[gbegin + llen - 1].Body[1] };
  std::vector<double> X3(lenX, 0);
  mat_vec_reference(eval, body_local[0], body_local[1], &X3[0], Nbody, &body[0], &Xbody[0]);
  loadX(&X2[0], basis[levels].dimN, &X3[0], body_local[0], llen, &cell[gbegin]);

  solveRelErr(&cerr, &X1[0], &X2[0], lenX);

  std::cout << cerr << std::endl;
  
  for (int64_t i = 0; i <= levels; i++) {
    basis_free(&basis[i]);
    node_free(&nodes[i]);
  }
  cellComm_free(cell_comm, levels);
  
  free(cell);
  free(cell_comm);
  free(basis);
  free(nodes);

  MPI_Finalize();
  return 0;
}
