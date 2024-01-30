
#include <geometry.hpp>
#include <kernel.hpp>
#include <build_tree.hpp>
#include <basis.hpp>
#include <comm.hpp>
#include <solver.hpp>

#include <random>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int64_t Nbody = argc > 1 ? atol(argv[1]) : 8192;
  double theta = argc > 2 ? atof(argv[2]) : 1e0;
  int64_t leaf_size = argc > 3 ? atol(argv[3]) : 256;
  double epi = argc > 4 ? atof(argv[4]) : 1e-10;
  const char* fname = argc > 5 ? argv[5] : nullptr;

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  int64_t levels = (int64_t)log2((double)Nbody / leaf_size);
  int64_t Nleaf = (int64_t)1 << levels;
  int64_t ncells = Nleaf + Nleaf - 1;
  int64_t nrhs = 2;
  MPI_Comm world;
  MPI_Comm_dup(MPI_COMM_WORLD, &world);

  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(world, &mpi_rank);
  MPI_Comm_size(world, &mpi_size);
  
  //Laplace3D eval(1);
  //Yukawa3D eval(1, 1.);
  //Gaussian eval(8);
  Helmholtz3D eval(1.e-2, 1.);
  
  std::vector<double> body(Nbody * 3);
  std::vector<std::complex<double>> Xbody(Nbody * nrhs);
  std::vector<Cell> cell(ncells);

  std::vector<CellComm> cell_comm(levels + 1);
  std::vector<MatVecBasis> basis(levels + 1);

  if (fname == nullptr) {
    mesh_unit_sphere(&body[0], Nbody, std::pow(Nbody, 1./2.));
    //mesh_unit_cube(&body[0], Nbody);
    //uniform_unit_cube(&body[0], Nbody, 3);
    buildTree(&cell[0], &body[0], Nbody, levels);
  }
  else {
    std::vector<int64_t> buckets(Nleaf);
    read_sorted_bodies(&Nbody, Nleaf, &body[0], &buckets[0], fname);
    //buildTreeBuckets(cell, body, buckets, levels);
    buildTree(&cell[0], &body[0], Nbody, levels);
  }

  std::mt19937 gen(999);
  std::uniform_real_distribution<> dis(0., 1.);
  for (int64_t n = 0; n < (int64_t)Xbody.size(); ++n)
    Xbody[n] = std::complex<double>(dis(gen), 0.);

  /*cell.erase(cell.begin() + 1, cell.begin() + Nleaf - 1);
  cell[0].Child[0] = 1; cell[0].Child[1] = Nleaf + 1;
  ncells = Nleaf + 1;
  levels = 1;*/

  CSR cellNear('N', ncells, &cell[0], theta);
  CSR cellFar('F', ncells, &cell[0], theta);
  CSR cellFill(cellNear, cellNear);

  std::pair<double, double> timer(0, 0);
  std::vector<MPI_Comm> mpi_comms;
  std::vector<std::pair<int64_t, int64_t>> mapping = getProcessMapping(mpi_size, &cell[0], ncells);
  std::vector<int64_t> levelOffsets = getLevelOffsets(&cell[0], ncells);
  
  for (int64_t i = 0; i <= levels; i++) {
    int64_t ibegin = levelOffsets[i];
    int64_t iend = levelOffsets[i + 1];
    getLocalRange(ibegin, iend, mpi_rank, mapping);
    
    int64_t child = cell[ibegin].Child[0];
    int64_t cend = cell[ibegin].Child[1];
    cell_comm[i] = CellComm(ibegin, iend, child, cend, mapping, cellNear, cellFar, mpi_comms, world);
    cell_comm[i].timer = &timer;
  }

  int64_t llen = cell_comm[levels].lenLocal();
  int64_t gbegin = cell_comm[levels].oGlobal();

  MPI_Barrier(MPI_COMM_WORLD);
  double construct_time = MPI_Wtime(), construct_comm_time;
  buildBasis(eval, epi, &basis[0], &cell[0], cellNear, levels, &cell_comm[0], &body[0], Nbody);

  MPI_Barrier(MPI_COMM_WORLD);
  construct_time = MPI_Wtime() - construct_time;
  construct_comm_time = timer.first;
  timer.first = 0;

  int64_t body_local[2] = { cell[gbegin].Body[0], cell[gbegin + llen - 1].Body[1] };
  int64_t lenX = body_local[1] - body_local[0];
  std::vector<std::complex<double>> X1(lenX * nrhs, std::complex<double>(0., 0.));
  std::vector<std::complex<double>> X2(lenX * nrhs, std::complex<double>(0., 0.));

  MatVec mv(eval, &basis[0], &body[0], &cell[0], cellNear, cellFar, &cell_comm[0], levels);
  for (int64_t i = 0; i < nrhs; i++)
    std::copy(&Xbody[i * Nbody] + body_local[0], &Xbody[i * Nbody] + body_local[1], &X1[i * lenX]);

  MPI_Barrier(MPI_COMM_WORLD);
  double matvec_time = MPI_Wtime(), matvec_comm_time;
  mv(nrhs, &X1[0], lenX);

  MPI_Barrier(MPI_COMM_WORLD);
  matvec_time = MPI_Wtime() - matvec_time;
  matvec_comm_time = timer.first;
  timer.first = 0;

  double cerr = 0.;
  mat_vec_reference(eval, lenX, Nbody, nrhs, &X2[0], lenX, &Xbody[0], Nbody, &body[body_local[0] * 3], &body[0]);

  solveRelErr(&cerr, &X1[0], &X2[0], lenX * nrhs);

  std::cout << cerr << std::endl;
  std::cout << construct_time << ", " << construct_comm_time << std::endl;
  std::cout << matvec_time << ", " << matvec_comm_time << std::endl;

  //Solver solver(basis[levels].Dims.data(), cellNear, cell_comm[levels]);

  for (MPI_Comm& c : mpi_comms)
    MPI_Comm_free(&c);
  MPI_Comm_free(&world);
  MPI_Finalize();
  return 0;
}
