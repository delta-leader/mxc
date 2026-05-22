
#include <solver.hpp>

#include <hidr.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>

// complex double
template class H2MatrixSolver<std::complex<double>>;
template void H2MatrixSolver<std::complex<double>>::solveGMRES<std::complex<float>>(double, H2MatrixSolver<std::complex<float>>&, std::complex<double>[], const std::complex<double>[], long long, long long);
template double solveRelErr(long long, const std::complex<double> X[], const std::complex<double> ref[], MPI_Comm);
// complex float
template class H2MatrixSolver<std::complex<float>>;
template double solveRelErr(long long, const std::complex<float> X[], const std::complex<float> ref[], MPI_Comm);

template <typename DT>
H2MatrixSolver<DT>::H2MatrixSolver() : levels(-1), A(), comm(), allocedComm(), local_bodies(0, 0) {
}
/*
H2MatrixSolver::H2MatrixSolver(const Accessor& eval_d, const MatrixAccessor& eval, double epi, long long rank, long long leveled_rank, const std::vector<Cell>& cells, double theta, const double bodies[], long long levels, MPI_Comm world) : 
  levels(levels), A(levels + 1), local_bodies(0, 0) {
  
  CSR Near('N', cells, cells, theta);
  CSR Far('F', cells, cells, theta);
  CSR HSS_Far('F', cells, cells, 0.);
  int mpi_size = 1;
  MPI_Comm_size(world, &mpi_size);

  std::vector<std::pair<long long, long long>> mapping(mpi_size, std::make_pair(0, 1));
  std::vector<std::pair<long long, long long>> tree(cells.size());
  std::transform(cells.begin(), cells.end(), tree.begin(), [](const Cell& c) { return std::make_pair(c.Child[0], c.Child[1]); });
  
  for (long long i = 0; i <= levels; i++)
    comm.emplace_back(&tree[0], &mapping[0], Near.RowIndex.data(), Near.ColIndex.data(), Far.RowIndex.data(), Far.ColIndex.data(), allocedComm, world);

  bool fix_rank = (epi == 0.);
  auto rank_func = [=](long long l) { return (levels - l) * leveled_rank + rank; };
  std::vector<Hmatrix> wsa(levels + 1);
  for (long long l = 1; l <= levels; l++)
    wsa[l].construct(epi, eval_d, rank_func(l), rank * 2, 2, comm[l].oGlobal(), comm[l].lenLocal(), cells.data(), fix_rank ? HSS_Far : Far, bodies, wsa[l - 1]);

  A[levels].construct(eval, fix_rank ? (double)rank_func(levels) : epi, cells.data(), Near, bodies, wsa[levels], comm[levels], A[levels], comm[levels]);
  for (long long l = levels - 1; l >= 0; l--)
    A[l].construct(eval, fix_rank ? (double)rank_func(l) : epi, cells.data(), Near, bodies, wsa[l], comm[l], A[l + 1], comm[l + 1]);

  long long llen = comm[levels].lenLocal();
  long long gbegin = comm[levels].oGlobal();
  local_bodies = std::make_pair(cells[gbegin].Body[0], cells[gbegin + llen - 1].Body[1]);
}*/

template <typename DT>
H2MatrixSolver<DT>::H2MatrixSolver(const MatrixGenerator& matgen, double epi, long long rank, long long leveled_rank, const std::vector<Cell>& cells, double theta, long long levels, double omega, bool io, MPI_Comm world) : 
  levels(levels), A(levels + 1), local_bodies(0, 0) {
  
  CSR Near('N', cells, cells, theta);
  CSR Far('F', cells, cells, theta);
  int mpi_size = 1;
  MPI_Comm_size(world, &mpi_size);
  std::vector<std::pair<long long, long long>> mapping(mpi_size, std::make_pair(0, 1));
  std::vector<std::pair<long long, long long>> tree(cells.size());
  std::transform(cells.begin(), cells.end(), tree.begin(), [](const Cell& c) { return std::make_pair(c.Child[0], c.Child[1]); });
  for (long long i = 0; i <= levels; i++) {
    comm.emplace_back(&tree[0], &mapping[0], Near.RowIndex.data(), Near.ColIndex.data(), Far.RowIndex.data(), Far.ColIndex.data(), allocedComm, world);
  }
  bool fix_rank = (epi == 0.);
  auto rank_func = [=](long long l) { return (levels - l) * leveled_rank + rank; };
  //std::vector<Hmatrix> wsa(levels + 1);
  //for (long long l = 1; l <= levels; l++)
  //  wsa[l].construct(epi, eval_d, rank_func(l), rank * 2, 2, comm[l].oGlobal(), comm[l].lenLocal(), cells.data(), fix_rank ? HSS_Far : Far, bodies, wsa[l - 1]);

  //std::vector<HiDR> hidr(levels + 1);
  //hidr[levels].initialize(0, comm[levels].oGlobal(), comm[levels].lenLocal(), cells.data());
  // I don't think I need to do anything for node 0
  //for (long long l = levels - 1; l > 0; l--) {
    //std::cout<<"Level "<<l<<std::endl;
    //hidr[l].bottom_up_sweep(0, comm[l].oGlobal(), comm[l].lenLocal(), cells.data(), hidr[l + 1]);
  //}
  //for (long long l = 1; l <= levels; l++) {
  //  std::cout<<"Level "<<l<<std::endl;
  //  hidr[l].top_down_sweep(0, cells.data(), Far, hidr[l - 1]);
  //}
  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // read and write from file
  std::string filename = "../tmp/";
  if (io) {
    MPI_Barrier(MPI_COMM_WORLD);
    double read_time = MPI_Wtime();
    if (!A[levels].read(levels, filename)) {
      //std::cout<<"construct"<<std::endl;
      A[levels].construct(matgen, fix_rank ? (double)rank_func(levels) : epi, cells.data(), Near, comm[levels], A[levels], comm[levels], omega);
      //std::cout<<"write"<<std::endl;
      A[levels].write(levels, filename);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    read_time = MPI_Wtime() - read_time;
    if (mpi_rank == 0) {
      std::cout<<"Read time "<<read_time<<std::endl;
    }
  } else {
    //std::cout<<"construct"<<std::endl;
    A[levels].construct(matgen, fix_rank ? (double)rank_func(levels) : epi, cells.data(), Near, comm[levels], A[levels], comm[levels], omega);
  }
  A[levels].lowest = true;
  /*double Qsize = A[levels].Q.size() * sizeof(std::complex<double>);
  double Rsize = A[levels].R.size() * sizeof(std::complex<double>);
  double Asize = A[levels].A.size() * sizeof(std::complex<double>);
  double Csize = A[levels].C.size() * sizeof(std::complex<double>);
  double Usize = A[levels].U.size() * sizeof(std::complex<double>);
  double Xsize = A[levels].X.size() * sizeof(std::complex<double>);
  double Ysize = A[levels].Y.size() * sizeof(std::complex<double>);
  double Zsize = A[levels].Z.size() * sizeof(std::complex<double>);
  double Wsize = A[levels].W.size() * sizeof(std::complex<double>);
  double total_size = Qsize + Rsize + Asize + Csize + Usize + Xsize + Ysize + Zsize + Wsize;
  for (int i = 0; i < mpi_size; i++) {
    if (mpi_rank == i) {
      std::cout<<"MPI rank "<<i<<std::endl;
      std::cout<<"  Total size on level "<<levels<<": "<<total_size<<" bytes"<<std::endl;
      std::cout<<"  Qsize: "<<Qsize<<std::endl;
      std::cout<<"  Rsize: "<<Rsize<<std::endl;
      std::cout<<"  Asize: "<<Asize<<std::endl;
      std::cout<<"  Csize: "<<Csize<<std::endl;
      std::cout<<"  Usize: "<<Usize<<std::endl;
      std::cout<<"  Xsize: "<<Xsize<<std::endl;
      std::cout<<"  Ysize: "<<Ysize<<std::endl;
      std::cout<<"  Zsize: "<<Zsize<<std::endl;
      std::cout<<"  Wsize: "<<Wsize<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  comm[levels].level_sum(&total_size, 1);
  if (mpi_rank == 0) {
    std::cout<<"Total size (all processes) on level "<<levels<<": "<<total_size<<" bytes"<<std::endl;
  }*/
  for (long long l = levels - 1; l >= 0; l--) {
    if (mpi_rank == 0) {
      std::cout<<"Level "<<l<<std::endl;
    }
    if (io) {
      MPI_Barrier(MPI_COMM_WORLD);
      double read_time = MPI_Wtime();
      if (!A[l].read(l, filename)) {
        //std::cout<<"construct"<<std::endl;
        A[l].construct(matgen, fix_rank ? (double)rank_func(l) : epi, cells.data(), Near, comm[l], A[l + 1], comm[l + 1], omega);
        //std::cout<<"write"<<std::endl;
        A[l].write(l, filename);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      read_time = MPI_Wtime() - read_time;
      if (mpi_rank == 0) {
        std::cout<<"Read time "<<read_time<<std::endl;
      }
    } else {
      //std::cout<<"construct"<<std::endl;
      A[l].construct(matgen, fix_rank ? (double)rank_func(l) : epi, cells.data(), Near, comm[l], A[l + 1], comm[l + 1], omega);
    }
    /*Qsize = A[l].Q.size() * sizeof(std::complex<double>);
    Rsize = A[l].R.size() * sizeof(std::complex<double>);
    Asize = A[l].A.size() * sizeof(std::complex<double>);
    Csize = A[l].C.size() * sizeof(std::complex<double>);
    Usize = A[l].U.size() * sizeof(std::complex<double>);
    Xsize = A[l].X.size() * sizeof(std::complex<double>);
    Ysize = A[l].Y.size() * sizeof(std::complex<double>);
    Zsize = A[l].Z.size() * sizeof(std::complex<double>);
    Wsize = A[l].W.size() * sizeof(std::complex<double>);
    total_size = Qsize + Rsize + Asize + Csize + Usize + Xsize + Ysize + Zsize + Wsize;
    for (int i = 0; i < mpi_size; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (mpi_rank == i) {
        std::cout<<"MPI rank "<<i<<std::endl;
        std::cout<<"  Total size on level "<<l<<": "<<total_size<<" bytes"<<std::endl;
        std::cout<<"  Qsize: "<<Qsize<<std::endl;
        std::cout<<"  Rsize: "<<Rsize<<std::endl;
        std::cout<<"  Asize: "<<Asize<<std::endl;
        std::cout<<"  Csize: "<<Csize<<std::endl;
        std::cout<<"  Usize: "<<Usize<<std::endl;
        std::cout<<"  Xsize: "<<Xsize<<std::endl;
        std::cout<<"  Ysize: "<<Ysize<<std::endl;
        std::cout<<"  Zsize: "<<Zsize<<std::endl;
        std::cout<<"  Wsize: "<<Wsize<<std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    comm[levels].level_sum(&total_size, 1);
    if (mpi_rank == 0) {
      std::cout<<"Total size (all processes) on level "<<l<<": "<<total_size<<" bytes"<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);*/
  }
  //std::cout<<"Finished Writing"<<std::endl;
  long long llen = comm[levels].lenLocal();
  long long gbegin = comm[levels].oGlobal();
  local_bodies = std::make_pair(cells[gbegin].Body[0], cells[gbegin + llen - 1].Body[1]);
}
/*
H2MatrixSolver::H2MatrixSolver(const Eigen::Ref<const Eigen::MatrixXcd> &mat, double epi, long long rank, long long leveled_rank, const std::vector<Cell>& cells, double theta, long long levels, const MatrixGenerator& matgen, double omega, double scale, MPI_Comm world) : 
  levels(levels), A(levels + 1), local_bodies(0, 0) {
  
  CSR Near('N', cells, cells, theta);
  CSR Far('F', cells, cells, theta);
  int mpi_size = 1;
  MPI_Comm_size(world, &mpi_size);

  std::vector<std::pair<long long, long long>> mapping(mpi_size, std::make_pair(0, 1));
  std::vector<std::pair<long long, long long>> tree(cells.size());
  std::transform(cells.begin(), cells.end(), tree.begin(), [](const Cell& c) { return std::make_pair(c.Child[0], c.Child[1]); });
  
  for (long long i = 0; i <= levels; i++) {
    comm.emplace_back(&tree[0], &mapping[0], Near.RowIndex.data(), Near.ColIndex.data(), Far.RowIndex.data(), Far.ColIndex.data(), allocedComm, world);
  }

  bool fix_rank = (epi == 0.);
  auto rank_func = [=](long long l) { return (levels - l) * leveled_rank + rank; };
  //std::vector<Hmatrix> wsa(levels + 1);
  //for (long long l = 1; l <= levels; l++)
  //  wsa[l].construct(epi, eval_d, rank_func(l), rank * 2, 2, comm[l].oGlobal(), comm[l].lenLocal(), cells.data(), fix_rank ? HSS_Far : Far, bodies, wsa[l - 1]);

  //std::vector<HiDR> hidr(levels + 1);
  //hidr[levels].initialize(0, comm[levels].oGlobal(), comm[levels].lenLocal(), cells.data());
  // I don't think I need to do anything for node 0
  //for (long long l = levels - 1; l > 0; l--) {
    //std::cout<<"Level "<<l<<std::endl;
    //hidr[l].bottom_up_sweep(0, comm[l].oGlobal(), comm[l].lenLocal(), cells.data(), hidr[l + 1]);
  //}
  //for (long long l = 1; l <= levels; l++) {
  //  std::cout<<"Level "<<l<<std::endl;
  //  hidr[l].top_down_sweep(0, cells.data(), Far, hidr[l - 1]);
  //}
  std::cout<<"Level "<<levels<<std::endl;
  A[levels].construct(mat, fix_rank ? (double)rank_func(levels) : epi, cells.data(), Near, comm[levels], A[levels], comm[levels], matgen, omega, scale);
  //A[levels].constructBLR(mat, fix_rank ? (double)rank_func(levels) : epi, cells.data(), Near, comm[levels], A[levels], comm[levels]);
  for (long long l = levels - 1; l >= 0; l--) {
    std::cout<<"Level "<<l<<std::endl;
    A[l].construct(mat, fix_rank ? (double)rank_func(l) : epi, cells.data(), Near, comm[l], A[l + 1], comm[l + 1], matgen, omega, scale);
  }

  long long llen = comm[levels].lenLocal();
  long long gbegin = comm[levels].oGlobal();
  local_bodies = std::make_pair(cells[gbegin].Body[0], cells[gbegin + llen - 1].Body[1]);
}*/
/*
H2MatrixSolver::H2MatrixSolver(const Eigen::Ref<const Eigen::MatrixXcd> &mat, double epi, long long rank, long long leveled_rank, const std::vector<Cell>& cells, double theta, long long levels, MPI_Comm world) : 
  levels(levels), A(levels + 1), local_bodies(0, 0) {
  
  CSR Near('N', cells, cells, theta);
  CSR Far('F', cells, cells, theta);
  int mpi_size = 1;
  MPI_Comm_size(world, &mpi_size);

  std::vector<std::pair<long long, long long>> mapping(mpi_size, std::make_pair(0, 1));
  std::vector<std::pair<long long, long long>> tree(cells.size());
  std::transform(cells.begin(), cells.end(), tree.begin(), [](const Cell& c) { return std::make_pair(c.Child[0], c.Child[1]); });
  
  for (long long i = 0; i <= levels; i++) {
    comm.emplace_back(&tree[0], &mapping[0], Near.RowIndex.data(), Near.ColIndex.data(), Far.RowIndex.data(), Far.ColIndex.data(), allocedComm, world);
  }

  bool fix_rank = (epi == 0.);
  auto rank_func = [=](long long l) { return (levels - l) * leveled_rank + rank; };
  //std::vector<Hmatrix> wsa(levels + 1);
  //for (long long l = 1; l <= levels; l++)
  //  wsa[l].construct(epi, eval_d, rank_func(l), rank * 2, 2, comm[l].oGlobal(), comm[l].lenLocal(), cells.data(), fix_rank ? HSS_Far : Far, bodies, wsa[l - 1]);

  //std::vector<HiDR> hidr(levels + 1);
  //hidr[levels].initialize(0, comm[levels].oGlobal(), comm[levels].lenLocal(), cells.data());
  // I don't think I need to do anything for node 0
  //for (long long l = levels - 1; l > 0; l--) {
    //std::cout<<"Level "<<l<<std::endl;
    //hidr[l].bottom_up_sweep(0, comm[l].oGlobal(), comm[l].lenLocal(), cells.data(), hidr[l + 1]);
  //}
  //for (long long l = 1; l <= levels; l++) {
  //  std::cout<<"Level "<<l<<std::endl;
  //  hidr[l].top_down_sweep(0, cells.data(), Far, hidr[l - 1]);
  //}
  std::cout<<"Level "<<levels<<std::endl;
  A[levels].construct(mat, fix_rank ? (double)rank_func(levels) : epi, cells.data(), Near, comm[levels], A[levels], comm[levels]);
  //A[levels].constructBLR(mat, fix_rank ? (double)rank_func(levels) : epi, cells.data(), Near, comm[levels], A[levels], comm[levels]);
  for (long long l = levels - 1; l >= 0; l--) {
    std::cout<<"Level "<<l<<std::endl;
    A[l].construct(mat, fix_rank ? (double)rank_func(l) : epi, cells.data(), Near, comm[l], A[l + 1], comm[l + 1]);
  }

  long long llen = comm[levels].lenLocal();
  long long gbegin = comm[levels].oGlobal();
  local_bodies = std::make_pair(cells[gbegin].Body[0], cells[gbegin + llen - 1].Body[1]);
}
*/
/*
H2MatrixSolver::H2MatrixSolver(const Eigen::Ref<const Eigen::MatrixXcd> &mat, double epi, long long rank, long long leveled_rank, const std::vector<Cell>& cells, double theta, long long levels, std::vector<double>& pts, MPI_Comm world) : 
  levels(levels), A(levels + 1), local_bodies(0, 0) {
  
  CSR Near('N', cells, cells, theta);
  CSR Far('F', cells, cells, theta);
  int mpi_size = 1;
  MPI_Comm_size(world, &mpi_size);

  std::vector<std::pair<long long, long long>> mapping(mpi_size, std::make_pair(0, 1));
  std::vector<std::pair<long long, long long>> tree(cells.size());
  std::transform(cells.begin(), cells.end(), tree.begin(), [](const Cell& c) { return std::make_pair(c.Child[0], c.Child[1]); });
  
  for (long long i = 0; i <= levels; i++) {
    comm.emplace_back(&tree[0], &mapping[0], Near.RowIndex.data(), Near.ColIndex.data(), Far.RowIndex.data(), Far.ColIndex.data(), allocedComm, world);
  }

  bool fix_rank = (epi == 0.);
  auto rank_func = [=](long long l) { return (levels - l) * leveled_rank + rank; };
  //std::vector<Hmatrix> wsa(levels + 1);
  //for (long long l = 1; l <= levels; l++)
  //  wsa[l].construct(epi, eval_d, rank_func(l), rank * 2, 2, comm[l].oGlobal(), comm[l].lenLocal(), cells.data(), fix_rank ? HSS_Far : Far, bodies, wsa[l - 1]);

  std::vector<HiDR> hidr(levels + 1);
  const long long r1 = 10;
  hidr[levels].initialize_grid(r1, comm[levels].oGlobal(), comm[levels].lenLocal(), cells.data(), pts, false);
  // I don't think I need to do anything for node 0
  for (long long l = levels - 1; l > 0; l--) {
    std::cout<<"Level "<<l<<std::endl;
    hidr[l].bottom_up_sweep_grid(r1, comm[l].oGlobal(), comm[l].lenLocal(), cells.data(), hidr[l + 1]);
  }
  for (long long l = 1; l <= levels; l++) {
    long long r2 = rank_func(l) / 3 + 6;
    std::cout<<"Level "<<l<<" r2 = " << r2<<std::endl;
    hidr[l].top_down_sweep_grid(r2, cells.data(), Far, hidr[l - 1], false);
  }
  std::cout<<"Levelx "<<levels<<std::endl;
  //A[levels].construct(mat, fix_rank ? (double)rank_func(levels) : epi, cells.data(), Near, hidr[levels], comm[levels], A[levels], comm[levels]);
  //A[levels].constructBLR(mat, fix_rank ? (double)rank_func(levels) : epi, cells.data(), Near, comm[levels], A[levels], comm[levels]);
  //A[levels].construct(mat, fix_rank ? (double)rank_func(levels) : epi, cells.data(), Near, comm[levels], A[levels], comm[levels]);
  A[levels].construct_sparse(mat, fix_rank ? (double)rank_func(levels) : epi, cells.data(), Near, comm[levels], A[levels], comm[levels]);
  for (long long l = levels - 1; l >= 0; l--) {
    std::cout<<"Level "<<l<<std::endl;
    //A[l].construct(mat, fix_rank ? (double)rank_func(l) : epi, cells.data(), Near, comm[l], A[l + 1], comm[l + 1]);
    A[l].construct_sparse(mat, fix_rank ? (double)rank_func(l) : epi, cells.data(), Near, comm[l], A[l + 1], comm[l + 1]);
    //A[l].construct(mat, fix_rank ? (double)rank_func(l) : epi, cells.data(), Near, hidr[l], comm[l], A[l + 1], comm[l + 1]);
  }

  long long llen = comm[levels].lenLocal();
  long long gbegin = comm[levels].oGlobal();
  local_bodies = std::make_pair(cells[gbegin].Body[0], cells[gbegin + llen - 1].Body[1]);
}*/

std::vector<double> extract_coords(const std::vector<elastWave3d::element>& elems) {
  std::vector<double> pts(elems.size() * 3);
  for (size_t i = 0; i < elems.size(); ++i) {
    pts[i * 3] = elems[i].xc[0];
    pts[i * 3 + 1] = elems[i].xc[1];
    pts[i * 3 + 2] = elems[i].xc[3];
  }
  return pts;
}

template <typename DT>
H2MatrixSolver<DT>::H2MatrixSolver(const MatrixGenerator& matgen, double epi, long long rank, long long leveled_rank, const std::vector<Cell>& cells, double theta, long long levels, double omega, const std::vector<elastWave3d::element>& elems, long long r1, long long leveled_r1, long long r2, long long leveled_r2, MPI_Comm world) : 
  levels(levels), A(levels + 1), local_bodies(0, 0) {
  
  CSR Near('N', cells, cells, theta);
  CSR Far('F', cells, cells, theta);
  
  int mpi_size = 1;
  MPI_Comm_size(world, &mpi_size);
  std::vector<std::pair<long long, long long>> mapping(mpi_size, std::make_pair(0, 1));
  std::vector<std::pair<long long, long long>> tree(cells.size());
  std::transform(cells.begin(), cells.end(), tree.begin(), [](const Cell& c) { return std::make_pair(c.Child[0], c.Child[1]); });
  for (long long i = 0; i <= levels; i++) {
    comm.emplace_back(&tree[0], &mapping[0], Near.RowIndex.data(), Near.ColIndex.data(), Far.RowIndex.data(), Far.ColIndex.data(), allocedComm, world);
  }

  bool fix_rank = (epi == 0.);
  auto rank_func = [=](long long l) { return (levels - l) * leveled_rank + rank; };
  if (fix_rank)
   Far = CSR('F', cells, cells, 0);
  

  std::vector<HiDR> hidr(levels + 1);
  //long long r1 = 10; // 20;
  auto pts = extract_coords(elems);
  //hidr[levels].initialize_grid(r1, comm[levels].oGlobal(), comm[levels].lenLocal(), cells.data(), pts, true);
  //hidr[levels].initialize_f(r1, comm[levels].oGlobal(), comm[levels].lenLocal(), cells.data(), pts);
  // currently HiDR does not work distributed so we just re-create it on each process
  long long Nleaf = (long long)1 << levels;
  hidr[levels].initialize_f(r1, Nleaf - 1, Nleaf, cells.data(), pts);
  // I don't think I need to do anything for node 0
  for (long long l = levels - 1; l > 0; l--) {
    //std::cout<<"Level "<<l<<std::endl;
    //r1 = 18;
    r1 *= leveled_r1;
    Nleaf >>= 1;
    hidr[l].bottom_up_sweep_f(r1, Nleaf - 1, Nleaf, cells.data(), hidr[l + 1]);
  }
  for (long long l = 1; l <= levels; l++) {
    //long long r2 = 10; //75;//32; //80; //rank_func(l) / 3 + 6;
    //std::cout<<"Level "<<l<<" r2 = " << r2<<std::endl;
    hidr[l].top_down_sweep_f((levels - l) * leveled_r2 + r2, cells.data(), Far, hidr[l - 1]);
  }

  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  //std::cout<<"Levelx "<<levels<<std::endl;
  A[levels].construct_hidr(matgen, fix_rank ? (double)rank_func(levels) : epi, cells.data(), Near, hidr[levels], comm[levels], A[levels], comm[levels], omega);
  for (long long l = levels - 1; l >= 0; l--) {
    if (mpi_rank == 0) {
      std::cout<<"Level "<<l<<std::endl;
    }
    A[l].construct_hidr(matgen, fix_rank ? (double)rank_func(l) : epi, cells.data(), Near, hidr[l], comm[l], A[l + 1], comm[l + 1], omega);
  }
  long long llen = comm[levels].lenLocal();
  long long gbegin = comm[levels].oGlobal();
  local_bodies = std::make_pair(cells[gbegin].Body[0], cells[gbegin + llen - 1].Body[1]);
}

template <typename DT>
H2MatrixSolver<DT>::H2MatrixSolver(const H2MatrixSolver& solver) :
  levels(solver.levels), local_bodies(solver.local_bodies) {
  // this should duplicate all the allocated communicators
  for (size_t i = 0; i < solver.allocedComm.size(); ++i) {
    MPI_Comm mpi_comm = MPI_COMM_NULL;
    MPI_Comm_dup(solver.allocedComm[i], &mpi_comm);
    allocedComm.emplace_back(mpi_comm);
  }
  for (size_t i = 0; i < solver.comm.size(); ++i) {
    comm.emplace_back(ColCommMPI(solver.comm[i], allocedComm));
  }
  A.reserve(solver.A.size());
  for (size_t i = 0; i < solver.A.size(); ++i) {
    A.emplace_back(H2Matrix(solver.A[i]));
  }
}
/*
void H2MatrixSolver::init_gpu_handles(const ncclComms nccl_comms) {
  desc.resize(levels + 1);
  long long bdim = *std::max_element(A[levels].Dims.begin(), A[levels].Dims.end());
  long long rank = *std::max_element(A[levels].DimsLr.begin(), A[levels].DimsLr.end());
  createMatrixDesc(&desc[levels], bdim, rank, deviceMatrixDesc_t(), comm[levels], nccl_comms);

  for (long long l = levels - 1; l >= 0; l--) {
    long long bdim = *std::max_element(A[l].Dims.begin(), A[l].Dims.end());
    long long rank = *std::max_element(A[l].DimsLr.begin(), A[l].DimsLr.end());
    createMatrixDesc(&desc[l], bdim, rank, desc[l + 1], comm[l], nccl_comms);
  }

  long long lenX = bdim * comm[levels].lenLocal();
  cudaMalloc(reinterpret_cast<void**>(&X_dev), lenX * sizeof(CUDA_CTYPE));
}

void H2MatrixSolver::allocSparseMV(deviceHandle_t handle, const ncclComms nccl_comms) {
  A_mv.resize(levels + 1);
  for (long long l = 0; l <= levels; l++) {
    createSpMatrixDesc(handle, &A_mv[l], l == levels, A[l].LowerZ, A[l].Dims.data(), A[l].DimsLr.data(), A[l].U[0], A[l].C[0], A[l].A[0], comm[l], nccl_comms);
  }
}*/
/*
void H2MatrixSolver::matVecMulSp(deviceHandle_t handle, std::complex<double> X[]) {
  if (levels < 0)
    return;

  long long lenX = A[levels].lenX;
  cudaMemcpy(X_dev, X, lenX * sizeof(std::complex<double>), cudaMemcpyHostToDevice);
  matVecDeviceH2(handle, levels, A_mv.data(), reinterpret_cast<std::complex<double>*>(X_dev));
  cudaMemcpy(X, X_dev, lenX * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
}
*/
template <typename DT>
void H2MatrixSolver<DT>::matVecMul(DT X[]) {
  if (levels < 0)
    return;

  A[levels].matVecUpwardPass(X, comm[levels]);
  for (long long l = levels - 1; l >= 0; l--){
    //std::cout<<"Level "<<l<<std::endl;
    A[l].matVecUpwardPass(A[l + 1].Z[0], comm[l]);
  }

  for (long long l = 0; l < levels; l++)
    A[l].matVecHorizontalandDownwardPass(A[l + 1].W[0], comm[l]);

  A[levels].matVecLeafHorizontalPass(X, comm[levels]);
}


/*
void H2MatrixSolver::matVecMulDense(const std::complex<double> X[], std::complex<double> Y[]) {
  if (levels < 0)
    return;

  A[levels].matVecDense(X, Y, comm[levels]);
}*/

template <typename DT>
void H2MatrixSolver<DT>::factorizeM() {
  for (long long l = levels; l >= 0; l--) {
    A[l].factorize(comm[l]);
    if (0 < l)
      A[l - 1].factorizeCopyNext(A[l], comm[l]);
  }

  for (long long l = levels; l >= 0; l--)
    if (A[l].info)
      printf("singularity detected at level %lld.\n", l);
}
/*
void H2MatrixSolver::factorizeDeviceM(deviceHandle_t handle) {
  copyDataInMatrixDesc(desc[levels], A[levels].A[0], A[levels].Q[0], handle->compute_stream);
  compute_factorize(handle, desc[levels], deviceMatrixDesc_t());

  for (long long l = levels - 1; l >= 0; l--) {
    copyDataInMatrixDesc(desc[l], A[l].A[0], A[l].Q[0], handle->memory_stream);
    cudaDeviceSynchronize();
    copyDataOutMatrixDesc(desc[l + 1], A[l + 1].A[0], A[l + 1].R[desc[l + 1].diag_offset], handle->memory_stream);
    compute_factorize(handle, desc[l], desc[l + 1]);
  }

  copyDataOutMatrixDesc(desc[0], A[0].A[0], A[0].R[0], handle->compute_stream);
  cudaDeviceSynchronize();

  for (long long l = levels; l >= 0; l--)
    if (check_info(desc[l], comm[l]))
      printf("singularity detected at level %lld.\n", l);
}
*/

template <typename DT>
void H2MatrixSolver<DT>::solvePrecondition(DT X[]) {
  if (levels < 0)
    return;

  A[levels].forwardSubstitute(X, comm[levels]);
  for (long long l = levels - 1; l >= 0; l--)
    A[l].forwardSubstitute(A[l + 1].Z[0], comm[l]);

  for (long long l = 0; l < levels; l++)
    A[l].backwardSubstitute(A[l + 1].W[0], comm[l]);
  A[levels].backwardSubstitute(X, comm[levels]);
}

/*
void H2MatrixSolver::solvePreconditionDevice(deviceHandle_t handle, std::complex<double> X[]) {
  if (levels < 0)
    return;

  long long lenX = A[levels].lenX;
  cudaMemcpy(X_dev, X, lenX * sizeof(std::complex<double>), cudaMemcpyHostToDevice);
  matSolvePreconditionDeviceH2(handle, levels, desc.data(), reinterpret_cast<std::complex<double>*>(X_dev));
  cudaMemcpy(X, X_dev, lenX * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
}
*/

template <typename DT>
void H2MatrixSolver<DT>::solveGMRES(double tol, H2MatrixSolver& M, DT x[], const DT b[], long long inner_iters, long long outer_iters) {
  typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> Vector_dt;
  typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;

  long long N = A[levels].lenX;
  long long ld = inner_iters + 1;

  Eigen::Map<const Vector_dt> B(b, N);
  Eigen::Map<Vector_dt> X(x, N);

  DT nsum = B.adjoint() * B;
  comm[levels].level_sum(&nsum, 1);
  double normb = std::sqrt(nsum.real());
  if (normb == 0.)
    normb = 1.;

  Vector_dt R = B;
  resid.resize(outer_iters + 1);
  resid[0] = 1.;
  iters = 0;

  while (iters < outer_iters && tol <= resid[iters]) {
    M.solvePrecondition(R.data());
    nsum = R.adjoint() * R;
    comm[levels].level_sum(&nsum, 1);

    double beta = std::sqrt(nsum.real());
    Matrix_dt H = Matrix_dt::Zero(ld, inner_iters);
    Matrix_dt v = Matrix_dt::Zero(N, ld);
    v.col(0) = R * (1. / beta);
    
    for (long long i = 0; i < inner_iters; i++) {
      R = v.col(i);
      matVecMul(R.data());
      M.solvePrecondition(R.data());

      H.block(0, i, i + 1, 1).noalias() = v.leftCols(i + 1).adjoint() * R;
      comm[levels].level_sum(H.col(i).data(), i + 1);
      R.noalias() -= v.leftCols(i + 1) * H.block(0, i, i + 1, 1);

      nsum = R.adjoint() * R;
      comm[levels].level_sum(&nsum, 1);
      H(i + 1, i) = std::sqrt(nsum.real());
      v.col(i + 1) = R * ((DT)1. / H(i + 1, i));
    }

    Vector_dt s = Vector_dt::Zero(ld);
    s(0) = beta;

    R = H.householderQr().solve(s);
    X.noalias() += v.leftCols(inner_iters) * R;

    R = -X;
    matVecMul(R.data());
    R += B;

    nsum = R.adjoint() * R;
    comm[levels].level_sum(&nsum, 1);
    resid[++iters] = std::sqrt(nsum.real()) / normb;
  }
}

template <typename DT>
void H2MatrixSolver<DT>::solveGMRES(double tol, DT x[], const DT b[], long long inner_iters, long long outer_iters) {
  typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> Vector_dt;
  typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;

  long long N = A[levels].lenX;
  long long ld = inner_iters + 1;

  Eigen::Map<const Vector_dt> B(b, N);
  Eigen::Map<Vector_dt> X(x, N);

  DT nsum = B.adjoint() * B;
  comm[levels].level_sum(&nsum, 1);
  double normb = std::sqrt(nsum.real());
  if (normb == 0.)
    normb = 1.;

  Vector_dt R = B;
  resid.resize(outer_iters + 1);
  resid[0] = 1.;
  iters = 0;

  while (iters < outer_iters && tol <= resid[iters]) {
    nsum = R.adjoint() * R;
    comm[levels].level_sum(&nsum, 1);

    double beta = std::sqrt(nsum.real());
    Matrix_dt H = Matrix_dt::Zero(ld, inner_iters);
    Matrix_dt v = Matrix_dt::Zero(N, ld);
    v.col(0) = R * (1. / beta);
    
    for (long long i = 0; i < inner_iters; i++) {
      R = v.col(i);
      matVecMul(R.data());

      H.block(0, i, i + 1, 1).noalias() = v.leftCols(i + 1).adjoint() * R;
      comm[levels].level_sum(H.col(i).data(), i + 1);
      R.noalias() -= v.leftCols(i + 1) * H.block(0, i, i + 1, 1);

      nsum = R.adjoint() * R;
      comm[levels].level_sum(&nsum, 1);
      H(i + 1, i) = std::sqrt(nsum.real());
      v.col(i + 1) = R * ((DT)1. / H(i + 1, i));
    }

    Vector_dt s = Vector_dt::Zero(ld);
    s(0) = beta;

    R = H.householderQr().solve(s);
    X.noalias() += v.leftCols(inner_iters) * R;

    R = -X;
    matVecMul(R.data());
    R += B;

    nsum = R.adjoint() * R;
    comm[levels].level_sum(&nsum, 1);
    resid[++iters] = std::sqrt(nsum.real()) / normb;
  }
}

template <typename DT> template <typename OT>
void H2MatrixSolver<DT>::solveGMRES(double tol, H2MatrixSolver<OT>& M, DT x[], const DT b[], long long inner_iters, long long outer_iters) {
  typedef Eigen::Matrix<DT, Eigen::Dynamic, 1> Vector_dt;
  typedef Eigen::Matrix<OT, Eigen::Dynamic, 1> Vector_ot;
  typedef Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic> Matrix_dt;

  long long N = A[levels].lenX;
  long long ld = inner_iters + 1;

  Eigen::Map<const Vector_dt> B(b, N);
  Eigen::Map<Vector_dt> X(x, N);

  double nsum = B.squaredNorm();
  comm[levels].level_sum(&nsum, 1);
  double normb = std::sqrt(nsum);
  if (normb == 0.)
    normb = 1.;

  Vector_dt R = B;
  Vector_ot R_low;
  resid.resize(outer_iters + 1);
  resid[0] = 1.;
  iters = 0;

  while (iters < outer_iters && tol <= resid[iters]) {
    R_low = R.template cast<OT>();
    M.solvePrecondition(R_low.data());
    R = R_low.template cast<DT>();
    nsum = R.squaredNorm();
    comm[levels].level_sum(&nsum, 1);

    double beta = std::sqrt(nsum);
    Matrix_dt H = Matrix_dt::Zero(ld, inner_iters);
    Matrix_dt v = Matrix_dt::Zero(N, ld);
    v.col(0) = R * (1. / beta);
    
    for (long long i = 0; i < inner_iters; i++) {
      R = v.col(i);
      matVecMul(R.data());
      R_low = R.template cast<OT>();
      M.solvePrecondition(R_low.data());
      R = R_low.template cast<DT>();

      H.block(0, i, i + 1, 1).noalias() = v.leftCols(i + 1).adjoint() * R;
      comm[levels].level_sum(H.col(i).data(), i + 1);
      R.noalias() -= v.leftCols(i + 1) * H.block(0, i, i + 1, 1);

      nsum = R.squaredNorm();
      comm[levels].level_sum(&nsum, 1);
      H(i + 1, i) = std::sqrt(nsum);
      v.col(i + 1) = R * ((DT)1. / H(i + 1, i));
    }

    Vector_dt s = Vector_dt::Zero(ld);
    s(0) = beta;

    R = H.householderQr().solve(s);
    X.noalias() += v.leftCols(inner_iters) * R;

    R = -X;
    matVecMul(R.data());
    R += B;

    nsum = R.squaredNorm();
    comm[levels].level_sum(&nsum, 1);
    resid[++iters] = std::sqrt(nsum) / normb;
  }
}


/*
void H2MatrixSolver::solveGMRESDense(double tol, std::complex<double> x[], const std::complex<double> b[], long long inner_iters, long long outer_iters) {
  //std::cout<<"START GMRES"<<std::endl;
  long long n_mat = A[levels].n_mat;
  long long N = A[levels].lenX;
  long long ld = inner_iters + 1;
  //std::cout<<N<<" "<<n_mat<<std::endl;
  
  int mpi_size = 1;
  //int mpi_rank = 0, mpi_size = 1;
  //MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  long long offset = local_bodies.second * 3;
  //std::cout<<mpi_rank<<" Offset "<<offset<<std::endl;
  std::vector<long long> offsets(mpi_size+1);
  MPI_Allgather(&offset, 1, MPI_LONG_LONG_INT, &offsets[1], 1, MPI_LONG_LONG_INT, MPI_COMM_WORLD);
  std::vector<int> recvcounts(offsets.size() - 1), displs(offsets.size() - 1);
  std::transform(offsets.begin() + 1, offsets.end(), offsets.begin(), recvcounts.begin(), [](long long end, long long begin) { return (int)(end - begin); });
  std::transform(offsets.begin(), std::prev(offsets.end()), displs.begin(), [](long long begin) { return (int)begin; });

  Eigen::Map<const Eigen::VectorXcd> B(b, N);
  Eigen::Map<Eigen::VectorXcd> X(x, N);
  Eigen::VectorXcd global(n_mat);

  std::complex<double> nsum = B.adjoint() * B;
  comm[levels].level_sum(&nsum, 1);
  double normb = std::sqrt(nsum.real());
  if (normb == 0.)
    normb = 1.;

  Eigen::VectorXcd R = B;
  resid.resize(outer_iters + 1);
  resid[0] = 1.;
  iters = 0;

  while (iters < outer_iters && tol <= resid[iters]) {
    //std::cout<<"Iteration"<<std::endl;
    solvePrecondition(R.data());
    nsum = R.adjoint() * R;
    comm[levels].level_sum(&nsum, 1);

    double beta = std::sqrt(nsum.real());
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(ld, inner_iters);
    Eigen::MatrixXcd v = Eigen::MatrixXcd::Zero(N, ld);
    v.col(0) = R * (1. / beta);
    
    for (long long i = 0; i < inner_iters; i++) {
      //R = v.col(i);
      //matVecMul(R.data());
      // here we need to use the new matvec
      // R is only the local part, so that's fine, but v should be the full vector but it seems v is only local
      // this code will break if N is not equal on all processes
      //std::cout<<"Before Gather "<<v.col(i).data()<<std::endl;
      //std::cout<<"Before Gather "<<global.data()<<std::endl;
      //MPI_Allgather(v.col(i).data(), N, MPI_C_DOUBLE_COMPLEX, global.data(), N, MPI_C_DOUBLE_COMPLEX, MPI_COMM_WORLD);
      MPI_Allgatherv(v.col(i).data(), N, MPI_C_DOUBLE_COMPLEX, global.data(), &recvcounts[0], &displs[0], MPI_C_DOUBLE_COMPLEX, MPI_COMM_WORLD);
      //std::cout<<"Gather"<<std::endl;
      //R = mat * v.col(i);
      matVecMulDense(global.data(), R.data());
      //std::cout<<"MAtvec"<<std::endl;
      solvePrecondition(R.data());

      H.block(0, i, i + 1, 1).noalias() = v.leftCols(i + 1).adjoint() * R;
      comm[levels].level_sum(H.col(i).data(), i + 1);
      R.noalias() -= v.leftCols(i + 1) * H.block(0, i, i + 1, 1);

      nsum = R.adjoint() * R;
      comm[levels].level_sum(&nsum, 1);
      H(i + 1, i) = std::sqrt(nsum.real());
      v.col(i + 1) = R * (1. / H(i + 1, i));
    }

    Eigen::VectorXcd s = Eigen::VectorXcd::Zero(ld);
    s(0) = beta;
    R = H.householderQr().solve(s);
    X.noalias() += v.leftCols(inner_iters) * R;

    R = -X;
    //matVecMul(R.data());
    // and here also
    //MPI_Allgather(R.data(), N, MPI_C_DOUBLE_COMPLEX, global.data(), N, MPI_C_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Allgatherv(R.data(), N, MPI_C_DOUBLE_COMPLEX, global.data(), &recvcounts[0], &displs[0], MPI_C_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    matVecMulDense(global.data(), R.data());
    //R = mat * R;
    R += B;

    nsum = R.adjoint() * R;
    comm[levels].level_sum(&nsum, 1);
    resid[++iters] = std::sqrt(nsum.real()) / normb;
  }
}

void H2MatrixSolver::solveGMRESDense(double tol, const Eigen::Ref<const Eigen::MatrixXcd>& mat, std::complex<double> x[], const std::complex<double> b[], long long inner_iters, long long outer_iters) {
  long long N = A[levels].lenX;
  long long ld = inner_iters + 1;

  Eigen::Map<const Eigen::VectorXcd> B(b, N);
  Eigen::Map<Eigen::VectorXcd> X(x, N);

  std::complex<double> nsum = B.adjoint() * B;
  comm[levels].level_sum(&nsum, 1);
  double normb = std::sqrt(nsum.real());
  if (normb == 0.)
    normb = 1.;

  Eigen::VectorXcd R = B;
  resid.resize(outer_iters + 1);
  resid[0] = 1.;
  iters = 0;

  while (iters < outer_iters && tol <= resid[iters]) {
    solvePrecondition(R.data());
    nsum = R.adjoint() * R;
    comm[levels].level_sum(&nsum, 1);

    double beta = std::sqrt(nsum.real());
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(ld, inner_iters);
    Eigen::MatrixXcd v = Eigen::MatrixXcd::Zero(N, ld);
    v.col(0) = R * (1. / beta);
    
    for (long long i = 0; i < inner_iters; i++) {
      //R = v.col(i);
      //matVecMul(R.data());
      R = mat * v.col(i);
      solvePrecondition(R.data());

      H.block(0, i, i + 1, 1).noalias() = v.leftCols(i + 1).adjoint() * R;
      comm[levels].level_sum(H.col(i).data(), i + 1);
      R.noalias() -= v.leftCols(i + 1) * H.block(0, i, i + 1, 1);

      nsum = R.adjoint() * R;
      comm[levels].level_sum(&nsum, 1);
      H(i + 1, i) = std::sqrt(nsum.real());
      v.col(i + 1) = R * (1. / H(i + 1, i));
    }

    Eigen::VectorXcd s = Eigen::VectorXcd::Zero(ld);
    s(0) = beta;
    R = H.householderQr().solve(s);
    X.noalias() += v.leftCols(inner_iters) * R;

    R = -X;
    //matVecMul(R.data());
    R = mat * R;
    R += B;

    nsum = R.adjoint() * R;
    comm[levels].level_sum(&nsum, 1);
    resid[++iters] = std::sqrt(nsum.real()) / normb;
  }
}

void H2MatrixSolver::solveGMRESDenseNoPrecon(double tol, const Eigen::Ref<const Eigen::MatrixXcd>& mat, std::complex<double> x[], const std::complex<double> b[], long long inner_iters, long long outer_iters) {
  long long N = A[levels].lenX;
  long long ld = inner_iters + 1;

  Eigen::Map<const Eigen::VectorXcd> B(b, N);
  Eigen::Map<Eigen::VectorXcd> X(x, N);

  std::complex<double> nsum = B.adjoint() * B;
  comm[levels].level_sum(&nsum, 1);
  double normb = std::sqrt(nsum.real());
  if (normb == 0.)
    normb = 1.;

  Eigen::VectorXcd R = B;
  resid.resize(outer_iters + 1);
  resid[0] = 1.;
  iters = 0;

  while (iters < outer_iters && tol <= resid[iters]) {
    //solvePrecondition(R.data());
    nsum = R.adjoint() * R;
    comm[levels].level_sum(&nsum, 1);

    double beta = std::sqrt(nsum.real());
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(ld, inner_iters);
    Eigen::MatrixXcd v = Eigen::MatrixXcd::Zero(N, ld);
    v.col(0) = R * (1. / beta);
    
    for (long long i = 0; i < inner_iters; i++) {
      //R = v.col(i);
      //matVecMul(R.data());
      R = mat * v.col(i);
      //solvePrecondition(R.data());

      H.block(0, i, i + 1, 1).noalias() = v.leftCols(i + 1).adjoint() * R;
      comm[levels].level_sum(H.col(i).data(), i + 1);
      R.noalias() -= v.leftCols(i + 1) * H.block(0, i, i + 1, 1);

      nsum = R.adjoint() * R;
      comm[levels].level_sum(&nsum, 1);
      H(i + 1, i) = std::sqrt(nsum.real());
      v.col(i + 1) = R * (1. / H(i + 1, i));
    }

    Eigen::VectorXcd s = Eigen::VectorXcd::Zero(ld);
    s(0) = beta;
    R = H.householderQr().solve(s);
    X.noalias() += v.leftCols(inner_iters) * R;

    R = -X;
    //matVecMul(R.data());
    R = mat * R;
    R += B;

    nsum = R.adjoint() * R;
    comm[levels].level_sum(&nsum, 1);
    resid[++iters] = std::sqrt(nsum.real()) / normb;
  }
}

void H2MatrixSolver::solveGMRESDensePrecon(double tol, const Eigen::PartialPivLU<Eigen::MatrixXcd>& precon, const Eigen::Ref<const Eigen::MatrixXcd>& mat, std::complex<double> x[], const std::complex<double> b[], long long inner_iters, long long outer_iters) {
  long long N = A[levels].lenX;
  long long ld = inner_iters + 1;

  Eigen::Map<const Eigen::VectorXcd> B(b, N);
  Eigen::Map<Eigen::VectorXcd> X(x, N);

  std::complex<double> nsum = B.adjoint() * B;
  comm[levels].level_sum(&nsum, 1);
  double normb = std::sqrt(nsum.real());
  if (normb == 0.)
    normb = 1.;

  Eigen::VectorXcd R = B;
  resid.resize(outer_iters + 1);
  resid[0] = 1.;
  iters = 0;

  while (iters < outer_iters && tol <= resid[iters]) {
    //solvePrecondition(R.data());
    R = precon.solve(R);
    nsum = R.adjoint() * R;
    comm[levels].level_sum(&nsum, 1);

    double beta = std::sqrt(nsum.real());
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(ld, inner_iters);
    Eigen::MatrixXcd v = Eigen::MatrixXcd::Zero(N, ld);
    v.col(0) = R * (1. / beta);
    
    for (long long i = 0; i < inner_iters; i++) {
      //R = v.col(i);
      //matVecMul(R.data());
      R = mat * v.col(i);
      R = precon.solve(R);
      //solvePrecondition(R.data());

      H.block(0, i, i + 1, 1).noalias() = v.leftCols(i + 1).adjoint() * R;
      comm[levels].level_sum(H.col(i).data(), i + 1);
      R.noalias() -= v.leftCols(i + 1) * H.block(0, i, i + 1, 1);

      nsum = R.adjoint() * R;
      comm[levels].level_sum(&nsum, 1);
      H(i + 1, i) = std::sqrt(nsum.real());
      v.col(i + 1) = R * (1. / H(i + 1, i));
    }

    Eigen::VectorXcd s = Eigen::VectorXcd::Zero(ld);
    s(0) = beta;
    R = H.householderQr().solve(s);
    X.noalias() += v.leftCols(inner_iters) * R;

    R = -X;
    //matVecMul(R.data());
    R = mat * R;
    R += B;

    nsum = R.adjoint() * R;
    comm[levels].level_sum(&nsum, 1);
    resid[++iters] = std::sqrt(nsum.real()) / normb;
  }
}*/
/*
void H2MatrixSolver::solveGMRESDevice(deviceHandle_t handle, double tol, H2MatrixSolver& M, std::complex<double> X[], const std::complex<double> B[], long long inner_iters, long long outer_iters, const ncclComms nccl_comms) {
  resid.resize(outer_iters + 1);
  iters = solveDeviceGMRES(handle, levels, A_mv.data(), M.levels, M.desc.data(), tol, X, B, inner_iters, outer_iters, resid.data(), comm[levels], nccl_comms);
}
*/

template <typename DT>
void H2MatrixSolver<DT>::free_all_comms() {
  for (MPI_Comm& c : allocedComm)
    MPI_Comm_free(&c);
  allocedComm.clear();
}

template <typename DT>
void H2MatrixSolver<DT>::freeSparseMV() {
  for (long long l = 0; l <= levels; l++) {
    destroySpMatrixDesc(A_mv[l]);
  }
  A_mv.clear();
}

template <typename DT>
void H2MatrixSolver<DT>::free_gpu_handles() {
  for (long long l = levels; l >= 0; l--) {
    destroyMatrixDesc(desc[l]);
  }
  desc.clear();
  cudaFree(X_dev);
}

template <typename DT>
double solveRelErr(long long lenX, const DT X[], const DT ref[], MPI_Comm world) {
  double err[2] = { 0., 0. };
  for (long long i = 0; i < lenX; i++) {
    DT diff = X[i] - ref[i];
    err[0] = err[0] + (diff.real() * diff.real());
    err[1] = err[1] + (ref[i].real() * ref[i].real());
    //std::cout<<X[i]<<" - "<<ref[i] << " = " << diff << std::endl;
  }
  //std::cout<<"Error local " <<err[0]<<" "<<err[1]<<std::endl;
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, world);
  //std::cout<<"Error " <<err[0]<<" "<<err[1]<<std::endl;
  return std::sqrt(err[0] / err[1]);
}
