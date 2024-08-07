
#include <comm-mpi.hpp>
#include <data_container.hpp>

#include <algorithm>
#include <set>
#include <iostream>

// explicit template instantiation
template void ColCommMPI::level_merge<std::complex<double>>(std::complex<double>*, long long) const;
template void ColCommMPI::level_sum<std::complex<double>>(std::complex<double>*, long long) const;
template void ColCommMPI::level_merge<double>(double*, long long) const;
template void ColCommMPI::level_sum<double>(double*, long long) const;
template void ColCommMPI::neighbor_bcast<long long>(long long*) const;
template void ColCommMPI::neighbor_bcast<std::complex<double>>(MatrixDataContainer<std::complex<double>>&) const;
template void ColCommMPI::neighbor_bcast<double>(MatrixDataContainer<double>& dc) const;

MPI_Comm MPI_Comm_split_unique(std::vector<MPI_Comm>& allocedComm, int color, int mpi_rank, MPI_Comm world) {
  MPI_Comm comm = MPI_COMM_NULL;
  MPI_Comm_split(world, color, mpi_rank, &comm);

  if (comm != MPI_COMM_NULL) {
    auto iter = std::find_if(allocedComm.begin(), allocedComm.end(), [comm](MPI_Comm c) -> bool { 
      int result; MPI_Comm_compare(comm, c, &result); return result == MPI_IDENT || result == MPI_CONGRUENT; });
    if (iter == allocedComm.end())
      allocedComm.emplace_back(comm);
    else {
      MPI_Comm_free(&comm);
      comm = *iter;
    }
  }
  return comm;
}

void getNextLevelMapping(std::pair<long long, long long> Mapping[], const std::pair<long long, long long> Tree[], long long mpi_size) {
  long long p = 0;
  std::vector<std::pair<long long, long long>> MappingNext(mpi_size, std::make_pair(-1, -1));

  while (p < mpi_size) {
    long long lenP = std::distance(&Mapping[p], std::find_if_not(&Mapping[p], &Mapping[mpi_size], 
      [&](std::pair<long long, long long> a) { return a == Mapping[p]; }));
    long long pbegin = Mapping[p].first;
    long long pend = Mapping[p].second;
    long long child = Tree[pbegin].first;
    long long lenC = Tree[pend - 1].second - child;

    if (child >= 0 && lenC > 0)
      for (long long j = 0; j < lenP; j++) {
        long long c0 = j * lenC / lenP;
        long long c1 = (j + 1) * lenC / lenP;
        c1 = std::max(c1, c0 + 1);
        MappingNext[p + j] = std::make_pair(child + c0, child + c1);
      }
    p += lenP;
  }
  std::copy(MappingNext.begin(), MappingNext.end(), Mapping);
}

ColCommMPI::ColCommMPI(const std::pair<long long, long long> Tree[], std::pair<long long, long long> Mapping[], const long long Rows[], const long long Cols[], std::vector<MPI_Comm>& allocedComm, MPI_Comm world) {
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(world, &mpi_rank);
  MPI_Comm_size(world, &mpi_size);

  // gets the mapping for the mpi_rank (first element)
  long long pbegin = Mapping[mpi_rank].first;
  // second element
  long long pend = Mapping[mpi_rank].second;
  // looks for the first element in the mapping that has the same values as the current rank
  // Then calculates the distance from the first element
  long long p = std::distance(&Mapping[0], std::find(&Mapping[0], &Mapping[mpi_rank], Mapping[mpi_rank]));
  // find the next mapping whose values are not identical (from the previously found one)
  // and calculate the distance between those two
  long long lenp = std::distance(&Mapping[p], 
    std::find_if_not(&Mapping[p], &Mapping[mpi_size], [&](std::pair<long long, long long> i) { return i == Mapping[mpi_rank]; }));
  
  
  // my current guess is that the mapping stores the columns (for each level) that are stored on this process (first, last)
  // and this function finds the mpi_rank for any given column
  auto col_to_mpi_rank = [&](long long col) { return std::distance(&Mapping[0], std::find_if(&Mapping[0], &Mapping[mpi_size], 
    [=](std::pair<long long, long long> i) { return i.first <= col && col < i.second; })); };
  // stores the mpir rank for each column on that level
  std::set<long long> cols;
  std::for_each(&Cols[Rows[pbegin]], &Cols[Rows[pend]], [&](long long col) { cols.insert(col_to_mpi_rank(col)); });

  std::vector<long long> NeighborRanks(cols.begin(), cols.end());
  Proc = std::distance(NeighborRanks.begin(), std::find(NeighborRanks.begin(), NeighborRanks.end(), p));
  Boxes = std::vector<std::pair<long long, long long>>(NeighborRanks.size());
  NeighborComm = std::vector<std::pair<int, MPI_Comm>>(p == mpi_rank ? NeighborRanks.size() : 0);

  for (long long i = 0; i < (long long)NeighborRanks.size(); i++) {
    long long ibegin = Mapping[NeighborRanks[i]].first;
    long long iend = Mapping[NeighborRanks[i]].second;
    std::set<long long> icols;
    std::for_each(&Cols[Rows[ibegin]], &Cols[Rows[iend]], [&](long long col) { icols.insert(col_to_mpi_rank(col)); });

    Boxes[i] = std::make_pair(ibegin, iend - ibegin);
    if (p == mpi_rank)
      NeighborComm[i].first = std::distance(icols.begin(), std::find(icols.begin(), icols.end(), NeighborRanks[i]));
  }

  long long k = 0;
  for (int i = 0; i < mpi_size; i++) {
    k = std::distance(NeighborRanks.begin(), std::find_if(NeighborRanks.begin() + k, NeighborRanks.end(), [=](long long a) { return (long long)i <= a; }));
    MPI_Comm comm = MPI_Comm_split_unique(allocedComm, (p == mpi_rank && k != (long long)NeighborRanks.size() && NeighborRanks[k] == i) ? 1 : MPI_UNDEFINED, mpi_rank, world);
    if (comm != MPI_COMM_NULL)
      NeighborComm[k].second = comm;
  }

  getNextLevelMapping(&Mapping[0], Tree, mpi_size);
  long long p_next = std::distance(&Mapping[0], std::find(&Mapping[0], &Mapping[mpi_rank], Mapping[mpi_rank]));
  MergeComm = MPI_Comm_split_unique(allocedComm, (lenp > 1 && mpi_rank == p_next) ? p : MPI_UNDEFINED, mpi_rank, world);
  AllReduceComm = MPI_Comm_split_unique(allocedComm, mpi_rank == p ? 1 : MPI_UNDEFINED, mpi_rank, world);
  DupComm = MPI_Comm_split_unique(allocedComm, lenp > 1 ? p : MPI_UNDEFINED, mpi_rank, world);
}

long long ColCommMPI::iLocal(long long iglobal) const {
  std::vector<std::pair<long long, long long>>::const_iterator iter = std::find_if(Boxes.begin(), Boxes.end(), 
    [=](std::pair<long long, long long> i) { return i.first <= iglobal && iglobal < i.first + i.second; });
  return (0 <= iglobal && iter != Boxes.end()) ? (iglobal - (*iter).first + std::accumulate(Boxes.begin(), iter, 0ll, 
    [](const long long& init, std::pair<long long, long long> i) { return init + i.second; })) : -1;
}

long long ColCommMPI::iGlobal(long long ilocal) const {
  long long iter = 0;
  while (iter < (long long)Boxes.size() && Boxes[iter].second <= ilocal) {
    ilocal = ilocal - Boxes[iter].second;
    iter = iter + 1;
  }
  return (0 <= ilocal && iter <= (long long)Boxes.size()) ? (Boxes[iter].first + ilocal) : -1;
}

long long ColCommMPI::oLocal() const {
  return 0 <= Proc ? std::accumulate(Boxes.begin(), Boxes.begin() + Proc, 0ll,
    [](const long long& init, const std::pair<long long, long long>& p) { return init + p.second; }) : -1;
}

long long ColCommMPI::oGlobal() const {
  return 0 <= Proc ? Boxes[Proc].first : -1;
}

long long ColCommMPI::lenLocal() const {
  return 0 <= Proc ? Boxes[Proc].second : 0;
}

long long ColCommMPI::lenNeighbors() const {
  return 0 <= Proc ? std::accumulate(Boxes.begin(), Boxes.end(), 0ll,
    [](const long long& init, const std::pair<long long, long long>& p) { return init + p.second; }) : 0; 
}

template<class T> inline MPI_Datatype get_mpi_datatype() {
  if (typeid(T) == typeid(long long))
    return MPI_LONG_LONG_INT;
  if (typeid(T) == typeid(double))
    return MPI_DOUBLE;
  if (typeid(T) == typeid(std::complex<double>))
    return MPI_C_DOUBLE_COMPLEX;
  return MPI_DATATYPE_NULL;
}

template <typename DT>
void ColCommMPI::level_merge(DT* data, long long len) const {
  record_mpi();
  if (MergeComm != MPI_COMM_NULL)
    MPI_Allreduce(MPI_IN_PLACE, data, len, get_mpi_datatype<DT>(), MPI_SUM, MergeComm);
  if (DupComm != MPI_COMM_NULL)
    MPI_Bcast(data, len, get_mpi_datatype<DT>(), 0, DupComm);
  record_mpi();
}

template <typename DT>
void ColCommMPI::level_sum(DT* data, long long len) const {
  record_mpi();
  if (AllReduceComm != MPI_COMM_NULL)
    MPI_Allreduce(MPI_IN_PLACE, data, len, get_mpi_datatype<DT>(), MPI_SUM, AllReduceComm);
  if (DupComm != MPI_COMM_NULL)
    MPI_Bcast(data, len, get_mpi_datatype<DT>(), 0, DupComm);
  record_mpi();
}

template <typename DT>
void ColCommMPI::neighbor_bcast(DT* data) const {
  std::vector<long long> offsets(Boxes.size() + 1, 0);
  std::transform_inclusive_scan(Boxes.begin(), Boxes.end(), offsets.begin() + 1, std::plus<long long>(), 
    [](const std::pair<long long, long long>& p) { return p.second; });

  record_mpi();
  for (long long p = 0; p < (long long)NeighborComm.size(); p++) {
    long long llen = offsets[p + 1] - offsets[p];
    MPI_Bcast(&data[offsets[p]], llen, get_mpi_datatype<DT>(), NeighborComm[p].first, NeighborComm[p].second);
  }
  if (DupComm != MPI_COMM_NULL)
    MPI_Bcast(data, offsets.back(), get_mpi_datatype<DT>(), 0, DupComm);
  record_mpi();
}

template <typename DT>
void ColCommMPI::neighbor_bcast(MatrixDataContainer<DT>& dc) const {
  std::vector<long long> offsets(Boxes.size() + 1, 0);
  std::transform_inclusive_scan(Boxes.begin(), Boxes.end(), offsets.begin() + 1, std::plus<long long>(), 
    [](const std::pair<long long, long long>& p) { return p.second; });
  
  record_mpi();
  for (long long p = 0; p < (long long)NeighborComm.size(); p++) {
    DT* start = dc[offsets[p]], *end = dc[offsets[p + 1]];
    MPI_Bcast(start, std::distance(start, end), get_mpi_datatype<DT>(), NeighborComm[p].first, NeighborComm[p].second);
  }
  if (DupComm != MPI_COMM_NULL)
    MPI_Bcast(dc[0], dc.size(), get_mpi_datatype<DT>(), 0, DupComm);
  record_mpi();
}

std::pair<double, double> timer = std::make_pair(0., 0.);

double ColCommMPI::get_comm_time() {
  double lapse = timer.first;
  timer = std::make_pair(0., 0.);
  return lapse;
}

void ColCommMPI::record_mpi() {
  if (timer.second == 0.)
    timer.second = MPI_Wtime();
  else {
    timer.first = timer.first + (MPI_Wtime() - timer.second);
    timer.second = 0.;
  }
}

