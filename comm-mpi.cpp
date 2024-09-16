
#include <comm-mpi.hpp>
#include <data_container.hpp>

#include <algorithm>
#include <set>

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

  long long pbegin = Mapping[mpi_rank].first;
  long long pend = Mapping[mpi_rank].second;
  long long p = std::distance(&Mapping[0], std::find(&Mapping[0], &Mapping[mpi_rank], Mapping[mpi_rank]));
  long long lenp = std::distance(&Mapping[p], 
    std::find_if_not(&Mapping[p], &Mapping[mpi_size], [&](std::pair<long long, long long> i) { return i == Mapping[mpi_rank]; }));

  auto col_to_mpi_rank = [&](long long col) { return std::distance(&Mapping[0], std::find_if(&Mapping[0], &Mapping[mpi_size], 
    [=](std::pair<long long, long long> i) { return i.first <= col && col < i.second; })); };
  std::set<long long> cols;
  std::for_each(&Cols[Rows[pbegin]], &Cols[Rows[pend]], [&](long long col) { cols.insert(col_to_mpi_rank(col)); });

  std::vector<long long> NeighborRanks(cols.begin(), cols.end());
  Proc = std::distance(NeighborRanks.begin(), std::find(NeighborRanks.begin(), NeighborRanks.end(), p));
  Boxes = std::vector<std::pair<long long, long long>>(NeighborRanks.size());
  BoxOffsets = std::vector<long long>(NeighborRanks.size() + 1);
  NeighborComm = std::vector<std::pair<int, MPI_Comm>>(p == mpi_rank ? NeighborRanks.size() : 0);

  BoxOffsets[0] = 0;
  for (long long i = 0; i < (long long)NeighborRanks.size(); i++) {
    long long ibegin = Mapping[NeighborRanks[i]].first;
    long long iend = Mapping[NeighborRanks[i]].second;
    std::set<long long> icols;
    std::for_each(&Cols[Rows[ibegin]], &Cols[Rows[iend]], [&](long long col) { icols.insert(col_to_mpi_rank(col)); });

    Boxes[i] = std::make_pair(ibegin, iend - ibegin);
    BoxOffsets[i + 1] = BoxOffsets[i] + (iend - ibegin);
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
  return (0 <= iglobal && iter != Boxes.end()) ? (iglobal - (*iter).first + BoxOffsets[std::distance(Boxes.begin(), iter)]) : -1;
}

long long ColCommMPI::iGlobal(long long ilocal) const {
  std::vector<long long>::const_iterator iter = std::prev(std::find_if_not(BoxOffsets.begin() + 1, BoxOffsets.end(), [=](long long i) { return i <= ilocal; }));
  return (0 <= ilocal && ilocal < BoxOffsets.back()) ? (Boxes[std::distance(BoxOffsets.begin(), iter)].first + ilocal - *iter) : -1;
}

long long ColCommMPI::oLocal() const {
  return 0 <= Proc ? BoxOffsets[Proc] : -1;
}

long long ColCommMPI::oGlobal() const {
  return 0 <= Proc ? Boxes[Proc].first : -1;
}

long long ColCommMPI::lenLocal() const {
  return 0 <= Proc ? Boxes[Proc].second : 0;
}

long long ColCommMPI::lenNeighbors() const {
  return 0 <= Proc ? BoxOffsets.back() : 0; 
}

long long ColCommMPI::dataSizesToNeighborOffsets(long long Dims[]) const {
  std::inclusive_scan(Dims, Dims + BoxOffsets.back(), Dims);
  long long nranks = Boxes.size();
  for (long long i = 1; i <= nranks; i++)
    if (i != BoxOffsets[i])
      std::iter_swap(&Dims[i - 1], Dims + BoxOffsets[i] - 1);
  return nranks;
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

template<class T> inline void ColCommMPI::level_merge(T* data, long long len) const {
  record_mpi();
  if (MergeComm != MPI_COMM_NULL)
    MPI_Allreduce(MPI_IN_PLACE, data, len, get_mpi_datatype<T>(), MPI_SUM, MergeComm);
  if (DupComm != MPI_COMM_NULL)
    MPI_Bcast(data, len, get_mpi_datatype<T>(), 0, DupComm);
  record_mpi();
}

template<class T> inline void ColCommMPI::level_sum(T* data, long long len) const {
  record_mpi();
  if (AllReduceComm != MPI_COMM_NULL)
    MPI_Allreduce(MPI_IN_PLACE, data, len, get_mpi_datatype<T>(), MPI_SUM, AllReduceComm);
  if (DupComm != MPI_COMM_NULL)
    MPI_Bcast(data, len, get_mpi_datatype<T>(), 0, DupComm);
  record_mpi();
}

template<typename T> inline void ColCommMPI::neighbor_bcast(T* data) const {
  record_mpi();
  for (long long p = 0; p < (long long)NeighborComm.size(); p++) {
    long long llen = BoxOffsets[p + 1] - BoxOffsets[p];
    MPI_Bcast(&data[BoxOffsets[p]], llen, get_mpi_datatype<T>(), NeighborComm[p].first, NeighborComm[p].second);
  }
  if (DupComm != MPI_COMM_NULL)
    MPI_Bcast(data, BoxOffsets.back(), get_mpi_datatype<T>(), 0, DupComm);
  record_mpi();
}

template<class T> inline void ColCommMPI::neighbor_bcast(T* data, const long long noffsets[]) const {
  record_mpi();
  for (long long p = 0; p < (long long)NeighborComm.size(); p++) {
    T* start = data + (0 < p ? noffsets[p - 1] : 0), *end = data + noffsets[p];
    MPI_Bcast(start, std::distance(start, end), get_mpi_datatype<T>(), NeighborComm[p].first, NeighborComm[p].second);
  }
  if (DupComm != MPI_COMM_NULL)
    MPI_Bcast(data, noffsets[Boxes.size() - 1], get_mpi_datatype<T>(), 0, DupComm);
  record_mpi();
}

void ColCommMPI::level_merge(std::complex<double>* data, long long len) const {
  level_merge<std::complex<double>>(data, len);
}

void ColCommMPI::level_sum(std::complex<double>* data, long long len) const {
  level_sum<std::complex<double>>(data, len);
}

void ColCommMPI::neighbor_bcast(long long* data) const {
  neighbor_bcast<long long>(data);
}

void ColCommMPI::neighbor_bcast(MatrixDataContainer<double>& dc) const {
  std::vector<long long> dims(lenNeighbors());
  for (long long i = 0; i < lenNeighbors(); i++)
    dims[i] = std::distance(dc[i], dc[i + 1]);
  dataSizesToNeighborOffsets(dims.data());
  neighbor_bcast<double>(dc[0], dims.data());
}

void ColCommMPI::neighbor_bcast(MatrixDataContainer<std::complex<double>>& dc) const {
  std::vector<long long> dims(lenNeighbors());
  for (long long i = 0; i < lenNeighbors(); i++)
    dims[i] = std::distance(dc[i], dc[i + 1]);
  dataSizesToNeighborOffsets(dims.data());
  neighbor_bcast<std::complex<double>>(dc[0], dims.data());
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

