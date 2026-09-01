
#pragma once

#include <vector>


class Cell;
class CSR;

class HiDR {
private:
  // first index in the cell array for the current level
  long long lbegin = 0;
  // last index in the cell array for the current level
  long long lend = 0;
  // the sampled bodies for each cell in the level
  // local index (i.e. starts from 0 in each level)
  std::vector<std::vector<double>> xbodies;
  // stores the indices of the sampled bodies
  std::vector<std::vector<long long>> xbodies_indices;
  // the sampled bodies for the far field for each cell in the level
  std::vector<std::vector<double>> fbodies;
  // stores the indices of the sampled bodies
  std::vector<std::vector<long long>> fbodies_indices;
  // used for grid sampling
  std::vector<double> xgrid;
  std::vector<double> fgrid;

public:
  // initialize HiDR with uniform sampling
  void initialize(long long s1, long long cell_begin, long long ncells, const Cell cells[]);
  // initialize HiDR with farthest point sampling
  void initialize_f(long long s1, long long cell_begin, long long ncells, const Cell cells[], const std::vector<double>& pts);
  // initialize HiDR with grid sampling, either 2D (sphere) or 3D (ball) spherical grid
  void initialize_grid(long long s1, long long cell_begin, long long ncells, const Cell cells[], const std::vector<double>& pts, bool sphere=true);
  // bottom up sweep with uniform sampling
  void bottom_up_sweep(long long s1, long long cell_begin, long long ncells, const Cell cells[], const HiDR& lower_level);
  // bottom up sweep with farthest point sampling
  void bottom_up_sweep_f(long long s1, long long cell_begin, long long ncells, const Cell cells[], const HiDR& lower_level);
  // bottom up sweep with grid sampling
  void bottom_up_sweep_grid(long long s1, long long cell_begin, long long ncells, const Cell cells[], const HiDR& lower_level);
  // top down sweep with uniform sampling
  void top_down_sweep(long long s2, const Cell cells[], const CSR& Far, const HiDR& upper_level);
  // top down sweep with farthest point sampling
  void top_down_sweep_f(long long s2, const Cell cells[], const CSR& Far, const HiDR& upper_level);
  // top down sweep with grid sampling
  void top_down_sweep_grid(long long s2, const Cell cells[], const CSR& Far, const HiDR& upper_level, bool sphere=true);
  // returns the number of sampled bodies for the cell with index i
  long long fbodies_size_at_i(const long long i) const;
  // returns a pointer to the sampled bodies for the cell with index i
  const long long* fbodies_at_i(const long long i) const;
};
