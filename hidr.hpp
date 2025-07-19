
#pragma once

#include <vector>
#include <complex>

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

public:
  void initialize(long long cell_begin, long long ncells, const Cell cells[]);
  void bottom_up_sweep(long long cell_begin, long long ncells, const Cell cells[], const HiDR& lower_level);
  void top_down_sweep(const Cell cells[], const CSR& Far, const HiDR& upper_level);
  // returns the number of sampled bodies for the cell with index i
  long long fbodies_size_at_i(const long long i) const;
  // returns a pointer to the sampled bodies for the cell with index i
  const long long* fbodies_at_i(const long long i) const;
};
