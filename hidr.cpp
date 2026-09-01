#include <hidr.hpp>

#include <algorithm>
#include <numeric>
#include <random>

#include <build_tree.hpp>


void sphere_grid(double* bodies, long long nbodies, double r) {
  const double phi = M_PI * (3. - std::sqrt(5.));  // golden angle in radians
  const double d = r + r;
  const double s2 = r * r;

  for (long long i = 0; i < nbodies; ++i) {
    const double y = r - ((double)i / (double)(nbodies - 1)) * d;  // y goes from r to -r

    // Note: setting constant radius = 1 will produce a cylindrical shape
    const double radius = std::sqrt(s2 - y * y);  // radius at y
    const double theta = (double)i * phi;

    const double x = radius * std::cos(theta);
    const double z = radius * std::sin(theta);
    bodies[i * 3] = x;
    bodies[i * 3 + 1] = y;
    bodies[i * 3 + 2] = z;
  }
}

void ball_grid(double* bodies, long long nbodies, unsigned int seed=999) {
  double r = 1;//std::cbrt(3 * nbodies / (4 * M_PI));
  std::mt19937 gen(seed);
  std::uniform_real_distribution uniform_dist(-r, r);

  double x, y, z, d;
  for (long long i = 0; i < nbodies; ++i) {
    bool stop = false;
    while (!stop) {
      x = uniform_dist(gen);
      y = uniform_dist(gen);
      z = uniform_dist(gen);
      d = x*x + y*y + z*z;
      if (d <= r + r)
        stop = true;
    }
    bodies[i * 3] = x;
    bodies[i * 3 + 1] = y;
    bodies[i * 3 + 2] = z;
  } 
}


std::vector<long long> farthest_point_sampling(long long num_points, long long* const points_indices, double* points, double* selected_points, long long num_samples) {
  if (num_samples > num_points) {
    // if there are not enough points, just take all of them
    num_samples = num_points;
  }

  // not that this will yield the same idx in every call
  std::mt19937_64 rng(42);
  std::uniform_int_distribution<long long> dist(0, num_points-1);
  std::vector<long long> samples;
  samples.reserve(num_samples);
  long long idx = dist(rng);
  samples.emplace_back(points_indices[idx]);
  // swap the index
  std::swap(points_indices[0], points_indices[idx]);
  // swap the coordinates
  for (int dim = 0; dim < 3; ++dim)
    std::swap(points[dim], points[idx * 3 + dim]);
  

  for (long long i = 1; i < num_samples; ++i) {
    // calculate center 
    double C[3];
    for (int dim = 0; dim < 3; ++dim)
      C[dim] = 0;
    for (size_t i = 0; i < samples.size(); ++i) {
      for (int dim = 0; dim < 3; ++dim) {
        C[dim] += points[i * 3 + dim];
      }
    }
    for (int dim = 0; dim < 3; ++dim) {
      C[dim] /= samples.size();
    }

    // caluclate distances
    std::vector<double> distances(num_points - samples.size());
    for (long long i = samples.size(); i < num_points; ++i) {
      for (int dim = 0; dim < 3; ++dim) {
        distances[i - samples.size()] += (points[i * 3 + dim] - C[dim]) * (points[i * 3 + dim] - C[dim]);
      }
    }
    // get the farthest point
    idx =  std::distance(distances.begin(), std::max_element(distances.begin(), distances.end())) + samples.size();
    if (std::find(samples.begin(), samples.end(), points_indices[idx]) != samples.end())
      std::cerr << " Something went wrong during sampling, point already exists"<<std::endl;
    // swap the coordinates
    for (int dim = 0; dim < 3; ++dim)
      std::swap(points[samples.size() * 3 + dim], points[idx * 3 + dim]);
    samples.emplace_back(points_indices[idx]);
    std::swap(points_indices[samples.size() - 1], points_indices[idx]);
  }
  for (long long i = 0; i < num_samples; ++i) {
    for (int dim = 0; dim < 3; ++dim)
      selected_points[i * 3 + dim] = points[i * 3 + dim];
  }
  return samples;
}

std::vector<long long> grid_point_sampling(long long num_points, long long* const points_indices, double* points, double* grid_points, long long num_samples) {
  if (num_samples > num_points) {
    // if there are not enough points, just take all of them
    num_samples = num_points;
  }

  std::vector<long long> samples;
  samples.reserve(num_samples);

  for (long long j = 0; j < num_samples; ++j) {
    // caluclate distances to grid point [i]
    std::vector<double> distances(num_points - samples.size());
    for (long long i = samples.size(); i < num_points; ++i) {
      for (int dim = 0; dim < 3; ++dim) {
        distances[i - samples.size()] += (points[i * 3 + dim] - grid_points[j * 3 + dim]) * (points[i * 3 + dim] - grid_points[j * 3 + dim]);
      }
    }
    // get the closest point
    long long idx =  std::distance(distances.begin(), std::min_element(distances.begin(), distances.end())) + samples.size();;
    if (std::find(samples.begin(), samples.end(), points_indices[idx]) != samples.end())
      std::cout << " Something went wrong, point already exists"<<std::endl;
      // swap the coordinates
    for (int dim = 0; dim < 3; ++dim)
      std::swap(points[samples.size() * 3 + dim], points[idx * 3 + dim]);
    samples.emplace_back(points_indices[idx]);
    std::swap(points_indices[samples.size() - 1], points_indices[idx]);
  }
  return samples;
}

std::vector<long long> uniform_sampling(std::vector<long long>& points, long long num_samples) {
  if ((size_t) num_samples > points.size()) {
    // if there are not enough points, just take all of them
    num_samples = points.size();
  }
  std::mt19937_64 rng(42);
  std::uniform_int_distribution<long long> dist(0, points.size()-1);
  std::vector<long long> samples(num_samples);
  for (long long i = 0; i< num_samples; ++i) {
    long long pt = dist(rng);
    while (std::find(samples.begin(), samples.end(), pt) != samples.end())
      pt = dist(rng);
    samples.emplace_back(pt);
  }
  return samples;
}

// initialize with uniform sampling
void HiDR::initialize(long long s1, long long cell_begin, long long ncells, const Cell cells[]) {
  this->lbegin = cell_begin;
  this->lend = lbegin + ncells;
  xbodies_indices.resize(ncells);
  fbodies_indices.resize(ncells);

  // initializes the leaf level nodes to contain all the indices
  // loop over all the cells on the current level
  for (long long i = lbegin; i < lend; ++i) {
    long long idx = i - lbegin;
    std::vector<long long> tmp_indices(cells[i].Body[1] - cells[i].Body[0]);
    std::iota(tmp_indices.begin(), tmp_indices.end(), cells[i].Body[0]);
    // DATA REDUCT
    if (s1) {
      xbodies_indices[idx] = uniform_sampling(tmp_indices, s1);
    } else
      xbodies_indices[idx] = tmp_indices;
  }
}

// initialize with farthest point sampling
void HiDR::initialize_f(long long s1, long long cell_begin, long long ncells, const Cell cells[], const std::vector<double>& pts) {
  this->lbegin = cell_begin;
  this->lend = lbegin + ncells;
  xbodies.resize(ncells);
  xbodies_indices.resize(ncells);
  fbodies.resize(ncells);
  fbodies_indices.resize(ncells);

  // initializes the leaf level nodes to contain all the indices
  // loop over all the cells on the current level
  for (long long i = lbegin; i < lend; ++i) {
    long long idx = i - lbegin;
    std::vector<long long> tmp_indices(cells[i].Body[1] - cells[i].Body[0]);
    std::iota(tmp_indices.begin(), tmp_indices.end(), cells[i].Body[0]);
    std::vector<double> tmp_points(&pts[cells[i].Body[0] * 3], &pts[cells[i].Body[1] * 3]);
    // DATA REDUCT
    if (s1) {
      xbodies[idx].resize(s1 * 3);
      xbodies_indices[idx] = farthest_point_sampling(tmp_indices.size(), tmp_indices.data(), tmp_points.data(), xbodies[idx].data(), s1);
    } else {
      xbodies_indices[idx] = tmp_indices;
      xbodies[idx] = tmp_points;
    }
  }
}

// initialize with grid sampling
void HiDR::initialize_grid(long long s1, long long cell_begin, long long ncells, const Cell cells[], const std::vector<double>& pts, bool sphere) {
  this->lbegin = cell_begin;
  this->lend = lbegin + ncells;
  xbodies.resize(ncells);
  xbodies_indices.resize(ncells);
  fbodies.resize(ncells);
  fbodies_indices.resize(ncells);
  xgrid.resize(s1 * 3);
  if (sphere) {
    sphere_grid(xgrid.data(), s1, 1.);
  } else {
    ball_grid(xgrid.data(), s1);
  }

  // initializes the leaf level nodes to contain all the indices
  // loop over all the cells on the current level
  for (long long i = lbegin; i < lend; ++i) {
    long long idx = i - lbegin;
    std::vector<long long> tmp_indices(cells[i].Body[1] - cells[i].Body[0]);
    std::iota(tmp_indices.begin(), tmp_indices.end(), cells[i].Body[0]);
    std::vector<double> tmp_points(&pts[cells[i].Body[0] * 3], &pts[cells[i].Body[1] * 3]);
    // DATA REDUCT
    if (s1) {
      xbodies[idx].resize(s1 * 3);
      xbodies_indices[idx] = grid_point_sampling(tmp_indices.size(), tmp_indices.data(), tmp_points.data(), xgrid.data(), s1);
    } else {
      xbodies_indices[idx] = tmp_indices;
      xbodies[idx] = tmp_points;
    }
  }
}

// bottum up sweep with uniform sampling
void HiDR::bottom_up_sweep(long long s1, long long cell_begin, long long ncells, const Cell cells[], const HiDR& lower_level) {
  this->lbegin = cell_begin;
  this->lend = lbegin + ncells;
  xbodies_indices.resize(ncells);
  fbodies_indices.resize(ncells);

  // loop over all the cells on the current level
  for (long long i = lbegin; i < lend; ++i) {
    long long idx = i - lbegin;
    std::vector<long long> tmp_indices;
    // collect the sampled points from the children
    for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; ++c) {
      long long child_idx = c - lower_level.lbegin;
      tmp_indices.insert(tmp_indices.end(), lower_level.xbodies_indices[child_idx].begin(), lower_level.xbodies_indices[child_idx].end());
    }
    // DATA REDUCT
    if (s1)
      xbodies_indices[idx] = uniform_sampling(tmp_indices, s1);
    else
      xbodies_indices[idx] = tmp_indices;
  }
}

// bottom upu sweep with farthest point sampling
void HiDR::bottom_up_sweep_f(long long s1, long long cell_begin, long long ncells, const Cell cells[], const HiDR& lower_level) {
  this->lbegin = cell_begin;
  this->lend = lbegin + ncells;
  xbodies_indices.resize(ncells);
  xbodies_indices.resize(ncells);
  xbodies.resize(ncells);
  fbodies_indices.resize(ncells);
  fbodies.resize(ncells);

  // loop over all the cells on the current level
  for (long long i = lbegin; i < lend; ++i) {
    long long idx = i - lbegin;
    std::vector<long long> tmp_indices;
    std::vector<double> tmp_points;
    // collect the sampled points from the children
    for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; ++c) {
      long long child_idx = c - lower_level.lbegin;
      tmp_indices.insert(tmp_indices.end(), lower_level.xbodies_indices[child_idx].begin(), lower_level.xbodies_indices[child_idx].end());
      tmp_points.insert(tmp_points.end(), lower_level.xbodies[child_idx].begin(), lower_level.xbodies[child_idx].end());
    }
    // DATA REDUCT
    if (s1) {
      xbodies[idx].resize(s1 * 3);
      xbodies_indices[idx] = farthest_point_sampling(tmp_indices.size(), tmp_indices.data(), tmp_points.data(), xbodies[idx].data(), s1);
    } else {
      xbodies_indices[idx] = tmp_indices;
      xbodies[idx] = tmp_points;
    }
  }
}

// bottom up sweep with grid sampling
void HiDR::bottom_up_sweep_grid(long long s1, long long cell_begin, long long ncells, const Cell cells[], const HiDR& lower_level) {
  this->lbegin = cell_begin;
  this->lend = lbegin + ncells;
  xbodies_indices.resize(ncells);
  xbodies_indices.resize(ncells);
  xbodies.resize(ncells);
  fbodies_indices.resize(ncells);
  fbodies.resize(ncells);
  xgrid = lower_level.xgrid;

  // loop over all the cells on the current level
  for (long long i = lbegin; i < lend; ++i) {
    long long idx = i - lbegin;
    std::vector<long long> tmp_indices;
    std::vector<double> tmp_points;
    // collect the sampled points from the children
    for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; ++c) {
      long long child_idx = c - lower_level.lbegin;
      tmp_indices.insert(tmp_indices.end(), lower_level.xbodies_indices[child_idx].begin(), lower_level.xbodies_indices[child_idx].end());
      tmp_points.insert(tmp_points.end(), lower_level.xbodies[child_idx].begin(), lower_level.xbodies[child_idx].end());
    }
    // DATA REDUCT
    if (s1) {
      xbodies[idx].resize(s1 * 3);
      xbodies_indices[idx] = grid_point_sampling(tmp_indices.size(), tmp_indices.data(), tmp_points.data(), xgrid.data(), s1);
    } else {
      xbodies_indices[idx] = tmp_indices;
      xbodies[idx] = tmp_points;
    }
  }
}

// top down sweep with uniform sampling
void HiDR::top_down_sweep(long long s2, const Cell cells[], const CSR& Far, const HiDR& upper_level) {
  // loop over all the cells on the upper level
  for (long long i = upper_level.lbegin; i < upper_level.lend; ++i) {
    // collect the far field points from the upper level
    // and put them in the children
    for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; ++c) {
      // check that the child is actually on this node
      if (lbegin <= c && c < lend) {
        fbodies_indices[c - lbegin] = std::vector<long long>(upper_level.fbodies_indices[i - upper_level.lbegin]);
      }
    }
  }
  // loop over all the cells on the current level
  for (long long c = lbegin; c < lend; ++c) {
    long long idx = c - lbegin;
    std::vector<long long> tmp_indices(fbodies_indices[idx]);
    // for each cell in the far field
    for (long long i = Far.RowIndex[c]; i < Far.RowIndex[c + 1]; ++i) {
      long long j = Far.ColIndex[i] - lbegin;
      tmp_indices.insert(std::lower_bound(tmp_indices.begin(), tmp_indices.end(), xbodies_indices[j][0]), xbodies_indices[j].begin(), xbodies_indices[j].end());
    }
    // DATA REDUCT
    if (s2)
      fbodies_indices[idx] = uniform_sampling(tmp_indices, s2);
    else 
      fbodies_indices[idx] = tmp_indices;
  }
}

// top down sweep with farthest point sampling
void HiDR::top_down_sweep_f(long long s2, const Cell cells[], const CSR& Far, const HiDR& upper_level) {
  // loop over all the cells on the upper level
  for (long long i = upper_level.lbegin; i < upper_level.lend; ++i) {
    // collect the far field points from the upper level
    // and put them in the children
    for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; ++c) {
      // check that the child is actually on this node
      if (lbegin <= c && c < lend) {
        fbodies_indices[c - lbegin] = std::vector<long long>(upper_level.fbodies_indices[i - upper_level.lbegin]);
        fbodies[c - lbegin] = std::vector<double>(upper_level.fbodies[i - upper_level.lbegin]);
      }
    }
  }
  // loop over all the cells on the current level
  for (long long c = lbegin; c < lend; ++c) {
    long long idx = c - lbegin;
    std::vector<long long> tmp_indices(fbodies_indices[idx]);
    std::vector<double> tmp_points(fbodies[idx]);
    // for each cell in the far field
    for (long long i = Far.RowIndex[c]; i < Far.RowIndex[c + 1]; ++i) {
      long long j = Far.ColIndex[i] - lbegin;
      // indices are stored in order
      auto ordered_idx = std::lower_bound(tmp_indices.begin(), tmp_indices.end(), xbodies_indices[j][0]);
      long long ordered_pts_idx = std::distance(tmp_indices.begin(), ordered_idx) * 3;
      tmp_indices.insert(ordered_idx, xbodies_indices[j].begin(), xbodies_indices[j].end());
      tmp_points.insert(tmp_points.begin() + ordered_pts_idx, xbodies[j].begin(), xbodies[j].end());
    }
    // DATA REDUCT
    // only if far field is not empty
    if (Far.RowIndex[c] < Far.RowIndex[c + 1] && s2) {
      fbodies[idx].resize(s2 * 3);
      fbodies_indices[idx] = farthest_point_sampling(tmp_indices.size(), tmp_indices.data(), tmp_points.data(), fbodies[idx].data(), s2);
    } else { 
      fbodies_indices[idx] = tmp_indices;
      fbodies[idx] = tmp_points;
    }
  }
}

// top down sweep with grid sampling
void HiDR::top_down_sweep_grid(long long s2, const Cell cells[], const CSR& Far, const HiDR& upper_level, bool sphere) {
  if (upper_level.fgrid.size())
    fgrid = upper_level.fgrid;
  else {
    fgrid.resize(s2 * 3);
    if (sphere) {
      sphere_grid(fgrid.data(), s2, 1.); 
    } else {
      ball_grid(fgrid.data(), s2, 1.); 
    }
  }
      
  // loop over all the cells on the upper level
  for (long long i = upper_level.lbegin; i < upper_level.lend; ++i) {
    // collect the far field points from the upper level
    // and put them in the children
    for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; ++c) {
      // check that the child is actually on this node
      if (lbegin <= c && c < lend) {
        fbodies_indices[c - lbegin] = std::vector<long long>(upper_level.fbodies_indices[i - upper_level.lbegin]);
        fbodies[c - lbegin] = std::vector<double>(upper_level.fbodies[i - upper_level.lbegin]);
      }
    }
  }
  // loop over all the cells on the current level
  for (long long c = lbegin; c < lend; ++c) {
    long long idx = c - lbegin;

    std::vector<long long> tmp_indices(fbodies_indices[idx]);
    std::vector<double> tmp_points(fbodies[idx]);
    // for each cell in the far field
    for (long long i = Far.RowIndex[c]; i < Far.RowIndex[c + 1]; ++i) {
      long long j = Far.ColIndex[i] - lbegin;
      // indices are stored in order (TODO is this necessary or just so that the matrix is easier to create?)
      auto ordered_idx = std::lower_bound(tmp_indices.begin(), tmp_indices.end(), xbodies_indices[j][0]);
      long long ordered_pts_idx = std::distance(tmp_indices.begin(), ordered_idx) * 3;
      tmp_indices.insert(ordered_idx, xbodies_indices[j].begin(), xbodies_indices[j].end());
      tmp_points.insert(tmp_points.begin() + ordered_pts_idx, xbodies[j].begin(), xbodies[j].end());
    }
    // DATA REDUCT
    if (s2) {
      fbodies[idx].resize(s2 * 3);
      fbodies_indices[idx] = grid_point_sampling(tmp_indices.size(), tmp_indices.data(), tmp_points.data(), fgrid.data(), s2);
    } else { 
      fbodies_indices[idx] = tmp_indices;
      fbodies[idx] = tmp_points;
    }
  }
}

 // returns the number of sampled bodies for the cell with index i
long long HiDR::fbodies_size_at_i(const long long i) const {
  // return zero if empty
  return fbodies_indices.size() > (size_t) i ? fbodies_indices[i].size() : 0;
}
// returns a pointer to the sampled bodies for the cell with index i
const long long* HiDR::fbodies_at_i(const long long i) const {
  return fbodies_indices[i].data();
}
