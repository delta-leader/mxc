#include <hidr.hpp>
#include <build_tree.hpp>

#include <algorithm>
#include <numeric>
#include <iostream>
#include <random>

void sphere_grid(double* bodies, long long nbodies, double r) {
  const double phi = M_PI * (3. - std::sqrt(5.));  // golden angle in radians
  const double d = r + r;
  const double r2 = r * r;

  for (long long i = 0; i < nbodies; ++i) {
    const double y = r - ((double)i / (double)(nbodies - 1)) * d;  // y goes from r to -r

    // Note: setting constant radius = 1 will produce a cylindrical shape
    const double radius = std::sqrt(r2 - y * y);  // radius at y
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
  if (num_samples > num_points)
    std::cout<<"Number of samples is large than the number of points ("<<num_samples<<" > "<<num_points<<")"<<std::endl;

  /*for (long long k = 0; k < num_points; ++k)
    std::cout<<points_indices[k]<<" ";
  std::cout<<std::endl;
  for (long long k = 0; k < num_points; ++k)
    std::cout<<points[k * 3]<<" ";
  std::cout<<std::endl;
  for (long long k = 0; k < num_points; ++k)
    std::cout<<points[k * 3+1]<<" ";
  std::cout<<std::endl;
  for (long long k = 0; k < num_points; ++k)
    std::cout<<points[k * 3+2]<<" ";
  std::cout<<std::endl;*/

  // not that this will yield the same idx in every call
  std::mt19937_64 rng(42);
  std::uniform_int_distribution<long long> dist(0, num_points-1);
  std::vector<long long> samples;
  samples.reserve(num_samples);
  long long idx = dist(rng);
  //std::cout<<"First point "<<idx<< " = " << points_indices[idx]<<std::endl;
  samples.emplace_back(points_indices[idx]);
  // swap the index
  std::swap(points_indices[0], points_indices[idx]);
  // swap the coordinates
  for (int dim = 0; dim < 3; ++dim)
    std::swap(points[dim], points[idx * 3 + dim]);
  
  /*for (long long k = 0; k < num_points; ++k)
    std::cout<<points[k * 3]<<" ";
  std::cout<<std::endl;
  for (long long k = 0; k < num_points; ++k)
    std::cout<<points[k * 3+1]<<" ";
  std::cout<<std::endl;
  for (long long k = 0; k < num_points; ++k)
    std::cout<<points[k * 3+2]<<" ";
  std::cout<<std::endl;*/

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
    //std::cout<<"Center: ";
    for (int dim = 0; dim < 3; ++dim) {
      C[dim] /= samples.size();
      //std::cout<<C[dim]<<", ";
    }
    //std::cout<<std::endl;

    // caluclate distances
    std::vector<double> distances(num_points - samples.size());
    for (size_t i = samples.size(); i < num_points; ++i) {
      for (int dim = 0; dim < 3; ++dim) {
        distances[i - samples.size()] += (points[i * 3 + dim] - C[dim]) * (points[i * 3 + dim] - C[dim]);
      }
    }
    // get the farthest point
    idx =  std::distance(distances.begin(), std::max_element(distances.begin(), distances.end())) + samples.size();
    //std::cout<<"Selected point "<<idx<<" = " <<points_indices[idx] <<std::endl;
    if (std::find(samples.begin(), samples.end(), points_indices[idx]) != samples.end())
      std::cout << " Something went wrong, point already exists"<<std::endl;
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
  /*std::cout<<"Samples: ";
  for (auto& val : samples)
    std::cout<<val<<", ";
  std::cout<<std::endl;
  std::cout<<"Points: "<<std::endl;
  for (size_t i = 0; i < samples.size(); ++i){
    for (int dim = 0; dim < 3; ++dim)
      std::cout<<points[i * 3 + dim]<<", ";
    std::cout<<std::endl;
  }
  std::cout<<std::endl;*/
  return samples;
}

std::vector<long long> grid_point_sampling(long long num_points, long long* const points_indices, double* points, double* selected_points, double* grid_points, long long num_samples) {
  if (num_samples > num_points)
    std::cout<<"Number of samples is large than the number of points ("<<num_samples<<" > "<<num_points<<")"<<std::endl;

  /*for (long long k = 0; k < num_points; ++k)
    std::cout<<points_indices[k]<<" ";
  std::cout<<std::endl;
  for (long long k = 0; k < num_points; ++k)
    std::cout<<points[k * 3]<<" ";
  std::cout<<std::endl;
  for (long long k = 0; k < num_points; ++k)
    std::cout<<points[k * 3+1]<<" ";
  std::cout<<std::endl;
  for (long long k = 0; k < num_points; ++k)
    std::cout<<points[k * 3+2]<<" ";
  std::cout<<std::endl;*/

  std::vector<long long> samples;
  samples.reserve(num_samples);

  for (long long j = 0; j < num_samples; ++j) {
    /*std::cout<<"Grid point "<<j<<std::endl;
    for (int dim = 0; dim < 3; ++dim) {
      std::cout<<grid_points[j * 3 + dim]<<", ";
    }
    std::cout<<std::endl;*/
    // caluclate distances to grid point [i]
    std::vector<double> distances(num_points - samples.size());
    for (size_t i = samples.size(); i < num_points; ++i) {
      for (int dim = 0; dim < 3; ++dim) {
        distances[i - samples.size()] += (points[i * 3 + dim] - grid_points[j * 3 + dim]) * (points[i * 3 + dim] - grid_points[j * 3 + dim]);
      }
    }
    /*std::cout<<"Distances "<<std::endl;
    for (auto& dist : distances)
      std::cout<<dist<<", ";
    std::cout<<std::endl;*/
    // get the closest point
    long long idx =  std::distance(distances.begin(), std::min_element(distances.begin(), distances.end())) + samples.size();;
    //std::cout<<"Selected point "<<idx<<" = " <<points_indices[idx] <<std::endl;
    if (std::find(samples.begin(), samples.end(), points_indices[idx]) != samples.end())
      std::cout << " Something went wrong, point already exists"<<std::endl;
      // swap the coordinates
    for (int dim = 0; dim < 3; ++dim)
      std::swap(points[samples.size() * 3 + dim], points[idx * 3 + dim]);
    samples.emplace_back(points_indices[idx]);
    std::swap(points_indices[samples.size() - 1], points_indices[idx]);
  }
  /*std::cout<<"Samples: ";
  for (auto& val : samples)
    std::cout<<val<<", ";
  std::cout<<std::endl;
  std::cout<<"Points: "<<std::endl;
  for (size_t i = 0; i < samples.size(); ++i){
    for (int dim = 0; dim < 3; ++dim)
      std::cout<<points[i * 3 + dim]<<", ";
    std::cout<<std::endl;
  }
  std::cout<<std::endl;*/
  return samples;
}

std::vector<long long> uniform_sampling(std::vector<long long>& points, long long num_samples) {
  if (num_samples > points.size())
    std::cout<<"Number of samples is large than the number of points ("<<num_samples<<" > "<<points.size()<<")"<<std::endl;
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

  void HiDR::initialize(long long r1, long long cell_begin, long long ncells, const Cell cells[]) {
    this->lbegin = cell_begin;
    this->lend = lbegin + ncells;
    xbodies_indices.resize(ncells);
    fbodies_indices.resize(ncells);

    // initializes the leaf level nodes to contain all the indices
    // loop over all the cells on the current level
    for (long long i = lbegin; i < lend; ++i) {
      long long idx = i - lbegin;
      //std::cout<<"Node: "<<i<<" contains pts "<< cells[i].Body[0] << "-"<<cells[i].Body[1]<<std::endl;
      std::vector<long long> tmp_indices(cells[i].Body[1] - cells[i].Body[0]);
      std::iota(tmp_indices.begin(), tmp_indices.end(), cells[i].Body[0]);
      // DATA REDUCT
      if (r1) {
        xbodies_indices[idx] = uniform_sampling(tmp_indices, r1);
      } else
        xbodies_indices[idx] = tmp_indices;
    }
  }

  void HiDR::initialize_f(long long r1, long long cell_begin, long long ncells, const Cell cells[], const std::vector<double>& pts) {
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
      //std::cout<<"Node: "<<i<<" contains pts "<< cells[i].Body[0] << "-"<<cells[i].Body[1]<<std::endl;
      //std::cout<<"Node: "<<i<<" contains "<< cells[i].Body[1] - cells[i].Body[0] << " points"<<std::endl;
      std::vector<long long> tmp_indices(cells[i].Body[1] - cells[i].Body[0]);
      std::iota(tmp_indices.begin(), tmp_indices.end(), cells[i].Body[0]);
      std::vector<double> tmp_points(&pts[cells[i].Body[0] * 3], &pts[cells[i].Body[1] * 3]);
      // DATA REDUCT
      if (r1) {
        xbodies[idx].resize(r1 * 3);
        // I had a bug here passing &pts[cells[i].Body[1]] instead of the correct value
        xbodies_indices[idx] = farthest_point_sampling(tmp_indices.size(), tmp_indices.data(), tmp_points.data(), xbodies[idx].data(), r1);
        //std::cout<<"After reduction: "<<xbodies_indices[idx].size() <<" points"<<std::endl;
      } else {
        xbodies_indices[idx] = tmp_indices;
        xbodies[idx] = tmp_points;
      }
    }
  }

  void HiDR::initialize_grid(long long r1, long long cell_begin, long long ncells, const Cell cells[], const std::vector<double>& pts, bool sphere) {
    this->lbegin = cell_begin;
    this->lend = lbegin + ncells;
    xbodies.resize(ncells);
    xbodies_indices.resize(ncells);
    fbodies.resize(ncells);
    fbodies_indices.resize(ncells);
    xgrid.resize(r1 * 3);
    if (sphere) {
      sphere_grid(xgrid.data(), r1, 1.);
    } else {
      ball_grid(xgrid.data(), r1);
    }

    // initializes the leaf level nodes to contain all the indices
    // loop over all the cells on the current level
    for (long long i = lbegin; i < lend; ++i) {
      long long idx = i - lbegin;
      //std::cout<<"Node: "<<i<<" contains pts "<< cells[i].Body[0] << "-"<<cells[i].Body[1]<<std::endl;
      //std::cout<<"Node: "<<i<<" contains "<< cells[i].Body[1] - cells[i].Body[0] << " points"<<std::endl;
      std::vector<long long> tmp_indices(cells[i].Body[1] - cells[i].Body[0]);
      std::iota(tmp_indices.begin(), tmp_indices.end(), cells[i].Body[0]);
      std::vector<double> tmp_points(&pts[cells[i].Body[0] * 3], &pts[cells[i].Body[1] * 3]);
      // DATA REDUCT
      if (r1) {
        xbodies[idx].resize(r1 * 3);
        // I had a bug here passing &pts[cells[i].Body[1]] instead of the correct value
        xbodies_indices[idx] = grid_point_sampling(tmp_indices.size(), tmp_indices.data(), tmp_points.data(), xbodies[idx].data(), xgrid.data(), r1);
        //std::cout<<"After reduction: "<<xbodies_indices[idx].size() <<" points"<<std::endl;
      } else {
        xbodies_indices[idx] = tmp_indices;
        xbodies[idx] = tmp_points;
      }
    }
  }

  void HiDR::bottom_up_sweep(long long r1, long long cell_begin, long long ncells, const Cell cells[], const HiDR& lower_level) {
    this->lbegin = cell_begin;
    this->lend = lbegin + ncells;
    xbodies_indices.resize(ncells);
    fbodies_indices.resize(ncells);

    // loop over all the cells on the current level
    for (long long i = lbegin; i < lend; ++i) {
      //std::cout<<"Node: "<< i <<" contains pts ";
      long long idx = i - lbegin;
      std::vector<long long> tmp_indices;
      // collect the sampled points from the children
      for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; ++c) {
        long long child_idx = c - lower_level.lbegin;
        tmp_indices.insert(tmp_indices.end(), lower_level.xbodies_indices[child_idx].begin(), lower_level.xbodies_indices[child_idx].end());
      }
      // DATA REDUCT
      if (r1)
      xbodies_indices[idx] = uniform_sampling(tmp_indices, r1);
      else
        xbodies_indices[idx] = tmp_indices;
      //std::cout<<xbodies_indices[idx][0]<<"-"<<xbodies_indices[idx][tmp_indices.size()-1]<<std::endl;
    }
  }

  void HiDR::bottom_up_sweep_f(long long r1, long long cell_begin, long long ncells, const Cell cells[], const HiDR& lower_level) {
    this->lbegin = cell_begin;
    this->lend = lbegin + ncells;
    xbodies_indices.resize(ncells);
    xbodies_indices.resize(ncells);
    xbodies.resize(ncells);
    fbodies_indices.resize(ncells);
    fbodies.resize(ncells);

    // loop over all the cells on the current level
    for (long long i = lbegin; i < lend; ++i) {
      //std::cout<<"Node: "<< i <<" contains ";
      long long idx = i - lbegin;
      std::vector<long long> tmp_indices;
      std::vector<double> tmp_points;
      // collect the sampled points from the children
      for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; ++c) {
        long long child_idx = c - lower_level.lbegin;
        tmp_indices.insert(tmp_indices.end(), lower_level.xbodies_indices[child_idx].begin(), lower_level.xbodies_indices[child_idx].end());
        tmp_points.insert(tmp_points.end(), lower_level.xbodies[child_idx].begin(), lower_level.xbodies[child_idx].end());
      }
      /*std::cout<<tmp_indices.size() << " points" <<std::endl;
      std::cout<<"Points: "<<std::endl;
      for (size_t i = 0; i < tmp_indices.size(); ++i){
        for (int dim = 0; dim < 3; ++dim)
          std::cout<<tmp_points[i * 3 + dim]<<", ";
        std::cout<<std::endl;
      }
      std::cout<<std::endl;*/
      // DATA REDUCT
      if (r1) {
        xbodies[idx].resize(r1 * 3);
        xbodies_indices[idx] = farthest_point_sampling(tmp_indices.size(), tmp_indices.data(), tmp_points.data(), xbodies[idx].data(), r1);
      } else {
        xbodies_indices[idx] = tmp_indices;
        xbodies[idx] = tmp_points;
      }
      //std::cout<<xbodies_indices[idx][0]<<"-"<<xbodies_indices[idx][tmp_indices.size()-1]<<std::endl;
      //std::cout<<"After reduction: "<<xbodies_indices[idx].size() <<" points"<<std::endl;
    }
  }

  void HiDR::bottom_up_sweep_grid(long long r1, long long cell_begin, long long ncells, const Cell cells[], const HiDR& lower_level) {
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
      //std::cout<<"Node: "<< i <<" contains ";
      long long idx = i - lbegin;
      std::vector<long long> tmp_indices;
      std::vector<double> tmp_points;
      // collect the sampled points from the children
      for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; ++c) {
        long long child_idx = c - lower_level.lbegin;
        tmp_indices.insert(tmp_indices.end(), lower_level.xbodies_indices[child_idx].begin(), lower_level.xbodies_indices[child_idx].end());
        tmp_points.insert(tmp_points.end(), lower_level.xbodies[child_idx].begin(), lower_level.xbodies[child_idx].end());
      }
      /*std::cout<<tmp_indices.size() << " points" <<std::endl;
      std::cout<<"Points: "<<std::endl;
      for (size_t i = 0; i < tmp_indices.size(); ++i){
        for (int dim = 0; dim < 3; ++dim)
          std::cout<<tmp_points[i * 3 + dim]<<", ";
        std::cout<<std::endl;
      }
      std::cout<<std::endl;*/
      // DATA REDUCT
      if (r1) {
        xbodies[idx].resize(r1 * 3);
        xbodies_indices[idx] = grid_point_sampling(tmp_indices.size(), tmp_indices.data(), tmp_points.data(), xbodies[idx].data(), xgrid.data(), r1);
      } else {
        xbodies_indices[idx] = tmp_indices;
        xbodies[idx] = tmp_points;
      }
      //std::cout<<xbodies_indices[idx][0]<<"-"<<xbodies_indices[idx][tmp_indices.size()-1]<<std::endl;
      //std::cout<<"After reduction: "<<xbodies_indices[idx].size() <<" points"<<std::endl;
    }
  }
  
  void HiDR::top_down_sweep(long long r2, const Cell cells[], const CSR& Far, const HiDR& upper_level) {
    //std::cout<<"Top Down"<<std::endl;
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
      //std::cout<<"Node: "<<c<<std::endl;
      long long idx = c - lbegin;
      //std::cout<<idx<<", "<<fbodies_indices.size()<<std::endl;
      std::vector<long long> tmp_indices(fbodies_indices[idx]);
      // for each cell in the far field
      // TODO is this equivalent to the interaction list?
      //std::cout<<"Far RowIndex "<<Far.RowIndex[c]<<"-"<<Far.RowIndex[c+1]<<std::endl;
      for (long long i = Far.RowIndex[c]; i < Far.RowIndex[c + 1]; ++i) {
        //std::cout<<"Far ColIndex " <<Far.ColIndex[i]<<std::endl;
        long long j = Far.ColIndex[i] - lbegin;
        //std::cout<<"Far field node "<<j<<" contains pts ";
        tmp_indices.insert(std::lower_bound(tmp_indices.begin(), tmp_indices.end(), xbodies_indices[j][0]), xbodies_indices[j].begin(), xbodies_indices[j].end());
        //std::cout<<xbodies_indices[j][0]<<"-"<<xbodies_indices[j][xbodies_indices[j].size()-1]<<std::endl;
      }
      // DATA REDUCT
      if (r2)
        fbodies_indices[idx] = uniform_sampling(tmp_indices, r2);
      else 
        fbodies_indices[idx] = tmp_indices;
      /*if (fbodies_indices[idx].size()){
      std::cout<<"Far field contains pts "<<fbodies_indices[idx][0]<<"-";
      for (size_t i = 1; i<fbodies_indices[idx].size(); ++i) {
        if (fbodies_indices[idx][i] != fbodies_indices[idx][i - 1] + 1)
          std::cout<<fbodies_indices[idx][i-1]<<", "<<fbodies_indices[idx][i]<<"-";
      }
      std::cout<<fbodies_indices[idx][fbodies_indices[idx].size() -1 ]<<std::endl;
      }*/
    }
  }
  
  void HiDR::top_down_sweep_f(long long r2, const Cell cells[], const CSR& Far, const HiDR& upper_level) {
    //std::cout<<"Top Down"<<std::endl;
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
      //std::cout<<"Node: "<<c<<std::endl;
      long long idx = c - lbegin;
      //std::cout<<"Far field contains: "<<fbodies_indices[idx].size()<<" points from parent"<<std::endl;
      //std::cout<<idx<<", "<<fbodies_indices.size()<<std::endl;
      std::vector<long long> tmp_indices(fbodies_indices[idx]);
      std::vector<double> tmp_points(fbodies[idx]);
      // for each cell in the far field
      // TODO is this equivalent to the interaction list?
      //std::cout<<"Far RowIndex "<<Far.RowIndex[c]<<"-"<<Far.RowIndex[c+1]<<std::endl;
      // to get a HSS sample, we would neet to loop over all off-diagonal cells
      // so, similar to above, but exclude idx?
      for (long long i = Far.RowIndex[c]; i < Far.RowIndex[c + 1]; ++i) {
        //std::cout<<"Far ColIndex " <<Far.ColIndex[i]<<std::endl;
        long long j = Far.ColIndex[i] - lbegin;
        //std::cout<<"Far field node "<<j<<" contains pts ";
        // indices are stored in order (TODO is this necessary or just so that the matrix is easier to create?)
        auto ordered_idx = std::lower_bound(tmp_indices.begin(), tmp_indices.end(), xbodies_indices[j][0]);
        long long ordered_pts_idx = std::distance(tmp_indices.begin(), ordered_idx) * 3;
        tmp_indices.insert(ordered_idx, xbodies_indices[j].begin(), xbodies_indices[j].end());
        tmp_points.insert(tmp_points.begin() + ordered_pts_idx, xbodies[j].begin(), xbodies[j].end());
        //std::cout<<xbodies_indices[j][0]<<"-"<<xbodies_indices[j][xbodies_indices[j].size()-1]<<std::endl;
      }
      /*std::cout<<"Far field contains: "<<tmp_indices.size()<<" points in total"<<std::endl;
      for (size_t i = 0; i < tmp_indices.size(); ++i){
        for (int dim = 0; dim < 3; ++dim)
          std::cout<<tmp_points[i * 3 + dim]<<", ";
        std::cout<<std::endl;
      }
      std::cout<<std::endl;*/
      // DATA REDUCT
      // only if far field is not empty
      if (Far.RowIndex[c] < Far.RowIndex[c + 1] && r2) {
        fbodies[idx].resize(r2 * 3);
        fbodies_indices[idx] = farthest_point_sampling(tmp_indices.size(), tmp_indices.data(), tmp_points.data(), fbodies[idx].data(), r2);
        // sorting?
        //std::sort(fbodies_indices[idx].begin(), fbodies_indices[idx].end());
      } else { 
        fbodies_indices[idx] = tmp_indices;
        fbodies[idx] = tmp_points;
      }
      //std::cout<<"After reduction: "<<fbodies_indices[idx].size() <<" points"<<std::endl;
      /*if (fbodies_indices[idx].size()){
      std::cout<<"Far field contains pts "<<fbodies_indices[idx][0]<<"-";
      for (size_t i = 1; i<fbodies_indices[idx].size(); ++i) {
        if (fbodies_indices[idx][i] != fbodies_indices[idx][i - 1] + 1)
          std::cout<<fbodies_indices[idx][i-1]<<", "<<fbodies_indices[idx][i]<<"-";
      }
      std::cout<<fbodies_indices[idx][fbodies_indices[idx].size() -1 ]<<std::endl;
      }*/
    }
  }

  void HiDR::top_down_sweep_grid(long long r2, const Cell cells[], const CSR& Far, const HiDR& upper_level, bool sphere) {
    if (upper_level.fgrid.size())
      fgrid = upper_level.fgrid;
    else {
      fgrid.resize(r2 * 3);
      if (sphere) {
        sphere_grid(fgrid.data(), r2, 1.); 
      } else {
        ball_grid(fgrid.data(), r2, 1.); 
      }
    }
      
    //std::cout<<"Top Down"<<std::endl;
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
      //std::cout<<"Node: "<<c<<std::endl;
      long long idx = c - lbegin;
      //std::cout<<"Far field contains: "<<fbodies_indices[idx].size()<<" points from parent"<<std::endl;
      //std::cout<<idx<<", "<<fbodies_indices.size()<<std::endl;
      std::vector<long long> tmp_indices(fbodies_indices[idx]);
      std::vector<double> tmp_points(fbodies[idx]);
      // for each cell in the far field
      // TODO is this equivalent to the interaction list?
      //std::cout<<"Far RowIndex "<<Far.RowIndex[c]<<"-"<<Far.RowIndex[c+1]<<std::endl;
      for (long long i = Far.RowIndex[c]; i < Far.RowIndex[c + 1]; ++i) {
        //std::cout<<"Far ColIndex " <<Far.ColIndex[i]<<std::endl;
        long long j = Far.ColIndex[i] - lbegin;
        //std::cout<<"Far field node "<<j<<" contains pts ";
        // indices are stored in order (TODO is this necessary or just so that the matrix is easier to create?)
        auto ordered_idx = std::lower_bound(tmp_indices.begin(), tmp_indices.end(), xbodies_indices[j][0]);
        long long ordered_pts_idx = std::distance(tmp_indices.begin(), ordered_idx) * 3;
        tmp_indices.insert(ordered_idx, xbodies_indices[j].begin(), xbodies_indices[j].end());
        tmp_points.insert(tmp_points.begin() + ordered_pts_idx, xbodies[j].begin(), xbodies[j].end());
        //std::cout<<xbodies_indices[j][0]<<"-"<<xbodies_indices[j][xbodies_indices[j].size()-1]<<std::endl;
      }
      /*std::cout<<"Far field contains: "<<tmp_indices.size()<<" points in total"<<std::endl;
      for (size_t i = 0; i < tmp_indices.size(); ++i){
        for (int dim = 0; dim < 3; ++dim)
          std::cout<<tmp_points[i * 3 + dim]<<", ";
        std::cout<<std::endl;
      }
      std::cout<<std::endl;*/
      // DATA REDUCT
      if (r2) {
        fbodies[idx].resize(r2 * 3);
        fbodies_indices[idx] = grid_point_sampling(tmp_indices.size(), tmp_indices.data(), tmp_points.data(), fbodies[idx].data(), fgrid.data(), r2);
        // sorting?
        //std::sort(fbodies_indices[idx].begin(), fbodies_indices[idx].end());
      } else { 
        fbodies_indices[idx] = tmp_indices;
        fbodies[idx] = tmp_points;
      }
      //std::cout<<"After reduction: "<<fbodies_indices[idx].size() <<" points"<<std::endl;
      /*if (fbodies_indices[idx].size()){
      std::cout<<"Far field contains pts "<<fbodies_indices[idx][0]<<"-";
      for (size_t i = 1; i<fbodies_indices[idx].size(); ++i) {
        if (fbodies_indices[idx][i] != fbodies_indices[idx][i - 1] + 1)
          std::cout<<fbodies_indices[idx][i-1]<<", "<<fbodies_indices[idx][i]<<"-";
      }
      std::cout<<fbodies_indices[idx][fbodies_indices[idx].size() -1 ]<<std::endl;
      }*/
    }
  }

  // returns the number of sampled bodies for the cell with index i
  long long HiDR::fbodies_size_at_i(const long long i) const {
    return fbodies_indices[i].size();
  }
  // returns a pointer to the sampled bodies for the cell with index i
  const long long* HiDR::fbodies_at_i(const long long i) const {
    return fbodies_indices[i].data();
  }
