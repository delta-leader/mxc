
#include <complex>
#include <cmath>
#include <random>
#include <array>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

#include <include/elast3d.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline void uniform_unit_cube(double* bodies, long long nbodies, double diameter, long long dim) {
  long long side = std::ceil(std::pow(nbodies, 1. / dim));
  long long lens[3] = { dim > 0 ? side : 1, dim > 1 ? side : 1, dim > 2 ? side : 1 };
  double step = diameter / side;

  for (long long i = 0; i < lens[0]; ++i)
    for (long long j = 0; j < lens[1]; ++j)
       for (long long k = 0; k < lens[2]; ++k) {
    long long x = k + lens[2] * (j + lens[1] * i);
    if (x < nbodies) {
      bodies[x * 3] = i * step;
      bodies[x * 3 + 1] = j * step;
      bodies[x * 3 + 2] = k * step;
    }
  }
}

inline void uniform_unit_cube_rnd(double* bodies, long long nbodies, double diameter, long long dim, unsigned int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution uniform_dist(0., diameter);

  std::array<double, 3>* b3 = reinterpret_cast<std::array<double, 3>*>(bodies);
  std::array<double, 3>* b3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[3 * nbodies]);
  std::for_each(b3, b3_end, [&](std::array<double, 3>& body) {
    for (int i = 0; i < 3; i++)
      body[i] = i < dim ? uniform_dist(gen) : 0.;
  });
}

inline void mesh_sphere(double* bodies, long long nbodies, double r) {
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

inline void toPolar(double* cart_coords, double* polar_coords) {
  double radius = std::sqrt(cart_coords[0] * cart_coords[0] + cart_coords[1] * cart_coords[1] + cart_coords[2] * cart_coords[2]);
  double theta = std::acos(cart_coords[2] / radius);
  double phi = std::acos(cart_coords[0] / std::sqrt(cart_coords[0] * cart_coords[0] + cart_coords[1] * cart_coords[1] )) * (cart_coords[1] < 0 ? -1 : 1);
  std::cout<<radius<<" "<<theta<<" "<<phi<<std::endl;
  polar_coords[0] = theta;
  polar_coords[1] = phi;
}

inline void read_mesh_specs(long long& num_nodes, long long& num_elems, const std::string& fname) {
  //std::cout<<fname<<std::endl;
  std::ifstream file(fname);
  std::string line;
  std::getline(file, line);
  std::getline(file, line);

  std::istringstream iss(line);
  iss >> num_elems; 
  std::getline(file, line);
  iss = std::istringstream(line);
  iss >> num_nodes;
}

inline void read_mesh_fortran(long long& num_nodes, std::vector<struct elastWave3d::nodal_point>& nodes, long long& num_elems, std::vector<struct elastWave3d::element>& elems, int mat_num, int sphere_num=0) {
  int numNodeBasis = num_nodes;
  int numElemBasis = num_elems;
  elastWave3d::input_non_global(nodes.data(), numNodeBasis, elems.data(), numElemBasis, sphere_num, mat_num);
  // If there are duplicate nodes or element, shrink the vecors
  if (numNodeBasis < nodes.size()){
    std::cout << "shrink nodes" << std::endl;
    nodes.resize(numNodeBasis);
  }
  if (numElemBasis < elems.size()){
    std::cout << "shrink elems" << std::endl;
    elems.resize(numElemBasis);
  }
  num_nodes = numNodeBasis;
  num_elems = numElemBasis;
} 

inline void read_mesh_data(long long& num_nodes, std::vector<double>& nodes, long long& num_elems, std::vector<double>& elems, const std::string& fname) {
  // Open the file and skip the first two lines
  std::ifstream file(fname);
  std::string line;
  std::getline(file, line);
  std::getline(file, line);
  
  // read the number of nodes & the number of elements
  line.erase(line.begin(), std::find_if(line.begin(), line.end(), std::bind1st(std::not_equal_to<char>(), ' ')));
  std::stringstream line_stream(line);
  std::getline(line_stream, line, ' ');
  num_elems = std::stoi(line);
  std::getline(file, line);
  line.erase(line.begin(), std::find_if(line.begin(), line.end(), std::bind1st(std::not_equal_to<char>(), ' ')));
  line_stream = std::stringstream(line);
  std::getline(line_stream, line, ' ');
  num_nodes = std::stoi(line);
  std::cout<<"File contains "<<num_nodes<<" nodes and " << num_elems <<" elements"<<std::endl;

  // Read the coordinates of the nodes
  std::getline(file, line);
  nodes.resize(num_nodes * 3);
  for (long long i = 0; i < num_nodes; ++i) {
    file >> nodes[i * 3] >> nodes[i * 3 + 1] >> nodes[i * 3 + 2];
  }

  // read the elements (defined by their nodes)
  std::getline(file, line);
  std::getline(file, line);
  elems.resize(num_elems * 3);
  for (long long i = 0; i < num_elems; ++i) {
    double sum[3];
    // calculate the centroid for each element
    for (int j = 0; j < 3; ++j) {
      sum[j] = 0;
    }
    for (int j = 0; j < 3; ++j) {
      long long idx;
      file >> idx;
      idx--;
      for (int k = 0; k < 3; ++k) {
        sum[k] += nodes[idx * 3 + k];
      }
    }
    //std::cout<<"[";
    for (int j = 0; j < 3; ++j) {
      elems[i * 3 + j] = sum[j] / 3;
    }
  }
}

// I don't think this is fully implemented, why would we only convert the elements to polar coordinates?
inline void read_mesh_data_polar(long long& num_nodes, std::vector<double>& nodes, long long& num_elems, std::vector<double>& elems, std::vector<double>& elems_polar, const std::string& fname) {
  // Open the file and skip the first two lines
  std::ifstream file(fname);
  std::string line;
  std::getline(file, line);
  std::getline(file, line);
  
  // read the number of nodes & the number of elements
  line.erase(line.begin(), std::find_if(line.begin(), line.end(), std::bind1st(std::not_equal_to<char>(), ' ')));
  std::stringstream line_stream(line);
  std::getline(line_stream, line, ' ');
  num_elems = std::stoi(line);
  std::getline(file, line);
  line.erase(line.begin(), std::find_if(line.begin(), line.end(), std::bind1st(std::not_equal_to<char>(), ' ')));
  line_stream = std::stringstream(line);
  std::getline(line_stream, line, ' ');
  num_nodes = std::stoi(line);
  std::cout<<"File contains "<<num_nodes<<" nodes and " << num_elems <<" elements"<<std::endl;

  // Read the coordinates of the nodes
  std::getline(file, line);
  nodes.resize(num_nodes * 3);
  for (long long i = 0; i < num_nodes; ++i) {
    file >> nodes[i * 3] >> nodes[i * 3 + 1] >> nodes[i * 3 + 2];
  }

  // read the elements (defined by their nodes)
  std::getline(file, line);
  std::getline(file, line);
  elems.resize(num_elems * 3);
  elems_polar.resize(num_elems * 2);
  for (long long i = 0; i < num_elems; ++i) {
    double sum[3];
    // calculate the centroid for each element
    for (int j = 0; j < 3; ++j) {
      sum[j] = 0;
    }
    for (int j = 0; j < 3; ++j) {
      long long idx;
      file >> idx;
      idx--;
      for (int k = 0; k < 3; ++k) {
        sum[k] += nodes[idx * 3 + k];
      }
    }
    //std::cout<<"[";
    for (int j = 0; j < 3; ++j) {
      elems[i * 3 + j] = sum[j] / 3;
    }
    //std::cout<<"],"<<std::endl;
    //toPolar(&elems[i*3], &elems_polar[i*2]);
  }
  //std::cout<<std::endl;
}

inline void read_data(std::complex<double>* values, const std::string& fname, const long long n) {
  std::ifstream file(fname);
  long long a, b;
  file >> a >> b;
  /*if ((a != n) || (b != 3 * a)) {
    std::cout<<"Number of nodes in the file does not match"<<std::endl;
    return;
  }*/
  std::cout<<"File contains "<<a<<" data points * 3 = " <<b<<std::endl;
  std::string line;
  double real, img;
  for (long long i = 0; i < n; ++i) {
    std::getline(file, line);
    std::getline(file, line, '(');
    std::getline(file, line, ',');
    real = std::stod(line);
    std::getline(file, line, ')');
    img = std::stod(line);
    values[i] = std::complex<double>(real, img);
  }
}

inline void read_data2(std::complex<double>* values, const std::string& fname, const long long n) {
  std::ifstream file(fname);
  std::string line;
  double real, img;
  for (long long i = 0; i < n; ++i) {
    std::getline(file, line, '(');
    std::getline(file, line, ',');
    real = std::stod(line);
    std::getline(file, line, ')');
    img = std::stod(line);
    values[i] = std::complex<double>(real, img);
  }
}

inline void read_vector(std::complex<double>* values, long long* indices, const std::string& fname, const long long n) {
  std::ifstream file(fname);
  long long a, b;
  file >> a >> b;
  if ((a != n) || (b != 3 * a)) {
    std::cout<<"Number of nodes in the file does not match"<<std::endl;
    return;
  }

  std::string line;
  double real, img;
  for (long long i = 0; i < n; ++i) {
    for (long long ii = 0; ii < 3; ++ii) {
      std::getline(file, line);
      std::getline(file, line, '(');
      std::getline(file, line, ',');
      real = std::stod(line);
      std::getline(file, line, ')');
      img = std::stod(line);
      values[3 * indices[i] + ii] = std::complex<double>(real, img);
    }
  }
}

inline void read_matrix(std::complex<double>* values, long long* indices, const std::string& fname, const long long n) {
  std::ifstream file(fname);
  long long a, b;
  file >> a >> b;
  if ((a != n) || (b != 3 * a)) {
    std::cout<<"Number of nodes in the file does not match"<<std::endl;
    return;
  }

  std::string line;
  double real, img;
  for (long long i = 0; i < n; ++i) {
    for (long long ii = 0; ii < 3; ++ii) {
      for (long long j = 0; j < n; ++j) {
        for (long long jj = 0; jj < 3; ++jj) {
          std::getline(file, line);
          std::getline(file, line, '(');
          std::getline(file, line, ',');
          real = std::stod(line);
          std::getline(file, line, ')');
          img = std::stod(line);
          values[(3 * indices[i] + ii) * 3 * n + 3 * indices[j] + jj] = std::complex<double>(real, img);
        }
      }
    }
  }
}

inline void write_to_csv(const char* fname, int mpi_size, long long N, double theta, long long leaf_size, long long rank, double epi, const char* mode, 
  double h2cerr, double h2ctime, double h2ctime_comm, double h2mvtime, double h2mvtime_comm, double dense_mvtime,
  double mctime, double mctime_comm, double mcerr, double factor_time, double factor_time_comm, double sub_time, double sub_time_comm, double sub_err,
  double gmres_err, double gmres_iters, double gmres_time, double gmres_time_comm, const double* iter_err) {
  
  std::ofstream file(fname, std::ios_base::app);
  if (!file.bad())
  {
    file << mpi_size << ',' << N << ',' << theta << ',' << leaf_size << ',' << rank << ',' << epi << ',' << mode << ','; // 0 - 6
    file << h2cerr << ',' << h2ctime << ',' << h2ctime_comm << ',' << h2mvtime << ',' << h2mvtime_comm << ',' << dense_mvtime << ','; // 7 - 12
    file << mctime << ',' << mctime_comm << ',' << mcerr << ',' << factor_time << ',' << factor_time_comm << ',' << sub_time << ',' << sub_time_comm << ',' << sub_err << ','; // 13 - 20
    file << gmres_err << ',' << gmres_iters << ',' << gmres_time << ',' << gmres_time_comm; // 21 - 24
    for (long long i = 0; i <= gmres_iters; i++)
      file << ',' << iter_err[i];
    file << std::endl;
    file.close();
  }
}
