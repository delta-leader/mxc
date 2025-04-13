
#include <complex>
#include <cmath>
#include <random>
#include <array>
#include <algorithm>
#include <iostream>
#include <fstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void uniform_unit_cube(double* bodies, long long nbodies, double diameter, long long dim) {
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

void uniform_unit_cube_rnd(double* bodies, long long nbodies, double diameter, long long dim, unsigned int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution uniform_dist(0., diameter);

  std::array<double, 3>* b3 = reinterpret_cast<std::array<double, 3>*>(bodies);
  std::array<double, 3>* b3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[3 * nbodies]);
  std::for_each(b3, b3_end, [&](std::array<double, 3>& body) {
    for (int i = 0; i < 3; i++)
      body[i] = i < dim ? uniform_dist(gen) : 0.;
  });
}

void mesh_sphere(double* bodies, long long nbodies, double r) {
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

void toPolar(double* cart_coords, double* polar_coords) {
  double radius = std::sqrt(cart_coords[0] * cart_coords[0] + cart_coords[1] * cart_coords[1] + cart_coords[2] * cart_coords[2]);
  double theta = std::acos(cart_coords[2] / radius);
  double phi = std::acos(cart_coords[0] / std::sqrt(cart_coords[0] * cart_coords[0] + cart_coords[1] * cart_coords[1] )) * (cart_coords[1] < 0 ? -1 : 1);
  std::cout<<radius<<" "<<theta<<" "<<phi<<std::endl;
  polar_coords[0] = theta;
  polar_coords[1] = phi;
}

std::vector<long long> read_mesh_data(long long& n_nodes, std::vector<double>& nodes, long long& n_elems, std::vector<double>& elems, std::vector<double>& elems_polar, const char* fname) {
  std::ifstream file(fname);
  std::string line;
  std::getline(file, line);
  std::getline(file, line);
  
  line.erase(line.begin(), std::find_if(line.begin(), line.end(), std::bind1st(std::not_equal_to<char>(), ' ')));
  std::stringstream line_stream(line);
  std::getline(line_stream, line, ' ');
  n_elems = std::stoi(line);
  std::getline(file, line);
  line.erase(line.begin(), std::find_if(line.begin(), line.end(), std::bind1st(std::not_equal_to<char>(), ' ')));
  line_stream = std::stringstream(line);
  std::getline(line_stream, line, ' ');
  n_nodes = std::stoi(line);
  std::cout<<"File contains "<<n_nodes<<" nodes and " << n_elems <<" elements"<<std::endl;

  std::getline(file, line);
  nodes.resize(n_nodes * 3);
  std::vector<long long> indices(n_nodes);
  for (long long i = 0; i < n_nodes; ++i) {
    file >> nodes[i * 3] >> nodes[i * 3 + 1] >> nodes[i * 3 + 2];
    indices[i] = i;
  }

  std::getline(file, line);
  std::getline(file, line);
  elems.resize(n_elems * 3);
  elems_polar.resize(n_elems * 2);
  for (long long i = 0; i < n_elems; ++i) {
    double sum[3];
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
      //std::cout<<elems[i * 3 + j];
      //if (j != 2)
      //  std::cout<<", ";
    }
    //std::cout<<"],"<<std::endl;
    //toPolar(&elems[i*3], &elems_polar[i*2]);
  }
  std::cout<<std::endl;
  return indices;
}

void read_data(std::complex<double>* values, const char* fname, const long long n) {
  std::ifstream file(fname);
  long long a, b;
  file >> a >> b;
  /*if ((a != n) || (b != 3 * a)) {
    std::cout<<"Number of nodes in the file does not match"<<std::endl;
    return;
  }*/
  std::cout<<"File contains "<<a<<" nodes " <<b<<std::endl;
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

void read_vector(std::complex<double>* values, long long* indices, const char* fname, const long long n) {
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

void read_matrix(std::complex<double>* values, long long* indices, const char* fname, const long long n) {
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

void write_to_csv(const char* fname, int mpi_size, long long N, double theta, long long leaf_size, long long rank, double epi, const char* mode, 
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
