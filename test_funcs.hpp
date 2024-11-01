
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

void mesh_sphere(double* bodies, long long nbodies) {
  double r = std::sqrt(nbodies / (4 * M_PI));
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

void mesh_ball(double* bodies, long long nbodies, unsigned int seed) {
  double r = std::cbrt(3 * nbodies / (4 * M_PI));
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

void read_sorted_bodies(long long* nbodies, long long lbuckets, double* bodies, long long buckets[], const char* fname) {
  std::ifstream file(fname);

  long long curr = 1, cbegin = 0, iter = 0, len = *nbodies;
  while (iter < len && !file.eof()) {
    long long b = 0;
    double x = 0., y = 0., z = 0.;
    file >> x >> y >> z >> b;

    if (lbuckets < b)
      len = iter;
    else if (!file.eof()) {
      bodies[iter * 3] = x;
      bodies[iter * 3 + 1] = y;
      bodies[iter * 3 + 2] = z;
      while (curr < b && curr <= lbuckets) {
        buckets[curr - 1] = iter - cbegin;
        cbegin = iter;
        curr++;
      }
      iter++;
    }
  }
  while (curr <= lbuckets) {
    buckets[curr - 1] = iter - cbegin;
    cbegin = iter;
    curr++;
  }
  *nbodies = iter;
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
