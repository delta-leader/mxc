
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <array>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void uniform_unit_cube(double* bodies, long long nbodies, double diameter, long long dim) {
  long long side = ceil(pow(nbodies, 1. / dim));
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

void mesh_unit_cube(double* bodies, long long nbodies) {
  if (nbodies < 8) {
    std::cerr << "Error cubic mesh size (GT/EQ. 8 required): %" << nbodies << "." << std::endl;
    return;
  }

  // compute splits: solution to 6x^2 + 12x + 8 = nbodies.
  long long x_lower_bound = (long long)floor(sqrt(6 * nbodies - 12) / 6 - 1);
  long long x_splits[3] = { x_lower_bound, x_lower_bound, x_lower_bound };
  
  for (long long i = 0; i < 3; i++) {
    long long x = x_splits[0];
    long long y = x_splits[1];
    long long z = x_splits[2];
    long long mesh_points = 8 + 4 * x + 4 * y + 4 * z + 2 * x * y + 2 * x * z + 2 * y * z;
    if (mesh_points < nbodies)
      x_splits[i] = x_splits[i] + 1;
  }

  long long lens[7] = { 8, 4 * x_splits[0], 4 * x_splits[1], 4 * x_splits[2],
    2 * x_splits[0] * x_splits[1], 2 * x_splits[0] * x_splits[2], 2 * x_splits[1] * x_splits[2] };

  double segment_x = 2. / (1. + x_splits[0]);
  double segment_y = 2. / (1. + x_splits[1]);
  double segment_z = 2. / (1. + x_splits[2]);

  for (long long i = 0; i < nbodies; i++) {
    long long region = 0;
    long long ri = i;
    while (region < 6 && ri >= lens[region]) {
      ri = ri - lens[region];
      region = region + 1;
    }

    switch (region) {
    case 0: { // Vertex
      bodies[i * 3] = (double)(1 - 2 * ((ri & 4) >> 2));
      bodies[i * 3 + 1] = (double)(1 - 2 * ((ri & 2) >> 1));
      bodies[i * 3 + 2] = (double)(1 - 2 * (ri & 1));
      break;
    }
    case 1: { // edges parallel to X-axis
      bodies[i * 3] = -1 + ((ri >> 2) + 1) * segment_x;
      bodies[i * 3 + 1] = (double)(1 - 2 * ((ri & 2) >> 1));
      bodies[i * 3 + 2] = (double)(1 - 2 * (ri & 1));
      break;
    }
    case 2: { // edges parallel to Y-axis
      bodies[i * 3] = (double)(1 - 2 * ((ri & 2) >> 1));
      bodies[i * 3 + 1] = -1 + ((ri >> 2) + 1) * segment_y;
      bodies[i * 3 + 2] = (double)(1 - 2 * (ri & 1));
      break;
    }
    case 3: { // edges parallel to Z-axis
      bodies[i * 3] = (double)(1 - 2 * ((ri & 2) >> 1));
      bodies[i * 3 + 1] = (double)(1 - 2 * (ri & 1));
      bodies[i * 3 + 2] = -1 + ((ri >> 2) + 1) * segment_z;
      break;
    }
    case 4: { // surface parallel to X-Y plane
      long long x = (ri >> 1) / x_splits[1];
      long long y = (ri >> 1) - x * x_splits[1];
      bodies[i * 3] = -1 + (x + 1) * segment_x;
      bodies[i * 3 + 1] = -1 + (y + 1) * segment_y;
      bodies[i * 3 + 2] = (double)(1 - 2 * (ri & 1));
      break;
    }
    case 5: { // surface parallel to X-Z plane
      long long x = (ri >> 1) / x_splits[2];
      long long z = (ri >> 1) - x * x_splits[2];
      bodies[i * 3] = -1 + (x + 1) * segment_x;
      bodies[i * 3 + 1] = (double)(1 - 2 * (ri & 1));
      bodies[i * 3 + 2] = -1 + (z + 1) * segment_z;
      break;
    }
    case 6: { // surface parallel to Y-Z plane
      long long y = (ri >> 1) / x_splits[2];
      long long z = (ri >> 1) - y * x_splits[2];
      bodies[i * 3] = (double)(1 - 2 * (ri & 1));
      bodies[i * 3 + 1] = -1 + (y + 1) * segment_y;
      bodies[i * 3 + 2] = -1 + (z + 1) * segment_z;
      break;
    }
    default:
      break;
    }
  }
}

void mesh_unit_sphere(double* bodies, long long nbodies, double r) {
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
