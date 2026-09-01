#include <fstream>
#include <sstream>

#include <include/elast3d.hpp>


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

inline void read_mesh_data(long long& num_nodes, std::vector<struct elastWave3d::nodal_point>& nodes, long long& num_elems, std::vector<struct elastWave3d::element>& elems, int mat_num, int sphere_num=0) {
  int numNodeBasis = num_nodes;
  int numElemBasis = num_elems;
  elastWave3d::input_non_global(nodes.data(), numNodeBasis, elems.data(), numElemBasis, sphere_num, mat_num);
  // If duplicate nodes/elements are found, remove them
  if ((size_t)numNodeBasis < nodes.size()){
    std::cerr << "Warning: Duplicate nodes encountered and removed." << std::endl;
    nodes.resize(numNodeBasis);
  }
  if ((size_t)numElemBasis < elems.size()){
    std::cerr << "Warning: Duplicate elements encountered and removed." << std::endl;
    elems.resize(numElemBasis);
  }
  num_nodes = numNodeBasis;
  num_elems = numElemBasis;
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
