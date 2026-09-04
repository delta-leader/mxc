#include <fstream>
#include <sstream>

#include <include/elast3d.hpp>


inline void read_mesh_specs(long long& num_nodes, long long& num_elems, const std::string& fname) {
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
