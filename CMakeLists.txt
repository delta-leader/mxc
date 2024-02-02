
cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -m64 -Wall -Wextra")

project ("shimo" LANGUAGES CXX)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(main main.cpp kernel.cpp basis.cpp comm.cpp build_tree.cpp lowrank.cpp solver.cpp)
target_compile_definitions(main PRIVATE EIGEN_DONT_PARALLELIZE)
target_include_directories(main PRIVATE ./)
target_link_libraries(main MPI::MPI_CXX OpenMP::OpenMP_CXX Eigen3::Eigen BLAS::BLAS LAPACK::LAPACK -llapacke)
