
cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3 -m64 -Wall -Wextra")

project ("shimo" LANGUAGES CXX)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(main main.cpp linalg.cpp basis.cpp comm.cpp build_tree.cpp umv.cpp)
target_include_directories(main PRIVATE ./)
target_link_directories(main PRIVATE)
target_link_libraries(main MPI::MPI_CXX OpenMP::OpenMP_CXX BLAS::BLAS LAPACK::LAPACK -llapacke)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "80")
