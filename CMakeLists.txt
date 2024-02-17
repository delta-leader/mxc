
cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -m64 -Wall -Wextra")

project ("shimo" LANGUAGES CXX)

set(MKL_ARCH intel64)
set(MKL_LINK dynamic)
set(MKL_INTERFACE_FULL intel_ilp64)
set(MKL_THREADING sequential)

find_package(MKL REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(main main.cpp kernel.cpp basis.cpp comm.cpp build_tree.cpp solver.cpp)
target_compile_definitions(main PRIVATE MKL_Complex16=std::complex<double>)
target_include_directories(main PRIVATE ./)
target_link_libraries(main MPI::MPI_CXX OpenMP::OpenMP_CXX MKL::MKL)

