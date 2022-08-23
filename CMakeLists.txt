
cmake_minimum_required (VERSION 3.18)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

project ("shimo" LANGUAGES C CXX)

set(MKL_ARCH intel64)
set(MKL_LINK dynamic)
set(MKL_INTERFACE_FULL intel_lp64)
set(MKL_THREADING sequential)
set(MKL_MPI intelmpi)
find_package(MKL)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(lorasp lorasp.c kernel.c linalg.c basis.c build_tree.c umv.c profile.c)
target_include_directories(lorasp PRIVATE "./")
target_link_libraries(lorasp MPI::MPI_C OpenMP::OpenMP_C MKL::MKL)
target_compile_options(lorasp PRIVATE -D_PROF -D_MKL)
