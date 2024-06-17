cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project ("shimo" LANGUAGES CXX CUDA)

set(MKL_ARCH intel64)
set(MKL_LINK dynamic)
set(MKL_INTERFACE_FULL intel_lp64)
set(MKL_THREADING sequential)

find_package(MKL REQUIRED)
find_package(MPI REQUIRED)

file(GLOB sources "*.cpp")
file(GLOB headers "*.hpp")

add_executable(main.app ${sources} ${headers})
target_include_directories(main.app PRIVATE ./)
target_compile_definitions(main.app PRIVATE MKL_Complex16=std::complex<double> EIGEN_USE_MKL_ALL)
target_compile_options(main.app PRIVATE "-O3" "-m64" "-Wall" "-Wextra")
target_link_libraries(main.app MKL::MKL MPI::MPI_CXX)
