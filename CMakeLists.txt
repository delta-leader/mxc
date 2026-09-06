cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_Fortran_COMPILER gfortran)

project ("shimo" LANGUAGES CXX Fortran)

option(USE_MKL "Compile with Intel MKL" OFF)

if(USE_MKL)
  find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
  message(STATUS "Imported oneMKL targets: ${MKL_IMPORTED_TARGETS}")
else()
  set(BLA_VENDOR OpenBLAS)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
endif()

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

file(GLOB sources "*.cpp")
file(GLOB headers "*.hpp")

add_subdirectory(
  BEM3D_transmission_each_component
)

add_executable(main.app ${sources} ${headers})

if(USE_MKL)
  target_compile_definitions(main.app PRIVATE MKL_COMPLEX16=std::complex<double> MKL_COMPLEX8=std::complex<float> EIGEN_USE_MKL_ALL EIGEN_MKL_NO_DIRECT_CALL)
  target_compile_options(main.app PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
  target_include_directories(main.app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
  target_link_libraries(main.app $<LINK_ONLY:MKL::MKL> OpenMP::OpenMP_CXX MPI::MPI_CXX Eigen3::Eigen elastWave3d)
else()
  target_compile_definitions(main.app PRIVATE EIGEN_USE_BLAS EIGEN_USE_LAPACKE)
  target_include_directories(main.app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(main.app BLAS::BLAS LAPACK::LAPACK lapacke OpenMP::OpenMP_CXX MPI::MPI_CXX Eigen3::Eigen elastWave3d)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(main.app PRIVATE "-Wall" "-Wextra" "-fexceptions" "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>" "-fPIE")
endif()
