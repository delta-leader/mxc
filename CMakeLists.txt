cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project ("shimo" LANGUAGES CXX)

set(MKL_ARCH intel64)
set(MKL_LINK dynamic)
set(MKL_INTERFACE_FULL intel_lp64)
set(MKL_THREADING sequential)

find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
find_package(MPI REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
message(STATUS "Imported oneMKL targets: ${MKL_IMPORTED_TARGETS}")

file(GLOB sources "*.cpp")
file(GLOB headers "*.hpp")

add_executable(main.app ${sources} ${headers})
target_link_libraries(main.app Eigen3::Eigen $<LINK_ONLY:MKL::MKL> MPI::MPI_CXX)
target_include_directories(main.app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(main.app PRIVATE MKL_Complex16=std::complex<double> EIGEN_USE_MKL_ALL)
target_compile_options(main.app PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(main.app PRIVATE "-Wall" "-Wpedantic" "-Wextra" "-fexceptions" "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>")
endif()
