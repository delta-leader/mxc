cmake_minimum_required(VERSION 3.22)

enable_language(Fortran)
project(elastWave3dTransmission Fortran)

set(CMAKE_VERBOSE_MAKEFILE 1)

option(CCACHE_ENABLE
  "If the command ccache is avilable, use it for compile."
  ON)
find_program(CCACHE_EXE ccache)
if(CCACHE_EXE)
  if(CCACHE_ENABLE)
    message(STATUS "Enable ccache")
    if(CMAKE_C_COMPILER_LAUNCHER)
      set(CMAKE_C_COMPILER_LAUNCHER "${CMAKE_C_COMPILER_LAUNCHER}" "${CCACHE_EXE}")
    else()
      set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_EXE}")
    endif()
    if(CMAKE_CXX_COMPILER_LAUNCHER)
      set(CMAKE_CXX_COMPILER_LAUNCHER "${CMAKE_CXX_COMPILER_LAUNCHER}" "${CCACHE_EXE}")
    else()
      set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_EXE}")
    endif()
  endif()
endif()

add_library(elastWave3d
  module/math_cst.f90
  module/struct_type.f90
  module/module_BEM.f90
  subroutine/integration/Gauss_line.f90
  subroutine/integration/Gauss_tri.f90  
  subroutine/fundamental/check_int.f90
  subroutine/fundamental/check_norm.f90
  subroutine/fundamental/Exp_sr_series_BEM.f90
  subroutine/fundamental/in_or_not.f90
  subroutine/fundamental/shape_function.f90
  cppInterface/inputNonGlobal.f90
  cppInterface/modulesNonGlobal.f90
  cppInterface/incident_wave.f90
  cppInterface/mkmat_entrywise_3d_elast.f90
  cppInterface/Galerkin_Wij_freq_ip_iq_nonGlobal.f90
  cppInterface/elast3d_Wij_l_iq_nonGlobal.f90
  cppInterface/Dyn_Sigma_Uij_nonGlobal.f90
  cppInterface/Galerkin_aTij_freq_ip_nonGlobal.f90
  cppInterface/elast3d_aTij_l_nonGlobal.f90
  cppInterface/Galerkin_Tij_freq_ip_iq_nonGlobal.f90
  cppInterface/Galerkin_Uij_freq_ip_nonGlobal.f90
  cppInterface/elast3d_Tij_l_iq_nonGlobal.f90
  cppInterface/elast3d_Uij_l_nonGlobal.f90
  cppInterface/elast3d_UTij_static_nonGlobal.f90
  cppInterface/static_Sigma_d_Uij_nonGlobal.f90
)

# openmp
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

target_compile_options(elastWave3d PRIVATE
  -cpp -march=native -O3 -fmax-errors=3 -ffree-line-length-none -fopt-info-vec
)
target_link_libraries(elastWave3d PRIVATE BLAS::BLAS LAPACK::LAPACK)
