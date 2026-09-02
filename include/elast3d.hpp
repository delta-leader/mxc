#pragma once

#include <iostream>
#include <complex>
#include <array>
#include <ranges>
#include <algorithm>

// for interaction with fortran code
namespace elastWave3d{
  struct nodal_point{
  public:
    int ident;
    int nel;
    int iel[40];
    double xc[3];
    double nvec[3];
    double svec[3];
    std::complex<double> u[3];

    // constructer
    nodal_point(){
      ident = -1;
      nel = -1;
      std::fill_n(iel, 40, -1);
      std::fill_n(xc, 3, 0.0);
      std::fill_n(nvec, 3, 0.0);
      std::fill_n(svec, 3, 0.0);
      std::fill_n(u, 3, 0.0);
    }
    
    bool check() const {
      if (nel > 40)
        return false;
      return true;
    };

    // print this menber
    void show() const {
      std::cout << "ident " << ident << std::endl;
      std::cout << "nel " << nel << std::endl;
      for(int j = 0; j < nel; j++){
        for(int i = 0; i < 2; i++){
          std::cout << "i, j, iel(i, j) " << i << " " << j << " " << iel[i + j*2] << std::endl;
        }
      }
      for(auto x : xc){std::cout << "xc " << x << std::endl;}
      for(auto x : nvec){std::cout << "nvec " << x << std::endl;}
      for(auto x : svec){std::cout << "svec " << x << std::endl;}
      for(auto x : u){std::cout << "u " << x << std::endl;}
      std::cout << std::endl;
    }
  };

  struct element{
  public:
    int id_e;
    int nedge;
    int nnear;
    int ind[3];
    int id[3];
    double Jgg;
    double xc[3];
    double nvec[3];
    double mvec[3];
    double svec[3];
    std::complex<double> t[3];

    // constructer
    element(){
      id_e = -1;
      nedge = 0;
      nnear = -1;
      std::fill_n(ind, 3, -1);
      std::fill_n(id, 3, 0);
      Jgg = 0.0;
      std::fill_n(xc, 3, 0.0);
      std::fill_n(nvec, 3, 0.0);
      std::fill_n(mvec, 3, 0.0);
      std::fill_n(svec, 3, 0.0);
      std::fill_n(t, 3, 0.0);
    }
  };

  extern "C" {
    void input_non_global(struct nodal_point nodals[], int &numNodeBasis, struct element elems[], int &numElemBasis, const int& sphere_num, const int& mat_num);
    void inc_disp_const_x(const struct nodal_point nodals[], const int &numNodeBasis, const struct element &elx, const double &om, std::complex<double> uout[]);
    void inc_disp(const struct nodal_point nodals[], const int &numNodeBasis, const int &xnode, const struct element elems[], const int &numElemBasis, const double &omega, std::complex<double> uout[]);
    void inc_trac(const struct nodal_point nodals[], const int &numNodeBasis, const int &xnode, const struct element elems[], const int &numElemBasis, const double &omega, std::complex<double> uout[]);
    std::complex<double> set_alpha(const double &omega);
    double get_mu(const int &out_in, const int &index);
    double set_theta(const double &theta);
    void mkmat_entrywise_3d_elast(const struct nodal_point x_nodals[], const int &xNumNodeBasis, const struct element x_elems[], const int &xNumElemBasis, const int &xindex, const struct nodal_point y_nodals[], const int &yNumNodeBasis, const struct element y_elems[], const int &yNumElemBasis, const int &yindex, const double &omega, const int &out_in, const int &slp_or_dlp, const int &linear_or_const, const int &symmetric_integration, std::complex<double> dummat[]);
  }
}
