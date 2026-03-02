#pragma once

#include <iostream>
#include <complex>
#include <array>
#include <ranges>
#include <algorithm>

//--------- for maruyama code -----------------
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
      if (nel > 20)
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
//    // print this menber
//    void show() const {
//      std::cout << "ind[0] , " << ind[0]  << std::endl;
//      std::cout << "ind[1] , " << ind[1]  << std::endl;
//      std::cout << "id[0]  , " << id[0]   << std::endl;
//      std::cout << "id[1]  , " << id[1]   << std::endl;
//      std::cout << "leng   , " << leng    << std::endl;
//      std::cout << "xc[0]  , " << xc[0]   << std::endl;
//      std::cout << "xc[1]  , " << xc[1]   << std::endl;
//      std::cout << "nvec[0], " << nvec[0] << std::endl;
//      std::cout << "nvec[1], " << nvec[1] << std::endl;
//      std::cout << "svec[0], " << svec[0] << std::endl;
//      std::cout << "svec[1], " << svec[1] << std::endl;
//      std::cout << std::endl;
//    }
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
//    void cpp_interface_entry_base_elast_wave_2d(const int &equation_type, const int &num_of_unkw, const int &num_basis, const int &mat_size, double &om, std::complex<double> &alpha, std::complex<double> cmat[], std::complex<double> rhs[], std::complex<double> kairef[], struct nodal_point nodals[], struct element elems[]);
//    void mkmat_cavity_node_base_entrywise(const double &om, const std::complex<double> &alpha, const int &xnode, const int &xNumBasis, const struct nodal_point x_nodals[], const struct element x_elems[], const int &ynode, const int &yNumBasis, const struct nodal_point y_nodals[], const struct element y_elems[], std::complex<double> cmat[]);
//    void mkmat_inclusion_entrywise_closed_curve_calderon_pmchwt(const double &om, const std::complex<double> &alpha, const int &xindex, const int &xNumBasis, const struct nodal_point x_nodals[], const struct element x_elems[], const int &yindex, const int &yNumBasis, const struct nodal_point y_nodals[], const struct element y_elems[], std::complex<double> cmat[]);
//    void mkmat_inclusion_entrywise_u(const double &om, const std::complex<double> &alpha, const int &xindex, const int &xNumBasis, const struct nodal_point x_nodals[], const struct element x_elems[], const int &yindex, const int &yNumBasis, const struct nodal_point y_nodals[], const struct element y_elems[], std::complex<double> cmat[]);
//    void mkmat_inclusion_entrywise_t(const double &om, const std::complex<double> &alpha, const int &xindex, const int &xNumBasis, const struct nodal_point x_nodals[], const struct element x_elems[], const int &yindex, const int &yNumBasis, const struct nodal_point y_nodals[], const struct element y_elems[], std::complex<double> cmat[]);
//    void mkmat_inclusion_entrywise_at(const double &om, const std::complex<double> &alpha, const int &xindex, const int &xNumBasis, const struct nodal_point x_nodals[], const struct element x_elems[], const int &yindex, const int &yNumBasis, const struct nodal_point y_nodals[], const struct element y_elems[], std::complex<double> cmat[]);
//    void mkmat_inclusion_entrywise_w(const double &om, const std::complex<double> &alpha, const int &xindex, const int &xNumBasis, const struct nodal_point x_nodals[], const struct element x_elems[], const int &yindex, const int &yNumBasis, const struct nodal_point y_nodals[], const struct element y_elems[], std::complex<double> cmat[]);
//    void mkmat_entrywise_2d_elast(const int &out_in, const int &slp_or_dlp, const int &linear_or_const, const double &om, const std::complex<double> &alpha, const int &xindex, const int &xNumBasis, const struct nodal_point x_nodals[], const struct element x_elems[], const int &yindex, const int &yNumBasis, const struct nodal_point y_nodals[], const struct element y_elems[], std::complex<double> cmat[]);
//    void making_mesh_om_alpha_elast_para(const int &num_basis, double &om, std::complex<double> &alpha, struct nodal_point nodals[], struct element elems[]);
//    void making_elast_incident_wave_cavity(const int &num_basis, const double pvec[], const double &om, const std::complex<double> &alpha, const int &node_number, const struct nodal_point nodals[], const struct element elems[], const int &bm_mode, std::complex<double> rhs_mirror[]);
//    void making_elast_incident_wave_inclusion(const int &num_basis, const double pvec[], const double &om, const int &node_number, const struct nodal_point nodals[], const struct element elems[], std::complex<double> rhs_mirror[]);
//    void making_elast_incident_wave_u_const_galerkin(const int &num_basis, const double pvec[], const double &om, const int &node_number, const struct nodal_point nodals[], const struct element elems[], std::complex<double> rhs_mirror[]);
//    void single_circle_mesh(const int &nnode, const double &rad, const double &x1center, const double &x2center, struct nodal_point nodals[], struct element elems[]);
  }
}
