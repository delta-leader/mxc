#include <hidr.hpp>
#include <build_tree.hpp>

#include <algorithm>
#include <numeric>
#include <iostream>


void HiDR::initialize(long long cell_begin, long long ncells, const Cell cells[]) {
    this->lbegin = cell_begin;
    this->lend = lbegin + ncells;
    xbodies_indices.resize(ncells);
    fbodies_indices.resize(ncells);

    // initializes the leaf level nodes to contain all the indices
    // loop over all the cells on the current level
    for (long long i = lbegin; i < lend; ++i) {
      long long idx = i - lbegin;
      //std::cout<<"Node: "<<i<<" contains pts "<< cells[i].Body[0] << "-"<<cells[i].Body[1]<<std::endl;
      std::vector<long long> tmp_indices(cells[i].Body[1] - cells[i].Body[0]);
      std::iota(tmp_indices.begin(), tmp_indices.end(), cells[i].Body[0]);
      // DATA REDUCT
      xbodies_indices[idx] = tmp_indices;
    }
  }

  void HiDR::bottom_up_sweep(long long cell_begin, long long ncells, const Cell cells[], const HiDR& lower_level) {
    this->lbegin = cell_begin;
    this->lend = lbegin + ncells;
    xbodies_indices.resize(ncells);
    fbodies_indices.resize(ncells);

    // loop over all the cells on the current level
    for (long long i = lbegin; i < lend; ++i) {
      //std::cout<<"Node: "<< i <<" contains pts ";
      long long idx = i - lbegin;
      std::vector<long long> tmp_indices;
      // collect the sampled points from the children
      for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; ++c) {
        long long child_idx = c - lower_level.lbegin;
        tmp_indices.insert(tmp_indices.end(), lower_level.xbodies_indices[child_idx].begin(), lower_level.xbodies_indices[child_idx].end());
      }
      // DATA REDUCT
      xbodies_indices[idx] = tmp_indices;
      //std::cout<<xbodies_indices[idx][0]<<"-"<<xbodies_indices[idx][tmp_indices.size()-1]<<std::endl;
    }
  }
  // TODO 1st level works, but there is still an error here when it is called for the 2nd level
  void HiDR::top_down_sweep(const Cell cells[], const CSR& Far, const HiDR& upper_level) {
    //std::cout<<"Top Down"<<std::endl;
    // loop over all the cells on the upper level
    for (long long i = upper_level.lbegin; i < upper_level.lend; ++i) {
      // collect the far field points from the upper level
      // and put them in the children
      for (long long c = cells[i].Child[0]; c < cells[i].Child[1]; ++c) {
        // check that the child is actually on this node
        if (lbegin <= c && c < lend) {
          fbodies_indices[c - lbegin] = std::vector<long long>(upper_level.fbodies_indices[i - upper_level.lbegin]);
        }
      }
    }
    // loop over all the cells on the current level
    for (long long c = lbegin; c < lend; ++c) {
      //std::cout<<"Node: "<<c<<std::endl;
      long long idx = c - lbegin;
      //std::cout<<idx<<", "<<fbodies_indices.size()<<std::endl;
      std::vector<long long> tmp_indices(fbodies_indices[idx]);
      // for each cell in the far field
      // TODO is this equivalent to the interaction list?
      //std::cout<<"Far RowIndex "<<Far.RowIndex[c]<<"-"<<Far.RowIndex[c+1]<<std::endl;
      for (long long i = Far.RowIndex[c]; i < Far.RowIndex[c + 1]; ++i) {
        //std::cout<<"Far ColIndex " <<Far.ColIndex[i]<<std::endl;
        long long j = Far.ColIndex[i] - lbegin;
        //std::cout<<"Far field node "<<j<<" contains pts ";
        tmp_indices.insert(std::lower_bound(tmp_indices.begin(), tmp_indices.end(), xbodies_indices[j][0]), xbodies_indices[j].begin(), xbodies_indices[j].end());
        //std::cout<<xbodies_indices[j][0]<<"-"<<xbodies_indices[j][xbodies_indices[j].size()-1]<<std::endl;
      }
      // DATA REDUCT
      fbodies_indices[idx] = tmp_indices;
      /*if (fbodies_indices[idx].size()){
      std::cout<<"Far field contains pts "<<fbodies_indices[idx][0]<<"-";
      for (size_t i = 1; i<fbodies_indices[idx].size(); ++i) {
        if (fbodies_indices[idx][i] != fbodies_indices[idx][i - 1] + 1)
          std::cout<<fbodies_indices[idx][i-1]<<", "<<fbodies_indices[idx][i]<<"-";
      }
      std::cout<<fbodies_indices[idx][fbodies_indices[idx].size() -1 ]<<std::endl;
      }*/

    }
  }

  // returns the number of sampled bodies for the cell with index i
  long long HiDR::fbodies_size_at_i(const long long i) const {
    return fbodies_indices[i].size();
  }
  // returns a pointer to the sampled bodies for the cell with index i
  const long long* HiDR::fbodies_at_i(const long long i) const {
    return fbodies_indices[i].data();
  }
