#ifdef __EXX
#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cassert>
#include "source_base/tool_title.h"
#include "source_io/module_parameter/parameter.h"
#include <map>
#include <RI/global/Tensor.h>

namespace BSE
{
    using TA = int;
    using TC = std::array<int, 3>;
    using TAC = std::pair<int, TC>;

    template <typename T>
    using TLRI = std::map<int, std::map<TAC, RI::Tensor<T>>>;

    inline void read_one_data(std::ifstream& ifs, double& data){
		std::string temp;
		ifs >> data >> temp;
	}    
    inline void read_one_data(std::ifstream& ifs, std::complex<double>& data){
		double real, imag;
		ifs >> real >> imag;
		data = std::complex<double>(real, imag);
	}
    
/// @brief pair:<occ, qs_energy>, vector as {ik, iband} 
/// @param ncore: as output, number of core orbitals parsed from file
std::vector<std::vector<std::pair<double,double>>> read_energy_qp(
    const std::string& file, const int nocc, const int nvirt, int& ncore, const int nk);
    
/// @brief read Wxc(R) = Wc(R) + Vx(R) from file
template<typename Tdata, typename TR>
std::map<TA,std::map<TAC,RI::Tensor<Tdata>>> read_Ws(const TLRI<TR>& Vs, const std::vector<TC>& Rlist);

}
#endif