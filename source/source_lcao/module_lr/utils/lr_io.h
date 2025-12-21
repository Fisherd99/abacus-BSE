#ifdef __EXX
#pragma once
#include "source_base/tool_title.h"
#include "source_io/module_parameter/parameter.h"
#include "source_lcao/module_ri/RI_Util.h" // for get_Born_von_Karmen_cells

#include <RI/global/Tensor.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace LR_IO
{
using TA = int;
using TC = std::array<int, 3>;
using TAC = std::pair<int, TC>;

template <typename T>
using TLRI = std::map<int, std::map<TAC, RI::Tensor<T>>>;

inline void read_one_data(std::ifstream& ifs, double& data)
{
    std::string temp;
    ifs >> data >> temp;
}
inline void read_one_data(std::ifstream& ifs, std::complex<double>& data)
{
    double real, imag;
    ifs >> real >> imag;
    data = std::complex<double>(real, imag);
}

inline void parse_band_out_file(const std::string& file, int& nbands_file, int& nk_file, int& nspin_file, int& nocc_file)
{
    std::ifstream ifs(file);
    if (!ifs) throw std::runtime_error(file + " not found");
    std::string tmp, line;
    double occ;
    int nocc_count = 0;

    ifs >> nk_file >> nspin_file >> nbands_file;
    for (int i = 0; i < 4; ++i) {std::getline(ifs, tmp); } //skip 4 lines

    while (ifs.peek() != EOF)
    {
        std::getline(ifs, line);
        std::istringstream iss(line);
        
        iss >> tmp >> occ;
        if (occ > 0.1) nocc_count++;
        else if (occ < 0.1) break;
    }
    nocc_file = nocc_count;
}

class RI_kRlist
{
  public:
    K_Vectors* klist = nullptr;
    std::vector<TC> Rlist;
    RI_kRlist() = default;
    RI_kRlist(const std::string& file, const UnitCell& ucell, K_Vectors* pkv);
    ~RI_kRlist() = default;
    void read_kpts(const std::string& file, const UnitCell& ucell, K_Vectors* klist);
};

/// @brief vector as {ik, iband, <occ, ks_ene, gw_ene>}
/// @param ncore: as output, number of core orbitals parsed from file
std::vector<double> read_energy_qp(const std::string& file,
                                    const int nocc,
                                    const int nvirt,
                                    int& ncore,
                                    const int nk,
                                    const int nspin_tmp,
                                    const int nspin_file);
template <typename TK>
void read_librpa_eigenvectors(psi::Psi<TK>& wfc_ks,
                              psi::Psi<TK>& wfc_ks_global,
                              const std::string& path,
                              const int ncore,
                              const int nbands_file,
                              const int nspin_tmp,
                              const int nspin_file,
                              Parallel_Orbitals& pmat);

/// only for blocking by atom pairs (abacus type)
template <typename TCs, typename TR>
TLRI<TR> read_coulomb_mat_k(const std::string& path, const TLRI<TCs>& Cs, const LR_IO::RI_kRlist& kRlist);

/// for any way of blocking (aims type)
template <typename TCs, typename TR>
TLRI<TR> read_coulomb_mat_general_k(const std::string& path, const TLRI<TCs>& Cs, const LR_IO::RI_kRlist& kRlist);

/// @brief read Wxc(R) = Wc(R) + Vx(R) from file
template <typename Tdata, typename TR>
std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> read_Ws(const TLRI<TR>& Vs, const std::vector<TC>& Rlist);

} // namespace BSE
#endif