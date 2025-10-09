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

namespace BSE_IO
{
using TA = int;
using TC = std::array<int, 3>;
using TAC = std::pair<int, TC>;

template <typename T>
using TLRI = std::map<int, std::map<TAC, RI::Tensor<T>>>;

class RI_kRlist
{
  public:
    std::unique_ptr<K_Vectors> klist;
    std::vector<TC> Rlist;
    RI_kRlist(const std::string& file, const UnitCell& ucell);
    ~RI_kRlist() = default;
    void read_kpts(const std::string& file, const UnitCell& ucell, std::unique_ptr<K_Vectors>& klist);
};

inline void parse_band_out_file(const std::string& file, int& nbands_file, int& nk_file, int& nspin_file)
{
    std::ifstream ifs(file);
    if (!ifs) throw std::runtime_error(file + " not found");

    ifs >> nk_file >> nspin_file >> nbands_file;
}
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

/// @brief vector as {ik, iband, <occ, ks_ene, gw_ene>}
/// @param ncore: as output, number of core orbitals parsed from file
std::vector<double> read_energy_qp(const std::string& file,
                                    const int nocc,
                                    const int nvirt,
                                    int& ncore,
                                    const int nk,
                                    const int nspin_tmp,
                                    const int nspin_file);

/// @brief read Wxc(R) = Wc(R) + Vx(R) from file
template <typename Tdata, typename TR>
std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> read_Ws(const TLRI<TR>& Vs, const std::vector<TC>& Rlist);

} // namespace BSE
#endif