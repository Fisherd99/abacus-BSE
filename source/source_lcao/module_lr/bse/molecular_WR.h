#pragma once
#include "source_base/global_function.h"
#include "source_lcao/module_ri/RI_Util.h"
#include "source_lcao/module_lr/ri_benchmark/ri_benchmark.h"
#include "source_lcao/module_lr/utils/lr_util_print.h"
#include "source_cell/unitcell.h"
#include "source_cell/klist.h"
namespace BSE
{

// alias following are also defined in ri_benchmark.h
using TA = int;
using TC = std::array<int, 3>;
using TAC = std::pair<int, TC>;    
template <typename T>
using TLRI = std::map<int, std::map<TAC, RI::Tensor<T>>>;
template <typename T>
using TCsR_mo = std::map<TA, std::map<std::pair<TC, TC>, RI::Tensor<T>>>;

/// @brief add value multiplied by phase to target
inline void add_c(double& target,
                const double& value,
                const std::complex<double>& phase)
{
    assert(std::abs(phase.real() - 1.0) < 1e-8);
    assert(std::abs(phase.imag()) < 1e-8);
    target += value;
};

inline void add_c(std::complex<double>& target,
                const std::complex<double>& value,
                const std::complex<double>& phase)
{
    target += value * phase;
};

template<typename T>
class MolecularWR
{
public:
    /// @brief calculate W[R_AI][R_BJ](i,a,j,b) with RI method
    MolecularWR(const UnitCell& ucell,
                const int& naos,
                const int& nk,
                const K_Vectors& kv_in,
                const int& nocc,
                const int& nvirt,
                const psi::Psi<T>& psi_ks_in, // < ATTENTION: psi_ks should be global
                std::weak_ptr<Exx_LRI<T>> exx_lri_in)
    : ucell(ucell), naos(naos), nk(nk), kv(kv_in), nocc(nocc), nvirt(nvirt), ndim(nk*nocc*nvirt),
    psi_ks(psi_ks_in), exx_lri(exx_lri_in)
    {
        ModuleBase::TITLE("MolecularWR", "MolecularWR");
        ModuleBase::timer::tick("MolecularWR", "MolecularWR");
        using RI::Communicate_Tensors_Map_Judge::comm_map2_first;
        
        this->period = {this->kv.nmp[0], this->kv.nmp[1], this->kv.nmp[2]};
        this->BvK_cells = RI_Util::get_Born_von_Karmen_cells(this->period);

        auto lri = this->exx_lri.lock();
        TLRI<T> Cs_ao = lri->exx_lri.lri.data_pool.at("Cs_").Ds_ab;

        //get all <I,<J,R>> Cs_ao
        std::set<int> list_I, list_J;
        for (int i = 0; i < this->ucell.nat; ++i)
        {
            list_I.insert(i);
            list_J.insert(i);
        }
        comm_map2_first(lri->mpi_comm, Cs_ao, list_I, list_J);

        // check whether Cs_ao is complete
        LRI_CV_Tools::write_Cs_ao(Cs_ao, PARAM.globalv.global_out_dir + "Cs_ao_in_WR_" + std::to_string(GlobalV::MY_RANK));

        this->Cs_oo_mo=cal_CsR_mo(ucell, Cs_ao, psi_ks_in, nocc, nvirt, LR::MO_TYPE::OO, "OO");
        this->Cs_vv_mo=cal_CsR_mo(ucell, Cs_ao, psi_ks_in, nocc, nvirt, LR::MO_TYPE::VV, "VV");

        // check
        //cal_CsR_mo_method2(ucell, Cs_ao, psi_ks_in, nocc, nvirt, LR::MO_TYPE::OO);
        //cal_CsR_mo_method2(ucell, Cs_ao, psi_ks_in, nocc, nvirt, LR::MO_TYPE::VV);
        ModuleBase::timer::tick("MolecularWR", "MolecularWR");
    };
    ~MolecularWR() {}

    /***********************************************************************/
    void cal_W_global(std::vector<T>& WA_global);

    // <iat, <{R_1-R_mu, R_2-R_mu}, tesnor{nabfs, nmo1, nmo2}>>
    TCsR_mo<T> cal_CsR_mo(const UnitCell& ucell,
                          const TLRI<T>& Cs_ao,
                          const psi::Psi<T>& psi_ks,
                          const int& nocc,
                          const int& nvirt,
                          const LR::MO_TYPE type,
                          const std::string& type_str);

    TCsR_mo<T> cal_CsR_mo_method2(const UnitCell& ucell,
                                  const TLRI<T>& Cs_ao,
                                  const psi::Psi<T>& psi_ks,
                                  const int& nocc,
                                  const int& nvirt,
                                  const LR::MO_TYPE type,
                                  const std::string& type_str);

    void print_CsR_mo_max(const TCsR_mo<T>& CsR_mo, const std::string& file_name);

    std::map<TAC, RI::Tensor<T>> slice_psi_R(const psi::Psi<T>& psi_ks,
                                            const int& imo,
                                            const int& nmo,
                                            const std::string& file_name);
protected:
    const UnitCell& ucell;
    const int& naos;
    const int& nk;
    const K_Vectors& kv;
    const int& nocc;
    const int& nvirt;
    const int ndim;
    const psi::Psi<T>& psi_ks;
    TC period;
    std::vector<TC> BvK_cells;
    std::weak_ptr<Exx_LRI<T>> exx_lri;
    TCsR_mo<T> Cs_oo_mo;
    TCsR_mo<T> Cs_vv_mo;

};// class MolecularWR

}// namespace BSE

#include "molecular_WR.hpp"