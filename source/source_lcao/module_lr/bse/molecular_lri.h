#pragma once
#include <RI/physics/LR.h>
#include "source_base/timer.h"
#include "source_base/global_function.h"
#include "source_base/module_container/base/third_party/blas.h"
#include "source_cell/unitcell.h"
#include "source_cell/klist.h"
#include "source_lcao/module_lr/bse/bse_util.h"
#include "source_lcao/module_lr/utils/lr_util.h"
#include "source_lcao/module_lr/utils/lr_util_print.h"
#include "source_lcao/module_lr/ao_to_mo_transformer/ao_to_mo.h"
#include "source_lcao/module_ri/LRI_CV_Tools.h"

namespace BSE
{

using TA = int;
using TC = std::array<int, 3>;
using Tk = std::array<double, 3>;
using TAC = std::pair<int, TC>;
using TatomR = std::array<double, 3>;

template <typename T>
using TLRI = std::map<TA, std::map<TAC, RI::Tensor<T>>>;
template <typename T>
using TLRIk = std::map<Tk, std::map<TA, std::map<TA, RI::Tensor<T>>>>;
template <typename T>
using TCsk_ao_mo = std::map<Tk, std::map<TA, RI::Tensor<T>>>;
template <typename T>
using TCsk_mo = std::map<std::pair<Tk, Tk>, std::map<TA, RI::Tensor<T>>>;

template<typename T>
class MolecularLRI
{
public:
    RI::LR<int,int,3,T> LR_lri;
    RI::Cell_Nearest<int, int, 3, double, 3> cell_nearest;
    /// @brief calculate V[k_ai][k_bj](j,b,i,a) and W[k_ai][k_bj](j,b,i,a) with RI method
    MolecularLRI(const UnitCell& ucell,
        const int nk,
        const K_Vectors& kv_in,
        const int nocc,
        const int nvirt,
        const psi::Psi<T>& psi_ks_in) // < ATTENTION: psi_ks should be global
    : ucell(ucell), nk(nk), kv(kv_in), nocc(nocc), nvirt(nvirt), ndim(nk*nocc*nvirt),
    psi_ks(psi_ks_in), period({kv_in.nmp[0], kv_in.nmp[1], kv_in.nmp[2]})
    {
        for (int i = 0; i < nk; ++i) // nk without spin, ignore nspin2 temporarily
        {
            Tk k_d = RI_Util::Vector3_to_array3(this->kv.kvec_d.at(i));
            this->kpoint_index_map[k_d] = i;
        }

        std::map<TA, TatomR> atoms_pos;
        for (int iat = 0; iat < this->ucell.nat; ++iat)
        {
            atoms_pos[iat] = RI_Util::Vector3_to_array3(
                this->ucell.atoms[this->ucell.iat2it[iat]].tau[this->ucell.iat2ia[iat]]);
        }
        const std::array<TatomR, 3> latvec = {RI_Util::Vector3_to_array3(this->ucell.a1),
                                              RI_Util::Vector3_to_array3(this->ucell.a2),
                                              RI_Util::Vector3_to_array3(this->ucell.a3)};
        this->LR_lri.set_parallel(MPI_COMM_WORLD, atoms_pos, latvec, period);
        this->cell_nearest.init(atoms_pos, latvec, period);
    };
    
    ~MolecularLRI() {}

    /// =============== calculation interface ====================
    void cal_W_for_A(std::vector<T>& m_global)
    {
        //TCsk_mo<T> Csk_oo_k21 = slice_Csk_mo(nocc, nvirt, LR_Util::MO_TYPE::OO, "OOk21", this->k2_list, this->k1_list, this->list_I);
        //TCsk_mo<T> Csk_vv_k12 = slice_Csk_mo(nocc, nvirt, LR_Util::MO_TYPE::VV, "VVk12", this->k1_list, this->k2_list, this->list_J);
    // check
    // cal_Csk_mo_method2(ucell, Csk_ao, psi_ks, nocc, nvirt, LR_Util::MO_TYPE::OO, "OO", this->k2_list, this->k1_list);
    // cal_Csk_mo_method2(ucell, Csk_ao, psi_ks, nocc, nvirt, LR_Util::MO_TYPE::VV, "VV", this->k1_list, this->k2_list);
        ModuleBase::TITLE("MolecularLRI", "cal_W_for_A");
        ModuleBase::timer::tick("MolecularLRI", "cal_W_for_A");
        std::map<Tk, std::map<Tk, RI::Tensor<T>>>
            Wk = LR_lri.lri.cal_cvc_mo_k_onthefly(this->Csk_ao_mo, this->map_psi, k1_list, k2_list, list_I, list_J, cell_nearest,
                {"O","O","V","V"}, (std::size_t)nocc, (std::size_t)nvirt, "Ws_", GlobalV::ofs_running, { 0,2,1,3 }); // (jiba) -> (jbia)
        ModuleBase::timer::tick("MolecularLRI", "cal_W_for_A");
        this->transform_k_global(m_global, Wk);
    }
    void cal_W_for_B(std::vector<T>& m_global)
    {
        ModuleBase::TITLE("MolecularLRI", "cal_W_for_B");
        ModuleBase::timer::tick("MolecularLRI", "cal_W_for_B");
        std::map<Tk, std::map<Tk, RI::Tensor<T>>>
            Wk = LR_lri.lri.cal_cvc_mo_k_onthefly(this->Csk_ao_mo, this->map_psi, k1_list, k2_list, list_I, list_J, cell_nearest,
                {"V","O","O","V"}, (std::size_t)nocc, (std::size_t)nvirt, "Ws_", GlobalV::ofs_running, { 2,0,1,3 }); // (bija) -> (jbia)
        ModuleBase::timer::tick("MolecularLRI", "cal_W_for_B");
        this->transform_k_global(m_global, Wk);
    }
    void cal_hartree_for_A(std::vector<T>& m_global)
    {
        ModuleBase::TITLE("MolecularLRI", "cal_hartree_for_A");
        ModuleBase::timer::tick("MolecularLRI", "cal_hartree_for_A");
        std::map<Tk, std::map<Tk, RI::Tensor<T>>>
            Vk = LR_lri.lri.cal_cvc_mo_k_hartree_onthefly(this->Csk_ao_mo, this->map_psi, k1_list, k2_list, list_I, list_J,
                {"O","V","O","V"}, (std::size_t)nocc, (std::size_t)nvirt, "Vs_", true);
        ModuleBase::timer::tick("MolecularLRI", "cal_hartree_for_A");
        this->transform_k_global(m_global, Vk);
    }
    void cal_hartree_for_B(std::vector<T>& m_global)
    {
        ModuleBase::TITLE("MolecularLRI", "cal_hartree_for_B");
        ModuleBase::timer::tick("MolecularLRI", "cal_hartree_for_B");
        std::map<Tk, std::map<Tk, RI::Tensor<T>>>
            Vk = LR_lri.lri.cal_cvc_mo_k_hartree_onthefly(this->Csk_ao_mo, this->map_psi, k1_list, k2_list, list_I, list_J,
                {"O","V","O","V"}, (std::size_t)nocc, (std::size_t)nvirt, "Vs_", false);
        ModuleBase::timer::tick("MolecularLRI", "cal_hartree_for_B");
        this->transform_k_global(m_global, Vk);
    }

    void init(TLRI<T>& Cs_in, TLRI<T>& Vs_in, TLRI<T>& Ws_in, const Exx_Info::Exx_Info_RI& info_ri);

    /// =============== print ====================
    inline void print_k(std::ostream& ofs, const std::vector<Tk>& vec, const std::string name)
    {
        ofs << name << ": size = " << vec.size() << std::endl;
        ofs << std::fixed << std::setprecision(4);
        int count = 0;
        for (auto& v : vec)
        {
            ofs << "(" << std::setw(6) << v[0] <<", "<< std::setw(6) << v[1] <<", "<< std::setw(6) << v[2] <<") ";
            count++;
            if (count % 5 == 0) { ofs << std::endl; }
        }
        ofs << std::endl;
        ofs << std::defaultfloat;
    }
    inline void print_a(std::ostream& ofs, const std::vector<int>& vec, const std::string name)
    {
        ofs << name << ": ";
        int count = 0;
        for (auto& v : vec)
        {
            ofs << v << " ";
            count++;
            if (count % 10 == 0) { ofs << std::endl; }
        }
        ofs << std::endl;
    }

protected:
    /// =============== inner function ====================
    void transform_k_global(std::vector<T>& m_global, std::map<Tk, std::map<Tk, RI::Tensor<T>>>& m_lri);

    // <k, <I, <J, tesnor{nabfs, nmo1, nmo2}>>>
    TLRIk<T> cal_Csk_ao(const TLRI<T>& CsR_ao, const std::vector<Tk>& k_list, const std::vector<TA>& list_IJ);

    // <k, <iat, tesnor{nabfs, nw, nmo}>>
    TCsk_ao_mo<T> cal_Csk_ao_mo(const UnitCell& ucell,
        const TLRIk<T>& Csk_ao,
        const std::vector<Tk>& k_list,
        const std::vector<TA>& list_IJ);
    
    // transform total psi to map type according k coordinate and atom index
    std::map<Tk, std::map<TA, RI::Tensor<T>>> transform_psi_k(const psi::Psi<T>& psi_ks,
        const std::vector<Tk>& k_list);
    
    /// ===== Below are functions not used, and reserver for reference =====
    
    // slice psi part according to imo and nmo, and return map type
    std::map<Tk, std::map<TA, RI::Tensor<T>>> slice_psi_k(const psi::Psi<T>& psi_ks,
            const int imo,
            const std::size_t nmo,
            const std::vector<Tk>& k_list);
    
    // <{k1, k2}, <iat, tesnor{nabfs, nmo1, nmo2}>>
    TCsk_mo<T> cal_Csk_mo(const UnitCell& ucell,
                          const TLRIk<T>& Csk_ao,
                          const psi::Psi<T>& psi_ks,
                          const std::vector<Tk>& k_list,
                          const std::vector<TA>& list_IJ);

    // slice Csk_mo according to k1_list and k2_list
    TCsk_mo<T> slice_Csk_mo(const int nocc,
                            const int nvirt,
                            const LR_Util::MO_TYPE type,
                            const std::string type_str,
                            const std::vector<Tk>& kmo1_list,
                            const std::vector<Tk>& kmo2_list,
                            const std::vector<TA>& list_IJ);

    TCsk_mo<T> cal_Csk_mo_method2(const UnitCell& ucell,
                                  const TLRIk<T>& Csk_ao,
                                  const psi::Psi<T>& psi_ks,
                                  const int nocc,
                                  const int nvirt,
                                  const LR_Util::MO_TYPE type,
                                  const std::string type_str,
                                  const std::vector<Tk>& kmo1_list,
                                  const std::vector<Tk>& kmo2_list);

    void print_Csk_mo_max(const TCsk_mo<T>& Csk_mo, const std::string file_name);

    const UnitCell& ucell;
    const int nk;
    const K_Vectors& kv;
    const TC period;
    const int nocc;
    const int nvirt;
    const int ndim;
    const psi::Psi<T>& psi_ks;
    std::map<Tk, std::map<TA, RI::Tensor<T>>> map_psi;
    std::vector<int> list_I;
    std::vector<int> list_J;
    std::vector<int> list_IJ;
    std::vector<Tk> k1_list;
    std::vector<Tk> k2_list;
    std::vector<Tk> k_list;
    std::map<Tk, int> kpoint_index_map;
    TCsk_ao_mo<T> Csk_ao_mo;
    TCsk_mo<T> Csk_mo;
};// class MolecularLRI



// class MolecularWR is abandoned, reserve for test reference

template <typename T>
using TCsR_mo = std::map<TA, std::map<std::pair<TC, TC>, RI::Tensor<T>>>;
template<typename T>
class MolecularWR
{
public:
    /// @brief calculate W[R_AI][R_BJ](j,b,i,a) with RI method
    MolecularWR(const UnitCell& ucell,
                const int nk,
                const K_Vectors& kv_in,
                const int nocc,
                const int nvirt,
                const psi::Psi<T>& psi_ks_in, // < ATTENTION: psi_ks should be global
                RI::LR<int,int,3,T>& LR_lri_in)
    : ucell(ucell), nk(nk), kv(kv_in), nocc(nocc), nvirt(nvirt), ndim(nk*nocc*nvirt),
    psi_ks(psi_ks_in), LR_lri(LR_lri_in)
    {
        ModuleBase::TITLE("MolecularWR", "MolecularWR");
        ModuleBase::timer::tick("MolecularWR", "MolecularWR");
        using RI::Communicate_Tensors_Map_Judge::comm_map2_first;
        
        this->period = {this->kv.nmp[0], this->kv.nmp[1], this->kv.nmp[2]};
        this->BvK_cells = RI_Util::get_Born_von_Karmen_cells(this->period);

        TLRI<T>& Cs_ao = LR_lri.lri.data_pool.at("Cs_").Ds_ab;

        //get all <I,<J,R>> Cs_ao
        std::set<int> list_I, list_J;
        for (int i = 0; i < this->ucell.nat; ++i)
        {
            list_I.insert(i);
            list_J.insert(i);
        }
        Cs_ao = comm_map2_first(LR_lri.lri.mpi_comm, Cs_ao, list_I, list_J);

        // check Cs_ao
        LRI_CV_Tools::write_Cs_ao(Cs_ao, PARAM.globalv.global_out_dir + "Cs_ao_in_WR_" + std::to_string(GlobalV::MY_RANK));

        this->CsR_oo_mo=cal_CsR_mo(ucell, Cs_ao, psi_ks_in, nocc, nvirt, LR_Util::MO_TYPE::OO, "OO");
        this->CsR_vv_mo=cal_CsR_mo(ucell, Cs_ao, psi_ks_in, nocc, nvirt, LR_Util::MO_TYPE::VV, "VV");

        // check
        //cal_CsR_mo_method2(ucell, Cs_ao, psi_ks_in, nocc, nvirt, LR_Util::MO_TYPE::OO, "OO");
        //cal_CsR_mo_method2(ucell, Cs_ao, psi_ks_in, nocc, nvirt, LR_Util::MO_TYPE::VV, "VV");
        ModuleBase::timer::tick("MolecularWR", "MolecularWR");
    };
    ~MolecularWR() {}

    /***********************************************************************/
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

    /***********************************************************************/
    void cal_W_global(std::vector<T>& WA_global);

    // <iat, <{R_1-R_mu, R_2-R_mu}, tesnor{nabfs, nmo1, nmo2}>>
    TCsR_mo<T> cal_CsR_mo(const UnitCell& ucell,
                          const TLRI<T>& Cs_ao,
                          const psi::Psi<T>& psi_ks,
                          const int nocc,
                          const int nvirt,
                          const LR_Util::MO_TYPE type,
                          const std::string type_str);

    TCsR_mo<T> cal_CsR_mo_method2(const UnitCell& ucell,
                                  const TLRI<T>& Cs_ao,
                                  const psi::Psi<T>& psi_ks,
                                  const int nocc,
                                  const int nvirt,
                                  const LR_Util::MO_TYPE type,
                                  const std::string type_str);

    void print_CsR_mo_max(const TCsR_mo<T>& CsR_mo, const std::string file_name);

    std::map<TAC, RI::Tensor<T>> slice_psi_R(const psi::Psi<T>& psi_ks,
                                            const int imo,
                                            const int nmo,
                                            const std::string file_name);
protected:
    const UnitCell& ucell;
    const int nk;
    const K_Vectors& kv;
    const int nocc;
    const int nvirt;
    const int ndim;
    const psi::Psi<T>& psi_ks;
    TC period;
    std::vector<TC> BvK_cells;
    RI::LR<int,int,3,T>& LR_lri;
    TCsR_mo<T> CsR_oo_mo;
    TCsR_mo<T> CsR_vv_mo;

};// class MolecularWR

}// namespace BSE

#include "molecular_lri.hpp"
#include "molecular_WR.hpp"
