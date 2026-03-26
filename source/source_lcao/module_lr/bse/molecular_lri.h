//=======================
// AUTHOR : Ziqing Guan
// DATE :   2026-03-22
//=======================
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
#include "source_lcao/module_lr/utils/lr_io.h"
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

template<typename T>
class MolecularLRI
{
public:
    RI::LR<int,int,3,T> LR_lri;
    RI::Cell_Nearest<int, int, 3, double, 3> cell_nearest;
    /// @brief calculate V[k_ai][k_bj](j,b,i,a) and W[k_ai][k_bj](j,b,i,a) with RI method
    MolecularLRI(const UnitCell& ucell,
        const int nk,
        const LR_IO::RI_kRlist& kRlist_in,
        const int nocc,
        const int nvirt,
        const psi::Psi<T>& psi_ks_in) // < ATTENTION: psi_ks should be global
    : ucell(ucell), nk(nk), kRlist(kRlist_in), kv(*kRlist_in.klist), nocc(nocc), nvirt(nvirt),
    ndim(nk*nocc*nvirt), psi_ks(psi_ks_in), is_local_k1(nk, false)
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
        this->LR_lri.set_parallel(MPI_COMM_WORLD, atoms_pos, latvec, kRlist.period);
        this->cell_nearest.init(atoms_pos, latvec, kRlist.period);
    };
    
    ~MolecularLRI() {}

    /// =============== calculation interface ====================
    void init(TLRI<T>& Cs_in, TLRI<T>& Vs_in, TLRI<T>& Ws_in, const Exx_Info::Exx_Info_RI& info_ri);

    void cal_W_for_A(std::vector<T>& m_2d, const Parallel_2D& pm_2d, const double beta=1.0)
    {
        ModuleBase::TITLE("MolecularLRI", "cal_W_for_A");
        ModuleBase::timer::tick("MolecularLRI", "cal_W_for_A");
        std::map<Tk, std::map<Tk, RI::Tensor<T>>>
            Wk = LR_lri.lri.cal_cvc_mo_k_onthefly(this->Csk_ao_mo, this->map_psi, k1_list, k2_list, list_I, list_J,
                {"O","O","V","V"}, (std::size_t)nocc, (std::size_t)nvirt, "Ws_", GlobalV::ofs_running, { 0,2,1,3 }); // (jiba) -> (jbia)
        ModuleBase::timer::tick("MolecularLRI", "cal_W_for_A");
        this->transform_k_2dlocal(m_2d, Wk, pm_2d, beta);
    }
    void cal_W_for_B(std::vector<T>& m_2d, const Parallel_2D& pm_2d, const double beta=1.0)
    {
        ModuleBase::TITLE("MolecularLRI", "cal_W_for_B");
        ModuleBase::timer::tick("MolecularLRI", "cal_W_for_B");
        std::map<Tk, std::map<Tk, RI::Tensor<T>>>
            Wk = LR_lri.lri.cal_cvc_mo_k_onthefly(this->Csk_ao_mo, this->map_psi, k1_list, k2_list, list_I, list_J,
                {"V","O","O","V"}, (std::size_t)nocc, (std::size_t)nvirt, "Ws_", GlobalV::ofs_running, { 2,0,1,3 }); // (bija) -> (jbia)
        ModuleBase::timer::tick("MolecularLRI", "cal_W_for_B");
        this->transform_k_2dlocal(m_2d, Wk, pm_2d, beta);
    }
    void cal_hartree_for_A(std::vector<T>& m_2d, const Parallel_2D& pm_2d, const double beta=1.0)
    {
        ModuleBase::TITLE("MolecularLRI", "cal_hartree_for_A");
        ModuleBase::timer::tick("MolecularLRI", "cal_hartree_for_A");
        std::map<Tk, std::map<Tk, RI::Tensor<T>>>
            Vk = LR_lri.lri.cal_cvc_mo_k_hartree_onthefly(this->Csk_ao_mo, this->map_psi, k1_list, k2_list, list_I, list_J,
                {"O","V","O","V"}, (std::size_t)nocc, (std::size_t)nvirt, "Vs_", true);
        ModuleBase::timer::tick("MolecularLRI", "cal_hartree_for_A");
        this->transform_k_2dlocal(m_2d, Vk, pm_2d, beta);
    }
    void cal_hartree_for_B(std::vector<T>& m_2d, const Parallel_2D& pm_2d, const double beta=1.0)
    {
        ModuleBase::TITLE("MolecularLRI", "cal_hartree_for_B");
        ModuleBase::timer::tick("MolecularLRI", "cal_hartree_for_B");
        std::map<Tk, std::map<Tk, RI::Tensor<T>>>
            Vk = LR_lri.lri.cal_cvc_mo_k_hartree_onthefly(this->Csk_ao_mo, this->map_psi, k1_list, k2_list, list_I, list_J,
                {"O","V","O","V"}, (std::size_t)nocc, (std::size_t)nvirt, "Vs_", false);
        ModuleBase::timer::tick("MolecularLRI", "cal_hartree_for_B");
        this->transform_k_2dlocal(m_2d, Vk, pm_2d, beta);
    }

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
    void transform_k_2dlocal(std::vector<T>& m_2d,
        const std::map<Tk, std::map<Tk, RI::Tensor<T>>>& m_lri,
        const Parallel_2D& pm_2d, const double beta);

    // <k, <iat, tesnor{nabfs, nw, nmo}>>
    TCsk_ao_mo<T> cal_Csk_ao_mo(const TLRI<T>& CsR_ao,
        const std::vector<Tk>& k_list,
        const std::vector<TA>& list_IJ);
    
    // transform total psi to map type according k coordinate and atom index
    std::map<Tk, std::map<TA, RI::Tensor<T>>> transform_psi_k(const psi::Psi<T>& psi_ks,
        const std::vector<Tk>& k_list);

    const UnitCell& ucell;
    const int nk;
    const K_Vectors& kv;
    const LR_IO::RI_kRlist& kRlist;
    const Parallel_2D pm_2d;
    const int nocc;
    const int nvirt;
    const int ndim;
    const psi::Psi<T>& psi_ks;
    std::map<Tk, std::map<TA, RI::Tensor<T>>> map_psi;
    std::vector<int> list_I;
    std::vector<int> list_J;
    std::vector<int> list_IJ;
    std::vector<bool> is_local_k1;
    std::vector<Tk> k1_list;
    std::vector<Tk> k2_list;
    std::vector<Tk> k_list;
    std::map<Tk, int> kpoint_index_map;
    TCsk_ao_mo<T> Csk_ao_mo;
};// class MolecularLRI

}// namespace BSE

#include "molecular_lri.hpp"
#include "molecular_lri_comm.hpp"
