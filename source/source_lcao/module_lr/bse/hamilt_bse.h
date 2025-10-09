// Purpose: simplify the hamilt_casida class, and explore to go beyond Tamm-Damcoff Approximation.
// For simplicity, only ELPA solver with MPI parallization is implementated.
// Thus, instead of iterative solver such as Davidson, here matrix is constructed directly.

#pragma once
#include "bse_io.h"
#include "bse_util.h"
#include "hamilt_bse_solver.h"
#include "source_cell/unitcell.h"
#include "source_base/parallel_2d.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_estate/module_dm/density_matrix.h"
#include "source_hamilt/hamilt.h"
#include "source_lcao/module_lr/dm_trans/dm_trans.h"
#include "source_lcao/module_lr/potentials/pot_hxc_lrtd.h"
#include "source_lcao/module_lr/ao_to_mo_transformer/ao_to_mo.h"
#include "source_lcao/module_lr/operator_casida/operator_lr_diag.h"
#include "source_lcao/module_lr/operator_casida/operator_lr_exx.h"
//#include "source_lcao/module_lr/operator_casida/operator_lr_hxc.h"
#include "source_lcao/module_lr/ri_benchmark/operator_ri_hartree.h"
#include "source_lcao/module_lr/ri_benchmark/ri_benchmark.h"
#include "source_lcao/module_lr/utils/lr_util.h"
#include "source_lcao/module_ri/LRI_CV_Tools.h"
#include "source_lcao/module_hcontainer/hcontainer_funcs.h"
#include "source_base/timer.h"

#include <typeinfo>

namespace BSE
{
template <typename T>
class HamiltBSE
{
  public:
    std::vector<T> BSE_A_global, BSE_B_global;
    std::vector<T> VA_global, WA_global;
    std::vector<T> VB_global, WB_global;
    std::vector<double> evals;
    std::vector<std::complex<double>> evecs; // eigenvectors in complex format
    int ndim = 0; // dimension of BSE matrix
    Parallel_2D pA;
    /// @brief constructor for BSE_Matrix
    HamiltBSE(const int& nspin,
              const int& naos,
              const std::vector<int>& nocc,
              const std::vector<int>& nvirt,
              const UnitCell& ucell_in,
              const std::vector<double>& orb_cutoff_in,
              const Grid_Driver& gd_in,
              const psi::Psi<T>& psi_ks_in,
              const ModuleBase::matrix& eig_gw_in,
#ifdef __EXX
              std::weak_ptr<Exx_LRI<T>> exx_lri_in,
#endif
              //typename LR::TGint<T>::type* gint_in,
              std::weak_ptr<LR::PotHxcLR> pot_in,
              const K_Vectors& kv_in,
              const std::vector<Parallel_2D>& pX_in,
              const Parallel_2D& pc_in,
              const Parallel_Orbitals& pmat_in,
              const std::vector<std::string>& spin_types_in,
              const std::string& tda,
              const std::string& ri_hartree_benchmark_in = "none");

    void cal_V_for_A();
    void cal_W_for_A();
    void cal_V_for_B() {std::cout << "cal_V_for_B() is not implemented yet." << std::endl;};
    void cal_W_for_B() {std::cout << "cal_W_for_B() is not implemented yet." << std::endl;};
    void tda_solver(const int& st_index, const int& nstates, double* ene_out, T* X_out);
    void full_solver(const int& st_index, const int& nstates, double* ene_out, T* X_out, T* Y_out);
    void grid_calculation(hamilt::HContainer<T>& VR) const;
  private:
    const int nspin;
    const int naos;
    const std::vector<int> nocc;
    const std::vector<int> nvirt;
    const UnitCell& ucell;
    const std::vector<double>& orb_cutoff;
    const Grid_Driver& gd;
    const psi::Psi<T>& psi_ks;
    const ModuleBase::matrix& eig_gw;
#ifdef __EXX
    std::weak_ptr<Exx_LRI<T>> exx_lri;
#endif
    //typename LR::TGint<T>::type* gint;
    std::weak_ptr<LR::PotHxcLR> pot;
    const K_Vectors& kv;
    int nk = 1;
    const std::vector<Parallel_2D>& pX; // for tda, also pY for full
    const Parallel_2D& pc;
    const Parallel_Orbitals& pmat;
    const std::vector<std::string>& spin_types; // singlet / triplet
    const std::string ri_hartree_benchmark;

    std::unique_ptr<elecstate::DensityMatrix<T, T>> DM_trans;
};
} // namespace BSE