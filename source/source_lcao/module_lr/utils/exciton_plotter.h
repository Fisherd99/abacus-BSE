#pragma once
#include "source_base/tool_title.h"
#include "source_cell/klist.h"
#include "source_estate/module_dm/density_matrix.h"
#include "source_io/module_parameter/parameter.h"
#include "source_io/cube_io.h"
#include "source_lcao/module_lr/dm_trans/dm_trans.h"
#include "source_lcao/module_lr/utils/lr_util.h"
#include "source_lcao/module_lr/utils/lr_util_hcontainer.h"
#include "source_lcao/module_gint/temp_gint/gint_interface.h"
#include <string>

namespace LR_Util
{
template <typename T>
class ExcitonPlotter
{
    // 目标：实现tda和full激子的绘制，同时支持固定空穴原胞和对空穴原胞求和
public:
    ExcitonPlotter(const int nspin_global, const int naos, const std::vector<int>& nocc, const std::vector<int>& nvirt,
                   const psi::Psi<T>& psi_ks_in,
                   const UnitCell& ucell_in,
                   const K_Vectors& kv_in,
                   const Grid_Driver& gd_in, const std::vector<double>& orb_cutoff_in, const Parallel_Grid& Pgrid_in,
                   const ModulePW::PW_Basis& rho_basis_in,
                   const std::vector<Parallel_2D>& pX_in, const Parallel_2D& pc_in, const Parallel_Orbitals& pmat_in,
                   const double* eig, const T* X,
                   const bool openshell)
    : nspin_x(openshell ? 2 : 1), naos(naos), nocc(nocc), nvirt(nvirt), ucell(ucell_in), kv(kv_in),
      gd_(gd_in), orb_cutoff_(orb_cutoff_in), Pgrid(Pgrid_in), rho_basis(rho_basis_in),
      pX(pX_in), pc(pc_in), pmat(pmat_in), eig(eig), X(X),
      nk(nspin_global == 2 ? kv_in.get_nks() / 2 : kv_in.get_nks()),
      ldim(nk* (nspin_x == 2 ? pX_in[0].get_local_size() + pX_in[1].get_local_size() : pX_in[0].get_local_size())),
      gdim(nk* std::inner_product(nocc.begin(), nocc.end(), nvirt.begin(), 0))
    {
        for (int is = 0;is < nspin_global;++is) { psi_ks_vec.emplace_back(LR_Util::get_psi_spin(psi_ks_in, is, nk)); }
    };
    void set_Y(T* Y_in) { this->Y = Y_in; };
    void set_full(bool tag) { this->is_full = tag; };
    elecstate::DensityMatrix<T, T> cal_transition_density_matrix(const int istate);
    void plot_exciton(const int istate, const std::string& type);
private:
    const int nspin_x = 1; ///< 1 for singlet/triplet, 2 for updown(openshell)
    const int naos;
    const std::vector<int>& nocc;
    const std::vector<int>& nvirt;
    const int nk;
    const int ldim; ///< local leading dimension of X, or the data size of each state
    const int gdim; ///< global leading dimension of X
    const double* eig;
    const T* X;
    T* Y = nullptr; ///< the deexcitation part of amplitudes
    bool is_full = false;
    const K_Vectors& kv;
    std::vector<psi::Psi<T>> psi_ks_vec;
    const UnitCell& ucell;
    const std::vector<double>& orb_cutoff_;
    const Grid_Driver& gd_;
    const Parallel_Grid& Pgrid;
    const ModulePW::PW_Basis& rho_basis;

    const std::vector<Parallel_2D>& pX;
    const Parallel_2D& pc;
    const Parallel_Orbitals& pmat;    

};


} // namespace LR_Util