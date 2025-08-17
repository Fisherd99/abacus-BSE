// Purpose: simplify the hamilt_casida class, and explore to go beyond Tamm-Damcoff Approximation.
// For simplicity, only ELPA solver with MPI parallization is implementated.
// Thus, instead of iterative solver such as Davidson, here matrix is constructed directly.

#pragma once
#define HAVE_SKEWSYMMETRIC // for elpa_skew_eigenvectors
#include "bse_io.h"
#include "source_base/module_external/blacs_connector.h"
#include "source_base/module_external/scalapack_connector.h"
#include "source_base/parallel_2d.h"
#include "source_base/parallel_reduce.h"
#include "source_basis/module_ao/parallel_orbitals.h"

#include "source_estate/module_dm/density_matrix.h"
#include "source_hamilt/hamilt.h"

#include "source_lcao/module_lr/dm_trans/dm_trans.h"
#include "source_lcao/module_lr/operator_casida/operator_lr_diag.h"
#include "source_lcao/module_lr/operator_casida/operator_lr_exx.h"
#include "source_lcao/module_lr/operator_casida/operator_lr_hxc.h"
#include "source_lcao/module_lr/ri_benchmark/operator_ri_hartree.h"
#include "source_lcao/module_lr/ri_benchmark/ri_benchmark.h"
#include "source_lcao/module_lr/utils/lr_util.h"
#include "source_lcao/module_ri/LRI_CV_Tools.h"

#include <typeinfo>

namespace BSE
{
template <typename T>
void printM(const std::vector<T>& A, int m, int n, std::string file, std::string name);

/// @brief result={{A,B},{-A*,-B*}}
void arrayFlatten1(int nA /*A_part dim*/,
                   const std::vector<std::complex<double>>& A,
                   const std::vector<std::complex<double>>& B,
                   std::vector<std::complex<double>>& result);

/// @brief result={{Re(A+B), Im(A-B)},{-Im(A+B), Re(A-B)}}
void arrayFlatten2(int nA /*A_part dim*/,
                   const std::vector<std::complex<double>>& A,
                   const std::vector<std::complex<double>>& B,
                   std::vector<double>& result);

/// @brief solve full BSE through matrix M = {{Re(A+B), Im(A-B)},{-Im(A+B), Re(A-B)}}
void solve_full(int my_rank,
                std::vector<std::complex<double>> A_part,
                std::vector<std::complex<double>> B_part,
                int nA /*part_dim*/,
                std::vector<double>& ev,
                std::vector<std::complex<double>>& global_v);
/// @brief solve TDA BSE
void solve_tda(int my_rank,
               std::vector<std::complex<double>> A_part,
               int nA /*part_dim*/,
               std::vector<double>& ev,
               std::vector<std::complex<double>>& global_v);

template <typename T>
class BSE_Matrix
{
  public:
    std::vector<T> BSE_A_local;  // local A matrix for BSE
    std::vector<T> BSE_A_global; // global A matrix for BSE, unocc continuous
    Parallel_2D pA;
    /// @brief constructor for BSE_Matrix
    BSE_Matrix() = default;

    void cal_matrix(const int& nspin,
                    const int& naos,
                    const std::vector<int>& nocc,
                    const std::vector<int>& nvirt,
                    const UnitCell& ucell_in,
                    const psi::Psi<T>& psi_ks_in,
                    const ModuleBase::matrix& eig_gw,
                    std::weak_ptr<Exx_LRI<T>> exx_lri_in,
                    const K_Vectors& kv_in,
                    const std::vector<Parallel_2D>& pX_in,
                    const Parallel_2D& pc_in,
                    const Parallel_Orbitals& pmat_in,
                    const std::string& spin_type,
                    const std::string& ri_hartree_benchmark = "none",
                    const std::vector<int>& aims_nbasis = {});

    void add_V();
    void add_W();

    void dm_onebase(const int&, const T*);

  private:
    int nspin = 1;
    std::vector<int> nocc;
    std::vector<int> nvirt;
    int nk = 1;
    const std::vector<Parallel_2D>& pX; // X for tda, (X,Y) for full
    std::unique_ptr<elecstate::DensityMatrix<T, T>> DM_trans;
};
} // namespace BSE