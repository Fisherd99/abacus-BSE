#ifndef FINITE_FIELD_FR_OVERLAP_H
#define FINITE_FIELD_FR_OVERLAP_H
#ifdef __LCAO
#include <complex>
#include <array>
#include <functional>
#include "source_basis/module_ao/orb_read.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_cell/module_neighbor/sltk_grid_driver.h"
#include "source_cell/unitcell.h"
#include "source_hamilt/module_hcontainer/hcontainer.h"
#include "source_base/math_lebedev_laikov.h"
#include "source_basis/module_nao/generalized_overlap_integrator.h"
#include "source_basis/module_nao/radial_collection.h"
#include "first_order_overlap_integrator.h"


// Keep this type distinct from the legacy global FR_overlap used by Wannier90.
// The two implementations have different object layouts; giving them the same
// template name violates the ODR and lets the linker select methods compiled
// for the wrong layout.
template <typename T>
class FiniteFieldFROverlap
{
public:
    using fr_ptr = std::function<T(ModuleBase::Vector3<double>)>;

    FiniteFieldFROverlap();

    void set_parameters(fr_ptr fr_in,
                        const UnitCell* ucell_in,
                        const LCAO_Orbitals* ptr_orb,
                        const Grid_Driver* GridD_in,
                        const Parallel_Orbitals* paraV,
                        int radial_grid_num = 140,
                        int degree = 110,
                        bool calculate_center_gradients = false,
                        bool analytic_center_gradients = false,
                        ModuleBase::Vector3<double> momentum_transfer
                            = ModuleBase::Vector3<double>());

    void set_two_center_parameters(
        const UnitCell* ucell_in,
        const LCAO_Orbitals* ptr_orb,
        const RadialCollection* radial_orbitals,
        const Grid_Driver* GridD_in,
        const Parallel_Orbitals* paraV,
        ModuleBase::Vector3<double> momentum_transfer,
        int plane_wave_lmax = 12,
        int radial_table_num = 0,
        double radial_table_cutoff = 0.0,
        bool calculate_center_gradients = false);

    void set_first_order_parameters(
        const UnitCell* ucell_in,
        const LCAO_Orbitals* ptr_orb,
        const Grid_Driver* GridD_in,
        const Parallel_Orbitals* paraV,
        ModuleBase::Vector3<double> momentum_transfer,
        bool calculate_center_gradients);

    FiniteFieldFROverlap(const FiniteFieldFROverlap<T>& FR_in);

    FiniteFieldFROverlap(FiniteFieldFROverlap<T>&& FR_in);

    ~FiniteFieldFROverlap();

    void calculate_FR();

    void calculate_center_gradient(int atom);

    hamilt::HContainer<T>* get_FR_pointer() const
    {
        return this->FR_container;
    }

    /** Derivative of the integral with respect to one atomic center, in
     * Cartesian bohr.  Available only when center gradients were requested. */
    const hamilt::HContainer<T>* get_dFR_pointer(int atom, int component) const;

protected:
  void initialize_FR(const Grid_Driver* GridD, const Parallel_Orbitals* paraV);

  void initialize_center_gradients(int atom);

  void cal_FR_IJR(const int& iat1,
                  const int& iat2,
                  const Parallel_Orbitals* paraV,
                  const ModuleBase::Vector3<double>& dtau,
                  T* data_pointer,
                  const std::array<T*, 3>& bra_derivative,
                  const std::array<T*, 3>& ket_derivative);

  void cal_FR_IJR_two_center(const int& iat1,
                             const int& iat2,
                             const Parallel_Orbitals* paraV,
                             const ModuleBase::Vector3<double>& dtau,
                             T* data_pointer,
                             const std::array<T*, 3>& bra_derivative,
                             const std::array<T*, 3>& ket_derivative);

  std::map<std::pair<int, int>, double> psi_inter(const int& T1,
                                                  const std::set<std::pair<int, int>>& LN_pair1,
                                                  const double& r_norm);

  double Polynomial_Interpolation(const double* psi_r, const int& mesh_r, const double& dr, const double& x);
  double Polynomial_Interpolation_Derivative(const double* psi_r,
                                             const int& mesh_r,
                                             const double& dr,
                                             const double& x);

  fr_ptr fr = nullptr;
  const UnitCell* ucell = nullptr;
  const LCAO_Orbitals* ptr_orb_ = nullptr;
  int radial_grid_num = 140;
  bool calculate_center_gradients = false;
  bool analytic_center_gradients = false;
  ModuleBase::Vector3<double> momentum_transfer;
  const RadialCollection* radial_orbitals = nullptr;
  std::unique_ptr<GeneralizedOverlapIntegrator> two_center_integrator;
  std::unique_ptr<FirstOrderOverlapIntegrator> first_order_integrator;
  ModuleBase::Lebedev_laikov_grid* Leb_grid = nullptr;
  hamilt::HContainer<T>* FR_container = nullptr;
  std::vector<std::unique_ptr<hamilt::HContainer<T>>> dFR_datom;
  int active_gradient_atom = -1;
};
#endif
#endif
