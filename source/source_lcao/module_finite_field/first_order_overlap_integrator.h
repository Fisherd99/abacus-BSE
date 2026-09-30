#ifndef FINITE_FIELD_FIRST_ORDER_OVERLAP_INTEGRATOR_H
#define FINITE_FIELD_FIRST_ORDER_OVERLAP_INTEGRATOR_H

#include "source_base/vector3.h"
#include "source_base/sph_bessel_recursive.h"
#include "source_basis/module_ao/orb_gaunt_table.h"
#include "source_basis/module_ao/orb_atomic_lm.h"
#include "source_lcao/center2orb_orb11.h"
#include "source_lcao/center2orb_orb21.h"

#include <complex>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

class LCAO_Orbitals;

/** first-order generalized overlap.
 *
 * Computes exp(-i q.R1) [ S(R2-R1) - i q.D(R2-R1) ], where
 * S=<phi1|phi2> and D=<phi1|(r-R1)|phi2>, using the same Orb11/Orb21
 * two-center tables as unk_overlap_lcao.cpp.
 */
class FirstOrderOverlapIntegrator
{
  public:
    using Complex = std::complex<double>;

    FirstOrderOverlapIntegrator(const LCAO_Orbitals& orbitals,
                                const ModuleBase::Vector3<double>& q,
                                bool calculate_gradients);
    ~FirstOrderOverlapIntegrator();

    void calculate(int type_bra, int l_bra, int n_bra, int encoded_m_bra,
                   int type_ket, int l_ket, int n_ket, int encoded_m_ket,
                   const ModuleBase::Vector3<double>& center_bra,
                   const ModuleBase::Vector3<double>& center_ket,
                   Complex* value, Complex* derivative_bra,
                   Complex* derivative_ket) const;

  private:
    using Key = std::tuple<int, int, int, int, int, int>;
    struct Tables
    {
        std::unique_ptr<Center2_Orb::Orb11> overlap;
        std::unique_ptr<Center2_Orb::Orb21> position;
    };

    const Tables& tables(int type_bra, int l_bra, int n_bra,
                         int type_ket, int l_ket, int n_ket) const;

    const LCAO_Orbitals& orbitals_;
    ModuleBase::Vector3<double> q_;
    bool calculate_gradients_;
    ModuleBase::Sph_Bessel_Recursive::D2* psb_ = nullptr;
    ORB_gaunt_table gaunt_;
    Numerical_Orbital_Lm coordinate_orbital_;
    mutable std::map<Key, Tables> tables_;
    mutable std::mutex tables_mutex_;
};

#endif
