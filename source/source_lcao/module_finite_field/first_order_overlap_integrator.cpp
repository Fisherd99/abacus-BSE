#include "first_order_overlap_integrator.h"

#include "source_base/constants.h"
#include "source_base/ylm.h"
#include "source_basis/module_ao/orb_read.h"
#include "source_lcao/center2orb.h"
#include "source_lcao/center2orb_orb11.h"
#include "source_lcao/center2orb_orb21.h"

#include <algorithm>
#include <cmath>

FirstOrderOverlapIntegrator::FirstOrderOverlapIntegrator(
    const LCAO_Orbitals& orbitals, const ModuleBase::Vector3<double>& q,
    const bool calculate_gradients)
    : orbitals_(orbitals), q_(q), calculate_gradients_(calculate_gradients)
{
    int lmax = -1;
    for (int type = 0; type < orbitals.get_ntype(); ++type)
    {
        lmax = std::max(lmax, orbitals.Phi[type].getLmax());
    }
    const double dr = orbitals.get_dR();
    const double dk = orbitals.get_dk();
    // Orb11/Orb21 radial-table construction only indexes the spherical-Bessel
    // table up to Numerical_Orbital_Lm::getNk(), which is get_kmesh() for the
    // orbitals passed here.  The legacy unk overlap allocated 4*kmesh+1 even
    // though those extra columns are never read.  With APNS orbitals that
    // wastes about 830 MB per MPI rank and can exceed a node's memory at 24
    // ranks, without changing any computed table entry.
    const int kmesh = orbitals.get_kmesh();
    int rmesh = static_cast<int>(orbitals.get_Rmax() / dr) + 4;
    rmesh += 1 - rmesh % 2;
    Center2_Orb::init_Table_Spherical_Bessel(2 * lmax + 1, dr, dk,
                                             kmesh, rmesh, psb_);
    ModuleBase::Ylm::set_coefficients();
    gaunt_.init_Gaunt_CH(lmax + 1);
    gaunt_.init_Gaunt(lmax + 1);

    const Numerical_Orbital_Lm& seed = orbitals.Phi[0].PhiLN(0, 0);
    coordinate_orbital_.set_orbital_info(
        seed.getLabel(), 0, 1, 1, seed.getNr(), seed.getRab(),
        seed.getRadial(), Numerical_Orbital_Lm::Psi_Type::Psi,
        seed.getRadial(), seed.getNk(), seed.getDk(), seed.getDruniform(),
        false, false, calculate_gradients_);
}

FirstOrderOverlapIntegrator::~FirstOrderOverlapIntegrator() = default;

const FirstOrderOverlapIntegrator::Tables& FirstOrderOverlapIntegrator::tables(
    const int type_bra, const int l_bra, const int n_bra,
    const int type_ket, const int l_ket, const int n_ket) const
{
    const Key key(type_bra, l_bra, n_bra, type_ket, l_ket, n_ket);
    std::lock_guard<std::mutex> guard(tables_mutex_);
    const auto found = tables_.find(key);
    if (found != tables_.end()) return found->second;

    const Numerical_Orbital_Lm& bra = orbitals_.Phi[type_bra].PhiLN(l_bra, n_bra);
    const Numerical_Orbital_Lm& ket = orbitals_.Phi[type_ket].PhiLN(l_ket, n_ket);
    Tables value;
    value.overlap.reset(new Center2_Orb::Orb11(bra, ket, psb_, gaunt_));
    value.position.reset(
        new Center2_Orb::Orb21(bra, coordinate_orbital_, ket, psb_, gaunt_,
                               false));
    value.overlap->init_radial_table();
    value.position->init_radial_table();
    return tables_.insert(std::make_pair(key, std::move(value))).first->second;
}

void FirstOrderOverlapIntegrator::calculate(
    const int type_bra, const int l_bra, const int n_bra,
    const int encoded_m_bra, const int type_ket, const int l_ket,
    const int n_ket, const int encoded_m_ket,
    const ModuleBase::Vector3<double>& center_bra,
    const ModuleBase::Vector3<double>& center_ket, Complex* value,
    Complex* derivative_bra, Complex* derivative_ket) const
{
    const Tables& table
        = tables(type_bra, l_bra, n_bra, type_ket, l_ket, n_ket);
    const ModuleBase::Vector3<double> origin(0.0, 0.0, 0.0);
    const ModuleBase::Vector3<double> displacement = center_ket - center_bra;
    const double overlap = table.overlap->cal_overlap(
        origin, displacement, encoded_m_bra, encoded_m_ket);
    const double factor = std::sqrt(ModuleBase::FOUR_PI / 3.0);
    ModuleBase::Vector3<double> position(
        -factor * table.position->cal_overlap(
                      origin, displacement, encoded_m_bra, 1, encoded_m_ket),
        -factor * table.position->cal_overlap(
                      origin, displacement, encoded_m_bra, 2, encoded_m_ket),
        factor * table.position->cal_overlap(
                     origin, displacement, encoded_m_bra, 0, encoded_m_ket));
    const Complex local(overlap, -(q_ * position));
    const Complex phase
        = std::exp(-ModuleBase::IMAG_UNIT * (q_ * center_bra));
    if (value) *value = phase * local;

    if (derivative_bra || derivative_ket)
    {
        const ModuleBase::Vector3<double> grad_overlap
            = table.overlap->cal_grad_overlap(
                origin, displacement, encoded_m_bra, encoded_m_ket);
        const ModuleBase::Vector3<double> grad_dx
            = -factor * table.position->cal_grad_overlap(
                            origin, displacement, encoded_m_bra, 1, encoded_m_ket);
        const ModuleBase::Vector3<double> grad_dy
            = -factor * table.position->cal_grad_overlap(
                            origin, displacement, encoded_m_bra, 2, encoded_m_ket);
        const ModuleBase::Vector3<double> grad_dz
            = factor * table.position->cal_grad_overlap(
                           origin, displacement, encoded_m_bra, 0, encoded_m_ket);
        for (int alpha = 0; alpha < 3; ++alpha)
        {
            const Complex grad_local(
                grad_overlap[alpha],
                -(q_.x * grad_dx[alpha] + q_.y * grad_dy[alpha]
                  + q_.z * grad_dz[alpha]));
            if (derivative_ket) derivative_ket[alpha] = phase * grad_local;
            if (derivative_bra)
            {
                derivative_bra[alpha]
                    = phase * (-ModuleBase::IMAG_UNIT * q_[alpha] * local
                               - grad_local);
            }
        }
    }
}
