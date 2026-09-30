#include "source_basis/module_nao/generalized_overlap_integrator.h"

#include "source_base/constants.h"
#include "source_base/cubic_spline.h"
#include "source_base/math_integral.h"
#include "source_base/math_sphbes.h"
#include "source_base/ylm.h"
#include "source_basis/module_nao/real_gaunt_table.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>

namespace
{
bool has_uniform_spacing(const NumericalRadial& radial, const double spacing)
{
    const double tolerance = 1.0e-10 * std::max(1.0, std::abs(spacing));
    for (int ir = 1; ir < radial.nr(); ++ir)
    {
        if (std::abs(radial.rgrid(ir) - radial.rgrid(ir - 1) - spacing)
            > tolerance)
        {
            return false;
        }
    }
    return true;
}

double double_factorial(const int n)
{
    double result = 1.0;
    for (int i = n; i > 1; i -= 2) result *= i;
    return result;
}

std::complex<double> minus_i_power(const int l)
{
    static const std::complex<double> powers[4] = {
        {1.0, 0.0}, {0.0, -1.0}, {-1.0, 0.0}, {0.0, 1.0}};
    return powers[l % 4];
}

}

GeneralizedOverlapIntegrator::GeneralizedOverlapIntegrator(
    const ModuleBase::Vector3<double>& q, const int plane_wave_lmax,
    const int nr, const double cutoff)
    : q_(q), qnorm_(0.0), plane_wave_lmax_(plane_wave_lmax),
      nr_(nr), cutoff_(cutoff), rgrid_(nr),
      cache_(new SharedCache)
{
    if (plane_wave_lmax < 0 || nr < 3 || cutoff <= 0.0)
    {
        throw std::invalid_argument(
            "invalid generalized-overlap integration parameters");
    }
    qnorm_ = q_.norm();
    const double dr = cutoff / static_cast<double>(nr - 1);
    for (int ir = 0; ir < nr; ++ir) rgrid_[ir] = ir * dr;

    const ModuleBase::Vector3<double> uq
        = qnorm_ == 0.0 ? ModuleBase::Vector3<double>(0.0, 0.0, 1.0)
                        : q / qnorm_;
    ModuleBase::Ylm::sph_harm(plane_wave_lmax, uq.x, uq.y, uq.z,
                              q_harmonics_);
}

void GeneralizedOverlapIntegrator::prepare_angular_momenta(
    const int orbital_lmax) const
{
    if (orbital_lmax < 0)
    {
        throw std::invalid_argument("orbital_lmax must be non-negative");
    }
    // Build only the sparse channels that can occur in the orbital basis.
    // Doing this before the parallel matrix assembly also removes first-use
    // contention from the hot path.
    for (int l1 = 0; l1 <= orbital_lmax; ++l1)
    {
        for (int m1 = -l1; m1 <= l1; ++m1)
        {
            for (int l2 = 0; l2 <= orbital_lmax; ++l2)
            {
                for (int m2 = -l2; m2 <= l2; ++m2)
                {
                    angular_channels(l1, m1, l2, m2);
                }
            }
        }
    }
}

const GeneralizedOverlapIntegrator::DressedRadial&
GeneralizedOverlapIntegrator::dressed_radial(
    const NumericalRadial& bra, const NumericalRadial& ket,
    const int plane_wave_l, const int dressed_l) const
{
    const PaddedRadial& ket_work = padded_radial(ket);
    const NumericalRadial* grid_identity
        = ket_work.radial.is_fft_compliant() ? nullptr : &ket;
    const DressedKey key(&bra, plane_wave_l, dressed_l,
                         grid_identity, ket_work.radial.nk(),
                         ket_work.radial.kmax());
    // Misses are deliberately serialized: set_grid may create an FFTW plan,
    // and FFTW's planner is not thread-safe unless configured globally.
    std::lock_guard<std::mutex> guard(cache_->dressed_mutex);
    const std::map<DressedKey, DressedRadial>::const_iterator found
        = cache_->dressed_values.find(key);
    if (found != cache_->dressed_values.end()) return found->second;

    const PaddedRadial& bra_work = padded_radial(bra);
    DressedRadial result;

    std::vector<double> bessel(bra_work.radial.nr());
    ModuleBase::Sphbes::sphbesj(bra_work.radial.nr(),
                                bra_work.radial.rgrid(), qnorm_,
                                plane_wave_l, bessel.data());
    std::vector<double> dressed_values(bra_work.radial.nr());
    for (int ir = 0; ir < bra_work.radial.nr(); ++ir)
    {
        dressed_values[ir] = bra_work.radial.rvalue(ir) * bessel[ir];
    }
    result.dressed.build(
        dressed_l, true, bra_work.radial.nr(), bra_work.radial.rgrid(),
        dressed_values.data(), static_cast<int>(bra_work.radial.pr()),
        bra_work.radial.izeta(), bra_work.radial.symbol(),
        bra_work.radial.itype());
    result.dressed.set_transformer(ModuleBase::SphericalBesselTransformer(), 0);
    result.dressed.set_grid(false, ket_work.radial.nk(),
                            ket_work.radial.kgrid(), 't');

    return cache_->dressed_values
        .insert(std::make_pair(key, std::move(result))).first->second;
}

const GeneralizedOverlapIntegrator::PaddedRadial&
GeneralizedOverlapIntegrator::padded_radial(const NumericalRadial& radial) const
{
    // Padding may also initialize a transform plan.  Cache misses therefore
    // remain serialized while all hits only take this short lookup lock.
    std::lock_guard<std::mutex> guard(cache_->padded_mutex);
    const std::map<const NumericalRadial*, PaddedRadial>::const_iterator found
        = cache_->padded_values.find(&radial);
    if (found != cache_->padded_values.end()) return found->second;
    PaddedRadial result;
    result.radial = radial;
    result.radial.set_transformer(ModuleBase::SphericalBesselTransformer(), 0);
    const double spacing = radial.rgrid(1) - radial.rgrid(0);
    const double padded_rmax = 2.0 * cutoff_;
    if (has_uniform_spacing(radial, spacing) && radial.rmax() < padded_rmax)
    {
        int padded_nr = static_cast<int>(std::ceil(padded_rmax / spacing)) + 1;
        padded_nr += 1 - padded_nr % 2;
        const double exact_rmax = spacing * (padded_nr - 1);
        result.radial.set_uniform_grid(true, padded_nr, exact_rmax, 'i', true);
    }
    return cache_->padded_values
        .insert(std::make_pair(&radial, std::move(result))).first->second;
}

const GeneralizedOverlapIntegrator::RadialTable&
GeneralizedOverlapIntegrator::radial_table(
    const NumericalRadial& bra, const NumericalRadial& ket,
    const int plane_wave_l, const int dressed_l, const int coupling_l,
    const ModuleBase::SphericalBesselTransformer* prepared_transformer) const
{
    const TableKey key(&bra, &ket, plane_wave_l, dressed_l, coupling_l);
    std::unique_lock<std::mutex> miss_guard(cache_->radial_mutex,
                                             std::defer_lock);
    if (prepared_transformer == nullptr)
    {
        // The lazy path can be entered concurrently by clients that skipped
        // prepare_radial_tables().  Serialize a miss so its private FFTW plan
        // is created safely and the same table is not computed repeatedly.
        miss_guard.lock();
    }
    if (miss_guard.owns_lock())
    {
        const std::map<TableKey, RadialTable>::const_iterator found
            = cache_->radial_values.find(key);
        if (found != cache_->radial_values.end()) return found->second;
    }
    else
    {
        std::lock_guard<std::mutex> guard(cache_->radial_mutex);
        const std::map<TableKey, RadialTable>::const_iterator found
            = cache_->radial_values.find(key);
        if (found != cache_->radial_values.end()) return found->second;
    }
    const DressedRadial& dressed = dressed_radial(
        bra, ket, plane_wave_l, dressed_l);
    const PaddedRadial& ket_work = padded_radial(ket);

    RadialTable table;
    table.value.resize(nr_);
    table.derivative.resize(nr_);
    // Give each table construction its own transformer workspace.  The radial
    // samples and k-space values are copied, while different ell transforms
    // can now execute concurrently without sharing FFTW scratch buffers.
    NumericalRadial dressed_work(dressed.dressed);
    const ModuleBase::SphericalBesselTransformer local_transformer;
    const ModuleBase::SphericalBesselTransformer& transformer
        = prepared_transformer ? *prepared_transformer : local_transformer;
    dressed_work.set_transformer(transformer, 0);
    dressed_work.radtab('S', ket_work.radial, coupling_l, table.value.data(), nr_,
                        cutoff_);

    const double dr = cutoff_ / static_cast<double>(nr_ - 1);
    if (coupling_l > 0)
    {
        for (int ir = 1; ir < nr_; ++ir)
        {
            table.value[ir] /= std::pow(rgrid_[ir], coupling_l);
        }
        std::vector<double> integrand(ket_work.radial.nk());
        std::vector<double> spacing(ket_work.radial.nk());
        std::adjacent_difference(ket_work.radial.kgrid(),
                                 ket_work.radial.kgrid() + ket_work.radial.nk(),
                                 spacing.begin());
        for (int ik = 0; ik < ket_work.radial.nk(); ++ik)
        {
            integrand[ik] = dressed_work.kvalue(ik)
                            * ket_work.radial.kvalue(ik)
                            * std::pow(ket_work.radial.kgrid(ik),
                                       coupling_l + 2);
        }
        table.value[0]
            = ModuleBase::Integral::simpson(ket_work.radial.nk(),
                                             integrand.data(),
                                             spacing.data() + 1)
              * ModuleBase::FOUR_PI
              / double_factorial(2 * coupling_l + 1);
    }
    ModuleBase::CubicSpline::build(
        nr_, rgrid_.data(), table.value.data(),
        {ModuleBase::CubicSpline::BoundaryType::first_deriv, 0.0},
        {ModuleBase::CubicSpline::BoundaryType::first_deriv, 0.0},
        table.derivative.data());
    if (miss_guard.owns_lock())
    {
        return cache_->radial_values
            .insert(std::make_pair(key, std::move(table))).first->second;
    }
    std::lock_guard<std::mutex> insert_guard(cache_->radial_mutex);
    return cache_->radial_values
        .insert(std::make_pair(key, std::move(table))).first->second;
}

void GeneralizedOverlapIntegrator::prepare_radial_tables(
    const std::vector<std::pair<const NumericalRadial*,
                                const NumericalRadial*> >& orbital_pairs) const
{
    std::map<DressedKey, const NumericalRadial*> dressed_kets;
    std::set<TableKey> table_keys;
    for (std::size_t ipair = 0; ipair < orbital_pairs.size(); ++ipair)
    {
        const NumericalRadial& bra = *orbital_pairs[ipair].first;
        const NumericalRadial& ket = *orbital_pairs[ipair].second;
        for (int m1 = -bra.l(); m1 <= bra.l(); ++m1)
        {
            for (int m2 = -ket.l(); m2 <= ket.l(); ++m2)
            {
                const std::vector<AngularChannel>& channels
                    = angular_channels(bra.l(), m1, ket.l(), m2);
                for (std::size_t ichannel = 0; ichannel < channels.size();
                     ++ichannel)
                {
                    const AngularChannel& channel = channels[ichannel];
                    const PaddedRadial& ket_work = padded_radial(ket);
                    const NumericalRadial* grid_identity
                        = ket_work.radial.is_fft_compliant() ? nullptr : &ket;
                    dressed_kets.insert(std::make_pair(
                        std::make_tuple(&bra, channel.plane_wave_l,
                                        channel.dressed_l, grid_identity,
                                        ket_work.radial.nk(),
                                        ket_work.radial.kmax()),
                        &ket));
                    table_keys.insert(std::make_tuple(
                        &bra, &ket, channel.plane_wave_l, channel.dressed_l,
                        channel.coupling_l));
                }
            }
        }
    }

    // Constructing a dressed orbital creates an FFT plan.  Keep that stage
    // deterministic and outside the matrix-assembly OpenMP region.
    for (std::map<DressedKey, const NumericalRadial*>::const_iterator it
             = dressed_kets.begin();
         it != dressed_kets.end(); ++it)
    {
        dressed_radial(*std::get<0>(it->first), *it->second,
                       std::get<1>(it->first), std::get<2>(it->first));
    }

    std::vector<TableKey> keys(table_keys.begin(), table_keys.end());
    // FFTW plan creation is not thread-safe by default.  Prepare one private
    // transformer per table serially; the following parallel region then only
    // executes already planned transforms and never shares scratch storage.
    std::vector<ModuleBase::SphericalBesselTransformer> transformers;
    transformers.reserve(keys.size());
    for (std::size_t ikey = 0; ikey < keys.size(); ++ikey)
    {
        transformers.emplace_back();
        const PaddedRadial& ket_work
            = padded_radial(*std::get<1>(keys[ikey]));
        std::vector<double> zeros(ket_work.radial.nk(), 0.0);
        std::vector<double> transformed(ket_work.radial.nk());
        transformers.back().radrfft(
            0, ket_work.radial.nk(), ket_work.radial.kmax(), zeros.data(),
            transformed.data());
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int ikey = 0; ikey < static_cast<int>(keys.size()); ++ikey)
    {
        const TableKey& key = keys[ikey];
        radial_table(*std::get<0>(key), *std::get<1>(key), std::get<2>(key),
                     std::get<3>(key), std::get<4>(key), &transformers[ikey]);
    }
}

const std::vector<GeneralizedOverlapIntegrator::AngularChannel>&
GeneralizedOverlapIntegrator::angular_channels(
    const int bra_l, const int bra_m, const int ket_l, const int ket_m) const
{
    const AngularKey key(bra_l, bra_m, ket_l, ket_m);
    std::lock_guard<std::mutex> guard(cache_->angular_mutex);
    const auto found = cache_->angular_values.find(key);
    if (found != cache_->angular_values.end()) return found->second;

    std::vector<AngularChannel> channels;
    std::map<std::tuple<int, int, int>, std::size_t> channel_index;
    for (int L = 0; L <= plane_wave_lmax_; ++L)
    {
        const Complex rayleigh = ModuleBase::FOUR_PI * minus_i_power(L);
        for (int M = -L; M <= L; ++M)
        {
            const double qylm = q_harmonics_[ylm_index(L, M)];
            if (std::abs(qylm) < 1.0e-15) continue;
            for (int lambda = std::abs(bra_l - L);
                 lambda <= bra_l + L; ++lambda)
            {
                for (int mu = -lambda; mu <= lambda; ++mu)
                {
                    const double product_gaunt
                        = RealGauntTable::instance().evaluate(
                            bra_l, L, lambda, bra_m, M, mu);
                    if (std::abs(product_gaunt) < 1.0e-15) continue;
                    for (int ell = std::abs(lambda - ket_l);
                         ell <= lambda + ket_l; ell += 2)
                    {
                        const int phase_power = (ell - lambda + ket_l) / 2;
                        const double sign = phase_power % 2 == 0 ? 1.0 : -1.0;
                        const auto radial_key = std::make_tuple(L, lambda, ell);
                        for (int m = -ell; m <= ell; ++m)
                        {
                            const double coupling_gaunt
                                = RealGauntTable::instance().evaluate(
                                    lambda, ket_l, ell, mu, ket_m, m);
                            if (std::abs(coupling_gaunt) < 1.0e-15) continue;
                            auto inserted = channel_index.emplace(
                                radial_key, channels.size());
                            if (inserted.second)
                            {
                                AngularChannel channel;
                                channel.plane_wave_l = L;
                                channel.dressed_l = lambda;
                                channel.coupling_l = ell;
                                channels.push_back(std::move(channel));
                            }
                            channels[inserted.first->second].terms.push_back(
                                {m, rayleigh * qylm * product_gaunt * sign
                                        * coupling_gaunt});
                        }
                    }
                }
            }
        }
    }
    return cache_->angular_values
        .insert(std::make_pair(key, std::move(channels))).first->second;
}

void GeneralizedOverlapIntegrator::radial_lookup(
    const RadialTable& table, const int, const double distance,
    double* value_over_rl, double* derivative_over_rl) const
{
    const double dr = cutoff_ / static_cast<double>(nr_ - 1);
    int index = static_cast<int>(distance / dr);
    if (index >= nr_ - 1) index = nr_ - 2;
    const double offset = distance - index * dr;
    const double inverse_dr = 1.0 / dr;
    const double secant
        = (table.value[index + 1] - table.value[index]) * inverse_dr;
    const double c0 = table.value[index];
    const double c1 = table.derivative[index];
    const double c3 = (c1 + table.derivative[index + 1] - 2.0 * secant)
                      * inverse_dr * inverse_dr;
    const double c2 = (secant - c1) * inverse_dr - c3 * dr;
    if (value_over_rl)
    {
        *value_over_rl = ((c3 * offset + c2) * offset + c1) * offset + c0;
    }
    if (derivative_over_rl)
    {
        *derivative_over_rl
            = (3.0 * c3 * offset + 2.0 * c2) * offset + c1;
    }
}

int GeneralizedOverlapIntegrator::ylm_index(const int l, const int m) const
{
    return l * l + (m > 0 ? 2 * m - 1 : -2 * m);
}

void GeneralizedOverlapIntegrator::calculate(
    const NumericalRadial& bra, const int m_bra,
    const NumericalRadial& ket, const int m_ket,
    const ModuleBase::Vector3<double>& center_bra,
    const ModuleBase::Vector3<double>& center_ket, Complex* value,
    Complex* derivative_bra, Complex* derivative_ket) const
{
    GeometryWorkspace workspace;
    const int orbital_lmax = std::max(bra.l(), ket.l());
    const bool calculate_gradients = derivative_bra || derivative_ket;
    prepare_geometry(center_bra, center_ket, orbital_lmax,
                     calculate_gradients, workspace);
    calculate(bra, m_bra, ket, m_ket, workspace, value, derivative_bra,
              derivative_ket);
}

void GeneralizedOverlapIntegrator::prepare_geometry(
    const ModuleBase::Vector3<double>& center_bra,
    const ModuleBase::Vector3<double>& center_ket, const int orbital_lmax,
    const bool calculate_gradients, GeometryWorkspace& workspace) const
{
    workspace.center_bra = center_bra;
    workspace.center_ket = center_ket;
    workspace.displacement = center_ket - center_bra;
    workspace.distance = workspace.displacement.norm();
    workspace.unit_displacement
        = workspace.distance == 0.0
              ? ModuleBase::Vector3<double>(0.0, 0.0, 1.0)
              : workspace.displacement / workspace.distance;
    workspace.phase = std::exp(
        -ModuleBase::IMAG_UNIT * (q_ * workspace.center_bra));
    const int angular_max = 2 * orbital_lmax + plane_wave_lmax_;
    ModuleBase::Ylm::rl_sph_harm(
        angular_max, workspace.displacement.x, workspace.displacement.y,
        workspace.displacement.z, workspace.solid_harmonics);
    if (calculate_gradients)
    {
        workspace.solid_harmonic_gradients.resize(
            3 * workspace.solid_harmonics.size());
        ModuleBase::Ylm::grad_rl_sph_harm(
            angular_max, workspace.displacement.x, workspace.displacement.y,
            workspace.displacement.z, workspace.solid_harmonics.data(),
            workspace.solid_harmonic_gradients.data());
    }
    else
    {
        workspace.solid_harmonic_gradients.clear();
    }
}

void GeneralizedOverlapIntegrator::calculate(
    const NumericalRadial& bra, const int m_bra,
    const NumericalRadial& ket, const int m_ket,
    const GeometryWorkspace& workspace, Complex* value,
    Complex* derivative_bra, Complex* derivative_ket) const
{
    if (!value && !derivative_bra && !derivative_ket) return;
    if (std::abs(m_bra) > bra.l() || std::abs(m_ket) > ket.l())
    {
        throw std::invalid_argument("invalid magnetic quantum number");
    }
    Complex result(0.0, 0.0);
    Complex grad_r[3] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    const bool need_gradient = derivative_bra || derivative_ket;
    if (need_gradient && workspace.solid_harmonic_gradients.empty())
    {
        throw std::invalid_argument(
            "geometry workspace does not contain gradients");
    }
    // Both NAOs are compactly supported.  A finite k-grid spherical-Bessel
    // transform can otherwise leave a small ringing tail in the tabulated
    // dressed channel beyond the exact overlap domain.
    const double support_cutoff = bra.rcut() + ket.rcut();
    if (workspace.distance < std::min(cutoff_, support_cutoff))
    {
        const std::vector<AngularChannel>& channels
            = angular_channels(bra.l(), m_bra, ket.l(), m_ket);
        for (const AngularChannel& channel : channels)
        {
            const RadialTable& table = radial_table(
                bra, ket, channel.plane_wave_l, channel.dressed_l,
                channel.coupling_l, nullptr);
            double radial = 0.0;
            double radial_derivative = 0.0;
            double* radial_derivative_pointer
                = need_gradient ? &radial_derivative : nullptr;
            radial_lookup(table, channel.coupling_l, workspace.distance,
                          &radial, radial_derivative_pointer);
            Complex angular(0.0, 0.0);
            Complex angular_gradient[3]
                = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
            for (const AngularTerm& term : channel.terms)
            {
                const int lm = ylm_index(channel.coupling_l, term.m);
                angular += term.coefficient * workspace.solid_harmonics[lm];
                if (need_gradient)
                {
                    for (int alpha = 0; alpha < 3; ++alpha)
                    {
                        angular_gradient[alpha]
                            += term.coefficient
                               * workspace.solid_harmonic_gradients[3 * lm
                                                                    + alpha];
                    }
                }
            }
            result += radial * angular;
            if (need_gradient)
            {
                for (int alpha = 0; alpha < 3; ++alpha)
                {
                    grad_r[alpha]
                        += radial_derivative
                               * workspace.unit_displacement[alpha] * angular
                           + radial * angular_gradient[alpha];
                }
            }
        }
    }

    if (value) *value = workspace.phase * result;
    for (int alpha = 0; alpha < 3; ++alpha)
    {
        const Complex ket_gradient = workspace.phase * grad_r[alpha];
        if (derivative_ket) derivative_ket[alpha] = ket_gradient;
        if (derivative_bra)
        {
            derivative_bra[alpha]
                = workspace.phase
                  * (-ModuleBase::IMAG_UNIT * q_[alpha] * result
                     - grad_r[alpha]);
        }
    }
}
