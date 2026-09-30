#ifndef GENERALIZED_OVERLAP_INTEGRATOR_H_
#define GENERALIZED_OVERLAP_INTEGRATOR_H_

#include "source_base/vector3.h"
#include "source_basis/module_nao/numerical_radial.h"

#include <complex>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

/**
 * Two-center matrix element of a plane-wave translation operator.
 *
 * Computes
 *   < phi_1(r-R1) | exp(-i q.r) | phi_2(r-R2) >
 * and, optionally, its derivatives with respect to both centers.  The
 * three-dimensional integral is reduced analytically with a Rayleigh
 * expansion and real Gaunt coefficients.  Only one-dimensional spherical
 * Bessel transforms are tabulated numerically.
 *
 * The Rayleigh expansion is truncated at plane_wave_lmax.  Convergence in
 * that parameter is therefore part of the numerical contract of this class.
 * Input radial objects must remain alive and unchanged while an integrator is
 * used because radial tables are cached by object identity.
 */
class GeneralizedOverlapIntegrator
{
  public:
    using Complex = std::complex<double>;

    /** Geometry-dependent quantities shared by every orbital pair belonging
     * to the same pair of centers. */
    struct GeometryWorkspace
    {
        ModuleBase::Vector3<double> center_bra;
        ModuleBase::Vector3<double> center_ket;
        ModuleBase::Vector3<double> displacement;
        ModuleBase::Vector3<double> unit_displacement;
        double distance = 0.0;
        Complex phase;
        std::vector<double> solid_harmonics;
        std::vector<double> solid_harmonic_gradients;
    };

    /**
     * Construct an integrator for one fixed momentum transfer and radial
     * table discretization.  The object may be reused for any number of
     * orbital and center pairs compatible with those settings.
     */
    GeneralizedOverlapIntegrator(const ModuleBase::Vector3<double>& q,
                                 int plane_wave_lmax,
                                 int nr,
                                 double cutoff);
    GeneralizedOverlapIntegrator(const GeneralizedOverlapIntegrator&) = default;

    /** Calculate one matrix element, preparing its geometry on demand. */
    void calculate(const NumericalRadial& bra,
                   int m_bra,
                   const NumericalRadial& ket,
                   int m_ket,
                   const ModuleBase::Vector3<double>& center_bra,
                   const ModuleBase::Vector3<double>& center_ket,
                   Complex* value,
                   Complex* derivative_bra = nullptr,
                   Complex* derivative_ket = nullptr) const;

    /**
     * Prepare center-dependent data shared by orbital pairs on the same two
     * atoms.  Set calculate_gradients when either center derivative will be
     * requested from the workspace overload of calculate().
     */
    void prepare_geometry(const ModuleBase::Vector3<double>& center_bra,
                          const ModuleBase::Vector3<double>& center_ket,
                          int orbital_lmax,
                          bool calculate_gradients,
                          GeometryWorkspace& workspace) const;

    /**
     * Calculate with a reusable geometry workspace.  Each output pointer is
     * optional.  derivative_bra and derivative_ket, when supplied, point to
     * arrays of three Cartesian components.
     */
    void calculate(const NumericalRadial& bra,
                   int m_bra,
                   const NumericalRadial& ket,
                   int m_ket,
                   const GeometryWorkspace& workspace,
                   Complex* value,
                   Complex* derivative_bra,
                   Complex* derivative_ket) const;

    int plane_wave_lmax() const { return plane_wave_lmax_; }
    const ModuleBase::Vector3<double>& momentum_transfer() const { return q_; }
    int radial_table_size() const { return nr_; }
    double radial_table_cutoff() const { return cutoff_; }
    /// Pre-build sparse angular channels before entering threaded evaluation.
    void prepare_angular_momenta(int orbital_lmax) const;

    /**
     * Pre-build every radial table required by the supplied orbital pairs.
     * Pointers must be non-null and the pointed-to NumericalRadial objects
     * must remain alive and unchanged for the lifetime of this integrator and
     * every copy sharing its cache.
     */
    void prepare_radial_tables(
        const std::vector<std::pair<const NumericalRadial*,
                                    const NumericalRadial*> >& orbital_pairs) const;

  private:
    struct AngularTerm
    {
        int m;
        Complex coefficient;
    };

    struct AngularChannel
    {
        int plane_wave_l;
        int dressed_l;
        int coupling_l;
        std::vector<AngularTerm> terms;
    };

    struct RadialTable
    {
        std::vector<double> value;
        std::vector<double> derivative;
    };

    struct DressedRadial
    {
        NumericalRadial dressed;
    };

    struct PaddedRadial
    {
        NumericalRadial radial;
    };

    typedef std::tuple<const NumericalRadial*, const NumericalRadial*, int, int, int> TableKey;
    // FFT-compliant grids are uniquely described by (nk, kmax).  For a
    // nonuniform grid retain the ket identity as well, so two grids with the
    // same endpoints can never alias in the dressed-orbital cache.
    typedef std::tuple<const NumericalRadial*, int, int,
                       const NumericalRadial*, int, double> DressedKey;
    typedef std::tuple<int, int, int, int> AngularKey;

    const RadialTable& radial_table(const NumericalRadial& bra,
                                    const NumericalRadial& ket,
                                    int plane_wave_l,
                                    int dressed_l,
                                    int coupling_l,
                                    const ModuleBase::SphericalBesselTransformer*
                                        prepared_transformer) const;

    const DressedRadial& dressed_radial(const NumericalRadial& bra,
                                        const NumericalRadial& ket,
                                        int plane_wave_l,
                                        int dressed_l) const;

    const PaddedRadial& padded_radial(const NumericalRadial& radial) const;

    const std::vector<AngularChannel>& angular_channels(int bra_l,
                                                        int bra_m,
                                                        int ket_l,
                                                        int ket_m) const;

    void radial_lookup(const RadialTable& table,
                       int coupling_l,
                       double distance,
                       double* value_over_rl,
                       double* derivative_over_rl) const;

    int ylm_index(int l, int m) const;

    ModuleBase::Vector3<double> q_;
    double qnorm_;
    int plane_wave_lmax_;
    int nr_;
    double cutoff_;
    std::vector<double> rgrid_;
    std::vector<double> q_harmonics_;
    struct SharedCache
    {
        std::map<TableKey, RadialTable> radial_values;
        std::map<DressedKey, DressedRadial> dressed_values;
        std::map<const NumericalRadial*, PaddedRadial> padded_values;
        std::map<AngularKey, std::vector<AngularChannel>> angular_values;
        std::mutex radial_mutex;
        std::mutex dressed_mutex;
        std::mutex padded_mutex;
        std::mutex angular_mutex;
    };
    std::shared_ptr<SharedCache> cache_;
};

#endif
