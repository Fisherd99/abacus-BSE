#include "source_basis/module_nao/two_center_integrator.h"
#include "source_basis/module_nao/generalized_overlap_integrator.h"

#include "source_base/constants.h"
#include "source_base/cubic_spline.h"
#include "source_base/math_integral.h"
#include "source_base/math_lebedev_laikov.h"
#include "source_base/math_sphbes.h"
#include "source_base/sph_bessel_tf.h"
#include "source_base/vector3.h"
#include "source_base/ylm.h"

#include "gtest/gtest.h"
#include <chrono>
#include <cstring>
using iclock = std::chrono::high_resolution_clock;
using ModuleBase::Sphbes;

namespace
{
std::complex<double> direct_generalized_overlap(
    const NumericalRadial& bra, const int m_bra,
    const NumericalRadial& ket, const int m_ket,
    const ModuleBase::Vector3<double>& center_bra,
    const ModuleBase::Vector3<double>& center_ket,
    const ModuleBase::Vector3<double>& q)
{
    constexpr int radial_points = 180;
    constexpr int angular_points = 770;
    std::vector<double> radius(radial_points);
    std::vector<double> radial_weight(radial_points);
    ModuleBase::Integral::Gauss_Legendre_grid_and_weight(
        0.0, bra.rcut(), radial_points, radius.data(), radial_weight.data());

    std::vector<double> bra_spline(bra.nr());
    std::vector<double> ket_spline(ket.nr());
    ModuleBase::CubicSpline::build(
        bra.nr(), bra.rgrid(), bra.rvalue(),
        {ModuleBase::CubicSpline::BoundaryType::not_a_knot},
        {ModuleBase::CubicSpline::BoundaryType::not_a_knot},
        bra_spline.data());
    ModuleBase::CubicSpline::build(
        ket.nr(), ket.rgrid(), ket.rvalue(),
        {ModuleBase::CubicSpline::BoundaryType::not_a_knot},
        {ModuleBase::CubicSpline::BoundaryType::not_a_knot},
        ket_spline.data());

    ModuleBase::Lebedev_laikov_grid angular(angular_points);
    angular.generate_grid_points();
    const ModuleBase::Vector3<double> displacement = center_ket - center_bra;
    std::complex<double> result(0.0, 0.0);
    for (int ir = 0; ir < radial_points; ++ir)
    {
        double bra_value = 0.0;
        ModuleBase::CubicSpline::eval(bra.nr(), bra.rgrid(), bra.rvalue(),
                                      bra_spline.data(), 1, &radius[ir],
                                      &bra_value);
        for (int ia = 0; ia < angular_points; ++ia)
        {
            const ModuleBase::Vector3<double> direction
                = angular.get_grid_coor()[ia];
            const ModuleBase::Vector3<double> relative_bra
                = radius[ir] * direction;
            const ModuleBase::Vector3<double> relative_ket
                = relative_bra - displacement;
            const double ket_radius = relative_ket.norm();
            if (ket_radius >= ket.rcut()) continue;
            double ket_value = 0.0;
            ModuleBase::CubicSpline::eval(ket.nr(), ket.rgrid(), ket.rvalue(),
                                          ket_spline.data(), 1, &ket_radius,
                                          &ket_value);
            std::vector<double> bra_harmonics;
            std::vector<double> ket_harmonics;
            ModuleBase::Ylm::sph_harm(bra.l(), direction.x, direction.y,
                                      direction.z, bra_harmonics);
            ModuleBase::Vector3<double> ket_direction(0.0, 0.0, 1.0);
            if (ket_radius > 1.0e-14) ket_direction = relative_ket / ket_radius;
            ModuleBase::Ylm::sph_harm(ket.l(), ket_direction.x,
                                      ket_direction.y, ket_direction.z,
                                      ket_harmonics);
            const int bra_lm = bra.l() * bra.l()
                               + (m_bra > 0 ? 2 * m_bra - 1 : -2 * m_bra);
            const int ket_lm = ket.l() * ket.l()
                               + (m_ket > 0 ? 2 * m_ket - 1 : -2 * m_ket);
            const ModuleBase::Vector3<double> position
                = center_bra + relative_bra;
            const std::complex<double> phase
                = std::exp(-ModuleBase::IMAG_UNIT * (q * position));
            result += radial_weight[ir] * radius[ir] * radius[ir]
                      * angular.get_weight()[ia] * bra_value * ket_value
                      * bra_harmonics[bra_lm] * ket_harmonics[ket_lm] * phase;
        }
    }
    return result;
}
}

#ifdef __MPI
#include <mpi.h>
#endif

/***********************************************************
 *      Unit test of class "TwoCenterIntegrator"
 ***********************************************************/
/*!
 *  Tested functions:
 *
 *  - build
 *      - builds an object for doing a specific two-center integral
 *                                                                      */
class TwoCenterIntegratorTest : public ::testing::Test
{
  protected:
    void SetUp();
    void TearDown();

    TwoCenterIntegrator S_intor;
    TwoCenterIntegrator T_intor;

    RadialCollection orb;
    int nfile = 0;                                                   //! number of orbital files
    std::string* file = nullptr;                                     //!< orbital files to read from
    std::string log_file = "./test_files/two_center_integrator.log"; //!< file for logging

    double elem_tol = 1e-6; //! tolerance for comparison between new and legacy matrix elements
};

void TwoCenterIntegratorTest::SetUp()
{
#ifdef __MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &GlobalV::MY_RANK);
#endif

    std::string dir = "../../../../../tests/PP_ORB/";
    nfile = 8;
    file = new std::string[nfile];
    file[0] = dir + "C_gga_8au_100Ry_2s2p1d.orb";
    file[1] = dir + "Fe_gga_9au_100Ry_4s2p2d1f.orb";
    file[2] = dir + "O_gga_10au_100Ry_2s2p1d.orb";
    file[3] = dir + "H_gga_8au_60Ry_2s1p.orb";
    file[4] = dir + "Cs_gga_10au_100Ry_4s2p1d.orb";
    file[5] = dir + "Pb_gga_7au_100Ry_2s2p2d1f.orb";
    file[6] = dir + "F_gga_7au_100Ry_2s2p1d.orb";
    file[7] = dir + "I_gga_7au_100Ry_2s2p2d1f.orb";

    ModuleBase::Ylm::set_coefficients();
}

void TwoCenterIntegratorTest::TearDown()
{
    delete[] file;
}

TEST_F(TwoCenterIntegratorTest, FiniteDifference)
{
    nfile = 3;
    orb.build(nfile, file, 'o');

    ModuleBase::SphericalBesselTransformer sbt;
    orb.set_transformer(sbt);

    double rmax = orb.rcut_max() * 2.0;
    double dr = 0.01;
    int nr = static_cast<int>(rmax / dr) + 1;

    // ModuleBase::SphericalBesselTransformer sbt;
    // sbt.set_fftw_plan_flag(FFTW_MEASURE); // not necessarily worth it!
    // orb.set_transformer(&sbt, 0);
    orb.set_uniform_grid(true, nr, rmax, 'i', true);

    iclock::time_point start;
    std::chrono::duration<double> dur;

    start = iclock::now();

    S_intor.tabulate(orb, orb, 'S', nr, rmax);
    T_intor.tabulate(orb, orb, 'T', nr, rmax);

    dur = iclock::now() - start;
    std::cout << "time elapsed = " << dur.count() << " s" << std::endl;

    // check whether analytical derivative and finite difference agree
    int ntype = nfile;
    double tol_d = 1e-4;

    ModuleBase::Vector3<double> vR0 = {1.0, 2.0, 3.0};
    ModuleBase::Vector3<double> vR;

    for (int t1 = 0; t1 < ntype; t1++)
    {
        for (int l1 = 0; l1 <= orb(t1).lmax(); l1++)
        {
            for (int izeta1 = 0; izeta1 < orb(t1).nzeta(l1); izeta1++)
            {
                for (int m1 = -l1; m1 <= l1; ++m1)
                {
                    for (int t2 = t1; t2 < ntype; t2++)
                    {
                        for (int l2 = 0; l2 <= orb(t2).lmax(); l2++)
                        {
                            for (int izeta2 = 0; izeta2 < orb(t2).nzeta(l2); izeta2++)
                            {
                                for (int m2 = -l2; m2 <= l2; ++m2)
                                {
                                    double dx = 1e-4;
                                    double elem_p;
                                    double elem_m;
                                    double grad_elem[3];

                                    // S
                                    vR = vR0;
                                    vR[2] += dx;
                                    S_intor.calculate(t1, l1, izeta1, m1, t2, l2, izeta2, m2, vR, &elem_p);

                                    vR = vR0;
                                    vR[2] -= dx;
                                    S_intor.calculate(t1, l1, izeta1, m1, t2, l2, izeta2, m2, vR, &elem_m);

                                    S_intor.calculate(t1, l1, izeta1, m1, t2, l2, izeta2, m2, vR, nullptr, grad_elem);

                                    EXPECT_NEAR((elem_p - elem_m) / (2. * dx), grad_elem[2], tol_d);

                                    // T
                                    vR = vR0;
                                    vR[2] += dx;
                                    T_intor.calculate(t1, l1, izeta1, m1, t2, l2, izeta2, m2, vR, &elem_p);

                                    vR = vR0;
                                    vR[2] -= dx;
                                    T_intor.calculate(t1, l1, izeta1, m1, t2, l2, izeta2, m2, vR, &elem_m);

                                    T_intor.calculate(t1, l1, izeta1, m1, t2, l2, izeta2, m2, vR, nullptr, grad_elem);

                                    EXPECT_NEAR((elem_p - elem_m) / (2. * dx), grad_elem[2], tol_d);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_F(TwoCenterIntegratorTest, SphericalBessel)
{
    int lmax = 3;
    int nbes = 5;
    int rcut = 7.0;
    double sigma = 0.0;
    double dr = 0.005;
    // The truncated spherical Bessel function has discontinuous first and
    // second derivative at the cutoff, so a small "dr" is required in order
    // to achieve sufficient accuracy.
    //
    // for dr = 0.01, the error of kinetic matrix element is about 1.5e-3
    // for dr = 0.001, the error of kinetic matrix element is about 1.5e-4

    orb.build(lmax, nbes, rcut, sigma, dr);

    ModuleBase::SphericalBesselTransformer sbt;
    orb.set_transformer(sbt);

    double rmax = orb.rcut_max() * 2.0;
    int nr = static_cast<int>(rmax / dr) + 1;

    orb.set_uniform_grid(true, nr, rmax, 'i', true);

    S_intor.tabulate(orb, orb, 'S', nr, rmax);
    T_intor.tabulate(orb, orb, 'T', nr, rmax);

    ModuleBase::Vector3<double> R0 = {0.0, 0.0, 0.0};

    // zeros of spherical bessel functions
    double* zeros = new double[nbes * (lmax + 1)];
    Sphbes::sphbes_zeros(lmax, nbes, zeros, true);

    // checks the diagonal elements with analytical expression
    double elem, ref;
    for (int l = 0; l <= lmax; ++l)
    {
        for (int zeta = 0; zeta < nbes; ++zeta)
        {
            S_intor.calculate(0, l, zeta, 0, 0, l, zeta, 0, R0, &elem);
            ref = 0.5 * std::pow(rcut, 3) * std::pow(Sphbes::sphbesj(l + 1, zeros[l * nbes + zeta]), 2);
            EXPECT_NEAR(elem, ref, 1e-5);

            T_intor.calculate(0, l, zeta, 0, 0, l, zeta, 0, R0, &elem);
            ref = 0.5 * rcut * std::pow(zeros[l * nbes + zeta] * Sphbes::sphbesj(l + 1, zeros[l * nbes + zeta]), 2);
            EXPECT_NEAR(elem, ref, 1e-3);

            // orthogonality
            for (int zeta2 = 0; zeta2 < zeta; ++zeta2)
            {
                S_intor.calculate(0, l, zeta, 0, 0, l, zeta2, 0, R0, &elem);
                ref = 0.0;
                EXPECT_NEAR(elem, ref, 1e-5);
            }
        }
    }
    delete[] zeros;
}

TEST_F(TwoCenterIntegratorTest, GeneralizedOverlapZeroMomentum)
{
    nfile = 1;
    orb.build(nfile, file, 'o');
    ModuleBase::SphericalBesselTransformer sbt;
    orb.set_transformer(sbt);
    const double rmax = 2.0 * orb.rcut_max();
    const double dr = 0.01;
    const int nr = static_cast<int>(rmax / dr) + 1;
    orb.set_uniform_grid(true, nr, rmax, 'i', true);
    S_intor.tabulate(orb, orb, 'S', nr, rmax);

    GeneralizedOverlapIntegrator generalized(
        ModuleBase::Vector3<double>(0.0, 0.0, 0.0), 0, nr, rmax);
    generalized.prepare_angular_momenta(orb.lmax());
    const ModuleBase::Vector3<double> origin(0.0, 0.0, 0.0);
    const ModuleBase::Vector3<double> center(1.0, -0.7, 0.4);
    const NumericalRadial& s = orb(0, 0, 0);
    const NumericalRadial& p = orb(0, 1, 0);

    for (int l1 = 0; l1 <= 1; ++l1)
    {
        const NumericalRadial& bra = l1 == 0 ? s : p;
        for (int m1 = -l1; m1 <= l1; ++m1)
        {
            for (int l2 = 0; l2 <= 1; ++l2)
            {
                const NumericalRadial& ket = l2 == 0 ? s : p;
                for (int m2 = -l2; m2 <= l2; ++m2)
                {
                    double reference = 0.0;
                    double reference_gradient[3];
                    S_intor.calculate(0, l1, 0, m1, 0, l2, 0, m2,
                                      center, &reference, reference_gradient);
                    std::complex<double> value;
                    std::complex<double> grad_bra[3];
                    std::complex<double> grad_ket[3];
                    generalized.calculate(bra, m1, ket, m2, origin, center,
                                          &value, grad_bra, grad_ket);
                    EXPECT_NEAR(value.real(), reference, 2.0e-5);
                    EXPECT_NEAR(value.imag(), 0.0, 1.0e-12);
                    for (int alpha = 0; alpha < 3; ++alpha)
                    {
                        EXPECT_NEAR(grad_ket[alpha].real(),
                                    reference_gradient[alpha], 2.0e-4);
                        EXPECT_NEAR(grad_bra[alpha].real(),
                                    -reference_gradient[alpha], 2.0e-4);
                    }
                }
            }
        }
    }
}

TEST_F(TwoCenterIntegratorTest, GeneralizedOverlapCenterDerivatives)
{
    nfile = 1;
    orb.build(nfile, file, 'o');
    ModuleBase::SphericalBesselTransformer sbt;
    orb.set_transformer(sbt);
    const double rmax = 2.0 * orb.rcut_max();
    const double dr = 0.01;
    const int nr = static_cast<int>(rmax / dr) + 1;
    orb.set_uniform_grid(true, nr, rmax, 'i', true);

    const ModuleBase::Vector3<double> q(0.17, -0.11, 0.09);
    GeneralizedOverlapIntegrator generalized(q, 8, nr, rmax);
    generalized.prepare_angular_momenta(orb.lmax());
    ModuleBase::Vector3<double> center_bra(0.2, -0.1, 0.3);
    ModuleBase::Vector3<double> center_ket(1.1, 0.6, -0.4);
    const NumericalRadial& bra = orb(0, 1, 0);
    const NumericalRadial& ket = orb(0, 1, 0);
    std::complex<double> value;
    std::complex<double> grad_bra[3];
    std::complex<double> grad_ket[3];
    generalized.calculate(bra, 1, ket, -1, center_bra, center_ket,
                          &value, grad_bra, grad_ket);

    const double h = 1.0e-4;
    for (int alpha = 0; alpha < 3; ++alpha)
    {
        ModuleBase::Vector3<double> plus = center_ket;
        ModuleBase::Vector3<double> minus = center_ket;
        plus[alpha] += h;
        minus[alpha] -= h;
        std::complex<double> value_plus, value_minus;
        generalized.calculate(bra, 1, ket, -1, center_bra, plus,
                              &value_plus);
        generalized.calculate(bra, 1, ket, -1, center_bra, minus,
                              &value_minus);
        const std::complex<double> finite_difference
            = (value_plus - value_minus) / (2.0 * h);
        EXPECT_NEAR(grad_ket[alpha].real(), finite_difference.real(), 2.0e-4);
        EXPECT_NEAR(grad_ket[alpha].imag(), finite_difference.imag(), 2.0e-4);
        const std::complex<double> translation
            = grad_bra[alpha] + grad_ket[alpha]
              + ModuleBase::IMAG_UNIT * q[alpha] * value;
        EXPECT_NEAR(std::abs(translation), 0.0, 1.0e-10);
    }
}

TEST_F(TwoCenterIntegratorTest, GeneralizedOverlapPreparedWorkspace)
{
    nfile = 1;
    orb.build(nfile, file, 'o');
    ModuleBase::SphericalBesselTransformer sbt;
    orb.set_transformer(sbt);
    const double rmax = 2.0 * orb.rcut_max();
    const int nr = static_cast<int>(rmax / 0.01) + 1;
    orb.set_uniform_grid(true, nr, rmax, 'i', true);

    GeneralizedOverlapIntegrator generalized(
        ModuleBase::Vector3<double>(0.17, -0.11, 0.09), 8, nr, rmax);
    generalized.prepare_angular_momenta(orb.lmax());
    std::vector<std::pair<const NumericalRadial*, const NumericalRadial*> >
        orbital_pairs;
    orbital_pairs.push_back(std::make_pair(&orb(0, 1, 0), &orb(0, 1, 0)));
    generalized.prepare_radial_tables(orbital_pairs);

    const ModuleBase::Vector3<double> center_bra(0.2, -0.1, 0.3);
    const ModuleBase::Vector3<double> center_ket(1.1, 0.6, -0.4);
    GeneralizedOverlapIntegrator::GeometryWorkspace geometry;
    generalized.prepare_geometry(center_bra, center_ket, orb.lmax(), true,
                                 geometry);

    std::complex<double> direct_value;
    std::complex<double> direct_bra[3];
    std::complex<double> direct_ket[3];
    std::complex<double> prepared_value;
    std::complex<double> prepared_bra[3];
    std::complex<double> prepared_ket[3];
    generalized.calculate(orb(0, 1, 0), 1, orb(0, 1, 0), -1, center_bra,
                          center_ket, &direct_value, direct_bra, direct_ket);
    generalized.calculate(orb(0, 1, 0), 1, orb(0, 1, 0), -1, geometry,
                          &prepared_value, prepared_bra, prepared_ket);

    EXPECT_EQ(prepared_value, direct_value);
    for (int alpha = 0; alpha < 3; ++alpha)
    {
        EXPECT_EQ(prepared_bra[alpha], direct_bra[alpha]);
        EXPECT_EQ(prepared_ket[alpha], direct_ket[alpha]);
    }

    GeneralizedOverlapIntegrator::GeometryWorkspace value_only_geometry;
    generalized.prepare_geometry(center_bra, center_ket, orb.lmax(), false,
                                 value_only_geometry);
    std::complex<double> value_only;
    generalized.calculate(orb(0, 1, 0), 1, orb(0, 1, 0), -1,
                          value_only_geometry, &value_only, nullptr, nullptr);
    EXPECT_EQ(value_only, direct_value);
}

TEST_F(TwoCenterIntegratorTest, GeneralizedOverlapNonzeroMomentumQuadrature)
{
    nfile = 1;
    orb.build(nfile, file, 'o');
    ModuleBase::SphericalBesselTransformer sbt;
    orb.set_transformer(sbt);
    const double rmax = 2.0 * orb.rcut_max();
    const double dr = 0.01;
    const int nr = static_cast<int>(rmax / dr) + 1;
    orb.set_uniform_grid(true, nr, rmax, 'i', true);

    const ModuleBase::Vector3<double> q(0.17, -0.11, 0.09);
    GeneralizedOverlapIntegrator generalized(q, 12, nr, rmax);
    const ModuleBase::Vector3<double> center_bra(0.2, -0.1, 0.3);
    const ModuleBase::Vector3<double> center_ket(1.1, 0.6, -0.4);
    const NumericalRadial& s = orb(0, 0, 0);
    const NumericalRadial& p = orb(0, 1, 0);
    const struct
    {
        const NumericalRadial* bra;
        int m_bra;
        const NumericalRadial* ket;
        int m_ket;
    } cases[] = {{&s, 0, &s, 0}, {&s, 0, &p, 1},
                 {&p, -1, &s, 0}, {&p, 1, &p, -1}};

    for (const auto& test_case : cases)
    {
        std::complex<double> value;
        generalized.calculate(*test_case.bra, test_case.m_bra,
                              *test_case.ket, test_case.m_ket, center_bra,
                              center_ket, &value);
        const std::complex<double> reference = direct_generalized_overlap(
            *test_case.bra, test_case.m_bra, *test_case.ket, test_case.m_ket,
            center_bra, center_ket, q);
        EXPECT_NEAR(value.real(), reference.real(), 3.0e-5);
        EXPECT_NEAR(value.imag(), reference.imag(), 3.0e-5);
    }
}

TEST_F(TwoCenterIntegratorTest, GeneralizedOverlapAllSPDChannelsQuadrature)
{
    nfile = 1;
    orb.build(nfile, file, 'o');
    ModuleBase::SphericalBesselTransformer sbt;
    orb.set_transformer(sbt);
    const double rmax = 2.0 * orb.rcut_max();
    const double dr = 0.01;
    // Match the production TwoCenterBundle grid: its real-space range is
    // 2*rcut while the point count is currently based on rcut/dr.
    const int nr = static_cast<int>(orb.rcut_max() / dr) + 1;
    orb.set_uniform_grid(true, nr, rmax, 'i', true);

    // Comparable to the adjacent-string momentum in the 1x1x4 diamond gate.
    const ModuleBase::Vector3<double> q(0.0, 0.0, 0.233);
    GeneralizedOverlapIntegrator generalized(q, 12, nr, rmax);
    const ModuleBase::Vector3<double> center_bra(0.0, 0.0, 0.0);
    const ModuleBase::Vector3<double> center_ket(1.3, -0.8, 1.1);
    for (int l1 = 0; l1 <= 2; ++l1)
    {
        for (int zeta1 = 0; zeta1 < orb.nzeta(0, l1); ++zeta1)
        {
            const NumericalRadial& bra = orb(0, l1, zeta1);
            for (int m1 = -l1; m1 <= l1; ++m1)
            {
                for (int l2 = 0; l2 <= 2; ++l2)
                {
                    for (int zeta2 = 0; zeta2 < orb.nzeta(0, l2); ++zeta2)
                    {
                        const NumericalRadial& ket = orb(0, l2, zeta2);
                        for (int m2 = -l2; m2 <= l2; ++m2)
                        {
                            std::complex<double> value;
                            generalized.calculate(
                                bra, m1, ket, m2, center_bra, center_ket,
                                &value);
                            const std::complex<double> reference
                                = direct_generalized_overlap(
                                    bra, m1, ket, m2, center_bra, center_ket,
                                    q);
                            EXPECT_NEAR(value.real(), reference.real(), 5.0e-5)
                                << "l1,z1,m1,l2,z2,m2 = " << l1 << ","
                                << zeta1 << "," << m1 << "," << l2 << ","
                                << zeta2 << "," << m2;
                            EXPECT_NEAR(value.imag(), reference.imag(), 5.0e-5)
                                << "l1,z1,m1,l2,z2,m2 = " << l1 << ","
                                << zeta1 << "," << m1 << "," << l2 << ","
                                << zeta2 << "," << m2;
                        }
                    }
                }
            }
        }
    }
}

TEST_F(TwoCenterIntegratorTest, HessianSymmetry)
{
    nfile = 3;
    orb.build(nfile, file, 'o');

    ModuleBase::SphericalBesselTransformer sbt;
    orb.set_transformer(sbt);

    double rmax = orb.rcut_max() * 2.0;
    double dr = 0.01;
    int nr = static_cast<int>(rmax / dr) + 1;

    orb.set_uniform_grid(true, nr, rmax, 'i', true);

    S_intor.tabulate(orb, orb, 'S', nr, rmax);
    T_intor.tabulate(orb, orb, 'T', nr, rmax);

    ModuleBase::Vector3<double> R(1.5, 2.0, 1.0);
    double hess[9];

    // Test S operator
    S_intor.calculate(0, 1, 0, 0, 1, 1, 0, 0, R, nullptr, nullptr, hess);

    EXPECT_NEAR(hess[1], hess[3], 1e-10);  // H_xy == H_yx
    EXPECT_NEAR(hess[2], hess[6], 1e-10);  // H_xz == H_zx
    EXPECT_NEAR(hess[5], hess[7], 1e-10);  // H_yz == H_zy

    // Test T operator
    T_intor.calculate(0, 1, 0, 0, 1, 1, 0, 0, R, nullptr, nullptr, hess);

    EXPECT_NEAR(hess[1], hess[3], 1e-10);  // H_xy == H_yx
    EXPECT_NEAR(hess[2], hess[6], 1e-10);  // H_xz == H_zx
    EXPECT_NEAR(hess[5], hess[7], 1e-10);  // H_yz == H_zy
}

TEST_F(TwoCenterIntegratorTest, HessianFiniteDifference)
{
    nfile = 3;
    orb.build(nfile, file, 'o');

    ModuleBase::SphericalBesselTransformer sbt;
    orb.set_transformer(sbt);

    double rmax = orb.rcut_max() * 2.0;
    double dr = 0.01;
    int nr = static_cast<int>(rmax / dr) + 1;

    orb.set_uniform_grid(true, nr, rmax, 'i', true);

    S_intor.tabulate(orb, orb, 'S', nr, rmax);
    T_intor.tabulate(orb, orb, 'T', nr, rmax);

    ModuleBase::Vector3<double> R(1.5, 2.0, 1.0);
    double hess_analytical[9];
    double hess_numerical[9];
    double eps = 1e-5;

    // Test S operator
    S_intor.calculate(0, 1, 0, 0, 1, 1, 0, 0, R, nullptr, nullptr, hess_analytical);

    // Compute numerical Hessian via finite differences
    for (int alpha = 0; alpha < 3; ++alpha)
    {
        for (int beta = 0; beta < 3; ++beta)
        {
            ModuleBase::Vector3<double> R_plus = R, R_minus = R;
            R_plus[beta] += eps;
            R_minus[beta] -= eps;

            double grad_plus[3], grad_minus[3];
            S_intor.calculate(0, 1, 0, 0, 1, 1, 0, 0, R_plus, nullptr, grad_plus, nullptr);
            S_intor.calculate(0, 1, 0, 0, 1, 1, 0, 0, R_minus, nullptr, grad_minus, nullptr);

            hess_numerical[alpha * 3 + beta] = (grad_plus[alpha] - grad_minus[alpha]) / (2.0 * eps);
        }
    }

    // Compare with tolerance appropriate for finite differences
    for (int i = 0; i < 9; ++i)
    {
        EXPECT_NEAR(hess_analytical[i], hess_numerical[i], 1e-5);
    }

    // Test T operator
    T_intor.calculate(0, 1, 0, 0, 1, 1, 0, 0, R, nullptr, nullptr, hess_analytical);

    for (int alpha = 0; alpha < 3; ++alpha)
    {
        for (int beta = 0; beta < 3; ++beta)
        {
            ModuleBase::Vector3<double> R_plus = R, R_minus = R;
            R_plus[beta] += eps;
            R_minus[beta] -= eps;

            double grad_plus[3], grad_minus[3];
            T_intor.calculate(0, 1, 0, 0, 1, 1, 0, 0, R_plus, nullptr, grad_plus, nullptr);
            T_intor.calculate(0, 1, 0, 0, 1, 1, 0, 0, R_minus, nullptr, grad_minus, nullptr);

            hess_numerical[alpha * 3 + beta] = (grad_plus[alpha] - grad_minus[alpha]) / (2.0 * eps);
        }
    }

    for (int i = 0; i < 9; ++i)
    {
        EXPECT_NEAR(hess_analytical[i], hess_numerical[i], 1e-5);
    }
}

TEST_F(TwoCenterIntegratorTest, HessianDoesNotBreakGradient)
{
    nfile = 3;
    orb.build(nfile, file, 'o');

    ModuleBase::SphericalBesselTransformer sbt;
    orb.set_transformer(sbt);

    double rmax = orb.rcut_max() * 2.0;
    double dr = 0.01;
    int nr = static_cast<int>(rmax / dr) + 1;

    orb.set_uniform_grid(true, nr, rmax, 'i', true);

    S_intor.tabulate(orb, orb, 'S', nr, rmax);
    T_intor.tabulate(orb, orb, 'T', nr, rmax);

    ModuleBase::Vector3<double> R(1.5, 2.0, 1.0);
    double grad_only[3], grad_with_hess[3], hess[9];

    // Test S operator
    S_intor.calculate(0, 1, 0, 0, 1, 1, 0, 0, R, nullptr, grad_only, nullptr);
    S_intor.calculate(0, 1, 0, 0, 1, 1, 0, 0, R, nullptr, grad_with_hess, hess);

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_NEAR(grad_only[i], grad_with_hess[i], 1e-12);
    }

    // Test T operator
    T_intor.calculate(0, 1, 0, 0, 1, 1, 0, 0, R, nullptr, grad_only, nullptr);
    T_intor.calculate(0, 1, 0, 0, 1, 1, 0, 0, R, nullptr, grad_with_hess, hess);

    for (int i = 0; i < 3; ++i)
    {
        EXPECT_NEAR(grad_only[i], grad_with_hess[i], 1e-12);
    }
}

int main(int argc, char** argv)
{

#ifdef __MPI
    MPI_Init(&argc, &argv);
#endif

    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

#ifdef __MPI
    MPI_Finalize();
#endif

    return result;
}
