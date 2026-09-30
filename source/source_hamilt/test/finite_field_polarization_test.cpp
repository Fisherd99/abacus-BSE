#include "gtest/gtest.h"

#include "source_hamilt/module_finite_field/finite_field_polarization.h"

#include <array>
#include <cmath>
#include <complex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace
{

using Complex = std::complex<double>;

ModuleCell::KPointStrings two_point_string()
{
    ModuleCell::KPointStrings strings;
    strings.direction = 3;
    strings.points_per_string = 3;
    strings.indices = {{0, 1, 0}};
    return strings;
}

ModuleCell::KPointStrings gamma_only_string()
{
    ModuleCell::KPointStrings strings;
    strings.direction = 1;
    strings.points_per_string = 2;
    strings.indices = {{0, 0}};
    return strings;
}

TEST(FiniteFieldGeometryTest, DecomposesCartesianFieldInSkewCell)
{
    using Vector = ModuleBase::Vector3<double>;
    const std::array<Vector, 3> lattice = {
        Vector(-2.0, 2.0, 0.0),
        Vector(2.0, 2.0, 0.0),
        Vector(0.0, 0.0, 5.0)};

    const hamilt::FiniteFieldCartesianGeometry geometry
        = hamilt::build_finite_field_cartesian_geometry(lattice, 1);

    EXPECT_NEAR(geometry.cartesian_projections[0],
                -1.0 / std::sqrt(2.0),
                1.0e-15);
    EXPECT_NEAR(geometry.cartesian_projections[1],
                1.0 / std::sqrt(2.0),
                1.0e-15);
    EXPECT_DOUBLE_EQ(geometry.cartesian_projections[2], 0.0);
}

TEST(FiniteFieldPolarizationTest, ComputesClosedLoopPhaseAndQeDipoles)
{
    const double phase = 0.4;
    const std::vector<Complex> determinants = {
        std::polar(2.0, 0.1), std::polar(3.0, 0.3)};
    hamilt::FiniteFieldIon ion;
    ion.valence_charge = 4.0;
    ion.direct_position = {{0.1, 0.2, 0.25}};

    const hamilt::FiniteFieldPolarization polarization
        = hamilt::calculate_finite_field_polarization(
            two_point_string(), determinants, 10.0, 2, {ion});
    const double pi = std::acos(-1.0);

    EXPECT_NEAR(polarization.berry_phase, phase, 1.0e-15);
    EXPECT_NEAR(polarization.electronic_dipole,
                2.0 * phase * std::sqrt(2.0) * 10.0 / (2.0 * pi),
                1.0e-14);
    EXPECT_NEAR(polarization.ionic_dipole,
                std::sqrt(2.0) * 10.0,
                1.0e-14);
    EXPECT_NEAR(polarization.polarization_quantum,
                std::sqrt(2.0) * 10.0,
                1.0e-14);
}

TEST(FiniteFieldPolarizationTest, ComputesGammaOnlyWilsonLoop)
{
    const hamilt::FiniteFieldPolarization polarization
        = hamilt::calculate_finite_field_polarization(
            gamma_only_string(), {std::polar(1.0, 0.35)}, 9.0, 2, {});

    EXPECT_NEAR(polarization.berry_phase, 0.35, 1.0e-15);
    EXPECT_NEAR(polarization.electronic_dipole,
                2.0 * 0.35 * std::sqrt(2.0) * 9.0 / (2.0 * std::acos(-1.0)),
                1.0e-14);
    EXPECT_NEAR(polarization.polarization_quantum,
                std::sqrt(2.0) * 9.0,
                1.0e-14);
}

TEST(FiniteFieldPolarizationTest, IsInvariantUnderKDependentGauge)
{
    const double original_phase_0 = 0.2;
    const double original_phase_1 = -0.7;
    const double gauge_0 = 1.1;
    const double gauge_1 = -0.4;
    const std::vector<Complex> original = {
        std::polar(1.0, original_phase_0),
        std::polar(1.0, original_phase_1)};
    const std::vector<Complex> transformed = {
        original[0] * std::polar(1.0, gauge_1 - gauge_0),
        original[1] * std::polar(1.0, gauge_0 - gauge_1)};

    const hamilt::FiniteFieldPolarization reference
        = hamilt::calculate_finite_field_polarization(
            two_point_string(), original, 8.0, 2, {});
    const hamilt::FiniteFieldPolarization gauged
        = hamilt::calculate_finite_field_polarization(
            two_point_string(), transformed, 8.0, 2, {});

    EXPECT_NEAR(gauged.berry_phase, reference.berry_phase, 1.0e-15);
    EXPECT_NEAR(gauged.electronic_dipole,
                reference.electronic_dipole,
                1.0e-15);
}

TEST(FiniteFieldPolarizationTest, AveragesEquivalentStrings)
{
    ModuleCell::KPointStrings strings;
    strings.direction = 1;
    strings.points_per_string = 3;
    strings.indices = {{0, 1, 0}, {2, 3, 2}};
    const std::vector<Complex> determinants = {
        std::polar(1.0, 0.2),
        std::polar(1.0, 0.4),
        std::polar(1.0, -0.1),
        std::polar(1.0, 0.3)};

    const hamilt::FiniteFieldPolarization polarization
        = hamilt::calculate_finite_field_polarization(
            strings, determinants, 5.0, 2, {});

    EXPECT_NEAR(polarization.berry_phase, 0.4, 1.0e-15);
}

TEST(FiniteFieldPolarizationBranchTest, UnwrapsWithQePolarizationQuantum)
{
    hamilt::FiniteFieldPolarizationBranch branch;
    hamilt::FiniteFieldPolarization polarization;
    polarization.polarization_quantum = 10.0;
    polarization.ionic_dipole = 2.0;
    polarization.electronic_dipole = 4.8;

    hamilt::FiniteFieldElectricEnthalpy result
        = branch.update(polarization, 0.01);
    EXPECT_DOUBLE_EQ(result.branch_offset, 0.0);
    EXPECT_NEAR(result.electric_enthalpy, -0.068, 1.0e-15);

    polarization.electronic_dipole = -4.9;
    result = branch.update(polarization, 0.01);
    EXPECT_DOUBLE_EQ(result.branch_offset, 10.0);
    EXPECT_NEAR(result.continuous_electronic_dipole, 5.1, 1.0e-15);
    EXPECT_NEAR(result.electric_enthalpy, -0.071, 1.0e-15);

    polarization.electronic_dipole = 4.7;
    result = branch.update(polarization, 0.01);
    EXPECT_DOUBLE_EQ(result.branch_offset, 0.0);
    EXPECT_NEAR(result.continuous_electronic_dipole, 4.7, 1.0e-15);
}

TEST(FiniteFieldPolarizationBranchTest, RestoresAndContinuesUnwrappedBranch)
{
    hamilt::FiniteFieldPolarizationBranch original;
    hamilt::FiniteFieldPolarization polarization;
    polarization.polarization_quantum = 10.0;
    polarization.electronic_dipole = 4.8;
    original.update(polarization, 0.01);
    polarization.electronic_dipole = -4.9;
    original.update(polarization, 0.01);

    hamilt::FiniteFieldPolarizationBranch restored;
    restored.restore(original.state(), polarization.polarization_quantum);
    polarization.electronic_dipole = -4.7;
    const hamilt::FiniteFieldElectricEnthalpy result
        = restored.update(polarization, 0.01);

    EXPECT_DOUBLE_EQ(result.branch_offset, 10.0);
    EXPECT_NEAR(result.continuous_electronic_dipole, 5.3, 1.0e-15);
}

TEST(FiniteFieldPolarizationBranchTest, RejectsIncompatibleRestoredOffset)
{
    hamilt::FiniteFieldPolarizationBranchState state;
    state.initialized = true;
    state.previous_principal_dipole = -4.9;
    state.branch_offset = 9.0;

    hamilt::FiniteFieldPolarizationBranch branch;
    EXPECT_THROW(branch.restore(state, 10.0), std::invalid_argument);
}

TEST(FiniteFieldPolarizationTest, ReproducesQeNcppIonicForce)
{
    EXPECT_NEAR(hamilt::finite_field_ionic_force(0.001, 4.0),
                std::sqrt(2.0) * 0.004,
                1.0e-16);
}

TEST(FiniteFieldPolarizationTest, ComputesCartesianIonicDipole)
{
    hamilt::FiniteFieldIon first;
    first.valence_charge = 4.0;
    first.cartesian_position_bohr = {{1.0, 2.0, 3.0}};
    hamilt::FiniteFieldIon second;
    second.valence_charge = 2.0;
    second.cartesian_position_bohr = {{-0.5, 0.25, 1.0}};

    EXPECT_NEAR(
        hamilt::finite_field_cartesian_ionic_dipole({first, second}, 1),
        std::sqrt(2.0) * 3.0,
        1.0e-15);
    EXPECT_NEAR(
        hamilt::finite_field_cartesian_ionic_dipole({first, second}, 2),
        std::sqrt(2.0) * 8.5,
        1.0e-15);
    EXPECT_THROW(
        hamilt::finite_field_cartesian_ionic_dipole({first}, 0),
        std::invalid_argument);
}

TEST(FiniteFieldPolarizationTest, ReconstructsCartesianPolarization)
{
    std::array<hamilt::FiniteFieldPolarization, 3> lattice;
    lattice[0].electronic_dipole = 2.0;
    lattice[1].electronic_dipole = -3.0;
    lattice[2].electronic_dipole = 4.0;
    const std::array<std::array<double, 3>, 3> unit_vectors = {{{{1.0, 0.0, 0.0}},
                                                                  {{0.0, 1.0, 0.0}},
                                                                  {{0.0, 0.0, 1.0}}}};
    hamilt::FiniteFieldIon ion;
    ion.valence_charge = 2.0;
    ion.cartesian_position_bohr = {{1.0, -2.0, 0.5}};
    const hamilt::FiniteFieldCartesianPolarization polarization
        = hamilt::finite_field_cartesian_polarization(
            lattice, unit_vectors, {ion}, 2.0);

    EXPECT_NEAR(polarization.electronic[0], 1.0, 1.0e-15);
    EXPECT_NEAR(polarization.electronic[1], -1.5, 1.0e-15);
    EXPECT_NEAR(polarization.electronic[2], 2.0, 1.0e-15);
    EXPECT_NEAR(polarization.ionic[0], std::sqrt(2.0), 1.0e-15);
    EXPECT_NEAR(polarization.total[1], -1.5 - 2.0 * std::sqrt(2.0), 1.0e-15);
    EXPECT_THROW(hamilt::finite_field_cartesian_polarization(
                     lattice, unit_vectors, {ion}, 0.0),
                 std::invalid_argument);
}

TEST(FiniteFieldPolarizationTest, ConvertsCartesianPolarizationToPhysicalChargeUnits)
{
    hamilt::FiniteFieldCartesianPolarization polarization_ry;
    polarization_ry.electronic = {{std::sqrt(2.0), -2.0 * std::sqrt(2.0), 0.0}};
    polarization_ry.ionic = {{-3.0 * std::sqrt(2.0), 0.0, 4.0 * std::sqrt(2.0)}};
    polarization_ry.total = {{-2.0 * std::sqrt(2.0), -2.0 * std::sqrt(2.0),
                              4.0 * std::sqrt(2.0)}};

    const hamilt::FiniteFieldCartesianPolarization physical
        = hamilt::finite_field_physical_polarization(polarization_ry);

    EXPECT_NEAR(physical.electronic[0], 1.0, 1.0e-15);
    EXPECT_NEAR(physical.electronic[1], -2.0, 1.0e-15);
    EXPECT_NEAR(physical.ionic[0], -3.0, 1.0e-15);
    EXPECT_NEAR(physical.ionic[2], 4.0, 1.0e-15);
    EXPECT_NEAR(physical.total[0], -2.0, 1.0e-15);
    EXPECT_NEAR(physical.total[2], 4.0, 1.0e-15);
}

TEST(FiniteFieldBranchStateTest, RoundTripsAllDirections)
{
    hamilt::FiniteFieldBranchState reference;
    reference.cartesian_axis = 3;
    for (int direction = 0; direction < 3; ++direction)
    {
        hamilt::FiniteFieldBranchDirectionState& direction_state
            = reference.directions[direction];
        direction_state.active = direction != 1;
        direction_state.lattice_period_bohr = 8.0 + direction;
        direction_state.polarization_quantum
            = hamilt::finite_field_polarization_quantum(8.0 + direction);
        if (direction_state.active)
        {
            direction_state.branch.initialized = true;
            direction_state.branch.previous_principal_dipole
                = -2.5 - direction;
            direction_state.branch.branch_offset
                = direction_state.polarization_quantum;
        }
    }

    std::ostringstream output;
    hamilt::write_finite_field_branch_state(output, reference);
    std::istringstream input(output.str());
    const hamilt::FiniteFieldBranchState restored
        = hamilt::read_finite_field_branch_state(input);

    EXPECT_EQ(restored.cartesian_axis, reference.cartesian_axis);
    for (int direction = 0; direction < 3; ++direction)
    {
        EXPECT_EQ(restored.directions[direction].active,
                  reference.directions[direction].active);
        EXPECT_DOUBLE_EQ(restored.directions[direction].lattice_period_bohr,
                         reference.directions[direction].lattice_period_bohr);
        EXPECT_DOUBLE_EQ(restored.directions[direction].polarization_quantum,
                         reference.directions[direction].polarization_quantum);
        EXPECT_EQ(restored.directions[direction].branch.initialized,
                  reference.directions[direction].branch.initialized);
        EXPECT_DOUBLE_EQ(
            restored.directions[direction].branch.previous_principal_dipole,
            reference.directions[direction].branch.previous_principal_dipole);
        EXPECT_DOUBLE_EQ(restored.directions[direction].branch.branch_offset,
                         reference.directions[direction].branch.branch_offset);
    }
}

TEST(FiniteFieldBranchStateTest, RejectsIncompatibleActiveDirections)
{
    hamilt::FiniteFieldBranchState state;
    state.cartesian_axis = 1;
    std::array<bool, 3> expected_active = {{true, true, false}};
    std::array<double, 3> periods = {{8.0, 9.0, 10.0}};
    std::array<double, 3> quanta;
    for (int direction = 0; direction < 3; ++direction)
    {
        state.directions[direction].active = expected_active[direction];
        state.directions[direction].lattice_period_bohr = periods[direction];
        state.directions[direction].polarization_quantum
            = hamilt::finite_field_polarization_quantum(periods[direction]);
        quanta[direction] = state.directions[direction].polarization_quantum;
    }
    state.directions[1].active = false;

    EXPECT_THROW(hamilt::validate_finite_field_branch_state(
                     state, 1, expected_active, periods, quanta),
                 std::invalid_argument);
}

TEST(FiniteFieldPolarizationTest, RejectsOpenStrings)
{
    ModuleCell::KPointStrings strings = two_point_string();
    strings.indices = {{0, 1, 1}};
    EXPECT_THROW(hamilt::calculate_finite_field_polarization(
                     strings,
                     {Complex(1.0, 0.0), Complex(1.0, 0.0)},
                     10.0,
                     2,
                     {}),
                 std::invalid_argument);
}

TEST(FiniteFieldPolarizationTest, CouplingUsesRyChargeAndCenteredDifference)
{
    const std::complex<double> coupling
        = hamilt::finite_field_coupling(0.002, 10.0, 4);
    EXPECT_DOUBLE_EQ(coupling.real(), 0.0);
    EXPECT_NEAR(coupling.imag(),
                -0.002 * std::sqrt(2.0) * 10.0 * 4.0
                    / (4.0 * std::acos(-1.0)),
                1.0e-15);
}

} // namespace
