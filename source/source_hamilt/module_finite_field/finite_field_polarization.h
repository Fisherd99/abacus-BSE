#ifndef MODULE_FINITE_FIELD_POLARIZATION_H
#define MODULE_FINITE_FIELD_POLARIZATION_H

#include "source_base/vector3.h"
#include "source_cell/kpoint_strings.h"

#include <array>
#include <complex>
#include <iosfwd>
#include <string>
#include <vector>

namespace hamilt
{

struct FiniteFieldCartesianGeometry
{
    int cartesian_axis = 0;
    std::array<double, 3> lattice_periods{{0.0, 0.0, 0.0}};
    std::array<double, 3> cartesian_projections{{0.0, 0.0, 0.0}};
};

struct FiniteFieldIon
{
    double valence_charge = 0.0;
    std::array<double, 3> direct_position{{0.0, 0.0, 0.0}};
    std::array<double, 3> cartesian_position_bohr{{0.0, 0.0, 0.0}};
};

struct FiniteFieldPolarization
{
    double berry_phase = 0.0;
    double electronic_dipole = 0.0;
    double ionic_dipole = 0.0;
    double polarization_quantum = 0.0;
};

/** Cartesian polarization reconstructed from the three lattice-string dipoles.
 * Values are in the Rydberg atomic-unit convention used internally by QE's
 * `lelfield`: sqrt(e2) charge per bohr squared. */
struct FiniteFieldCartesianPolarization
{
    std::array<double, 3> electronic{{0.0, 0.0, 0.0}};
    std::array<double, 3> ionic{{0.0, 0.0, 0.0}};
    std::array<double, 3> total{{0.0, 0.0, 0.0}};
};

struct FiniteFieldElectricEnthalpy
{
    FiniteFieldPolarization principal;
    double branch_offset = 0.0;
    double continuous_electronic_dipole = 0.0;
    double electric_enthalpy = 0.0;
};

struct FiniteFieldPolarizationBranchState
{
    bool initialized = false;
    double previous_principal_dipole = 0.0;
    double branch_offset = 0.0;
};

struct FiniteFieldBranchDirectionState
{
    bool active = false;
    double lattice_period_bohr = 0.0;
    double polarization_quantum = 0.0;
    FiniteFieldPolarizationBranchState branch;
};

struct FiniteFieldBranchState
{
    int cartesian_axis = 0;
    std::array<FiniteFieldBranchDirectionState, 3> directions;
};

FiniteFieldCartesianGeometry build_finite_field_cartesian_geometry(
    const std::array<ModuleBase::Vector3<double>, 3>& direct_lattice_vectors,
    int cartesian_axis);

/** QE-compatible coefficient multiplying the centered covariant derivative. */
std::complex<double> finite_field_coupling(double field_ry_au,
                                           double lattice_period_bohr,
                                           int kpoints_per_string);

FiniteFieldPolarization calculate_finite_field_polarization(
    const ModuleCell::KPointStrings& strings,
    const std::vector<std::complex<double>>& forward_overlap_determinants,
    double lattice_period_bohr,
    int spin_degeneracy,
    const std::vector<FiniteFieldIon>& ions);

/** Apply the zero-field berry_phase module's nspin=1 principal-branch rule.
 *
 * This is deliberately a representation constraint only.  The finite-field
 * controller applies it when a direction is initialized and when it reports
 * the final Cartesian polarization; its SCF-cycle branch accumulator remains
 * free to follow a continuous electric-enthalpy branch. */
FiniteFieldPolarization finite_field_legacy_principal_branch(
    const FiniteFieldPolarization& polarization,
    int lattice_direction,
    const std::vector<FiniteFieldIon>& ions);

double finite_field_polarization_quantum(double lattice_period_bohr);

double finite_field_ionic_force(double field_ry_au,
                                double valence_charge);

double finite_field_cartesian_ionic_dipole(
    const std::vector<FiniteFieldIon>& ions,
    int cartesian_axis);

FiniteFieldCartesianPolarization finite_field_cartesian_polarization(
    const std::array<FiniteFieldPolarization, 3>& lattice_polarizations,
    const std::array<std::array<double, 3>, 3>& lattice_unit_vectors,
    const std::vector<FiniteFieldIon>& ions,
    double cell_volume_bohr3);

/** Cartesian reconstruction for the final user-facing report.  Each lattice
 * contribution is first put on the legacy berry_phase principal branch. */
FiniteFieldCartesianPolarization finite_field_legacy_reported_polarization(
    const std::array<FiniteFieldPolarization, 3>& lattice_polarizations,
    const std::array<std::array<double, 3>, 3>& lattice_unit_vectors,
    const std::vector<FiniteFieldIon>& ions,
    double cell_volume_bohr3);

/** Convert an internally Ry-unit Cartesian polarization to physical
 * elementary-charge density (e / bohr^2) for user-facing output. */
FiniteFieldCartesianPolarization finite_field_physical_polarization(
    const FiniteFieldCartesianPolarization& polarization);

class FiniteFieldPolarizationBranch
{
  public:
    FiniteFieldElectricEnthalpy update(
        const FiniteFieldPolarization& principal,
        double field_ry_au);

    FiniteFieldPolarizationBranchState state() const;

    void restore(const FiniteFieldPolarizationBranchState& state,
                 double expected_polarization_quantum);

    void reset();

  private:
    bool initialized = false;
    double previous_principal_dipole = 0.0;
    double branch_offset = 0.0;
};

const char* finite_field_branch_state_filename();

void validate_finite_field_branch_state(
    const FiniteFieldBranchState& state,
    int expected_cartesian_axis,
    const std::array<bool, 3>& expected_active,
    const std::array<double, 3>& expected_lattice_period_bohr,
    const std::array<double, 3>& expected_polarization_quantum);

void write_finite_field_branch_state(
    std::ostream& output,
    const FiniteFieldBranchState& state);

FiniteFieldBranchState read_finite_field_branch_state(std::istream& input);

void write_finite_field_branch_state_file(
    const std::string& path,
    const FiniteFieldBranchState& state);

FiniteFieldBranchState read_finite_field_branch_state_file(
    const std::string& path);

} // namespace hamilt

#endif
