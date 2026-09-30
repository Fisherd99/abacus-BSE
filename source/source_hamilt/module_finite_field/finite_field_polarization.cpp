#include "finite_field_polarization.h"

#include "source_base/constants.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>

namespace hamilt
{

namespace
{

const char* const branch_format_header = "ABACUS_FINITE_FIELD_BRANCH_STATE";
const int branch_format_version = 2;
// ABACUS uses Rydberg atomic units in this finite-field implementation.
// In this convention e^2 = 2, hence the electronic charge magnitude is sqrt(e2).
const double electron_charge_ry = std::sqrt(ModuleBase::e2);

double vector_component(const ModuleBase::Vector3<double>& vector,
                        const int axis)
{
    return axis == 0 ? vector.x : (axis == 1 ? vector.y : vector.z);
}

double compatibility_tolerance(const double reference)
{
    return 1.0e-10 * std::max(1.0, std::abs(reference));
}

void require_compatible(const double value,
                        const double expected,
                        const std::string& quantity)
{
    if (!std::isfinite(value)
        || std::abs(value - expected) > compatibility_tolerance(expected))
    {
        throw std::invalid_argument(
            "finite-field branch state has incompatible " + quantity);
    }
}

template <typename Value>
Value parse_value(const std::map<std::string, std::string>& values,
                  const std::string& key)
{
    const std::map<std::string, std::string>::const_iterator iterator
        = values.find(key);
    if (iterator == values.end())
    {
        throw std::runtime_error(
            "finite-field branch state is missing key " + key);
    }
    std::istringstream parser(iterator->second);
    Value value;
    std::string trailing;
    if (!(parser >> value) || (parser >> trailing))
    {
        throw std::runtime_error(
            "finite-field branch state has invalid value for " + key);
    }
    return value;
}

void validate_strings(
    const ModuleCell::KPointStrings& strings,
    const std::vector<std::complex<double>>& forward_overlap_determinants)
{
    if (strings.direction < 1 || strings.direction > 3
        || strings.points_per_string < 2 || strings.indices.empty())
    {
        throw std::invalid_argument("finite-field polarization k strings are invalid");
    }
    if (forward_overlap_determinants.empty())
    {
        throw std::invalid_argument("finite-field polarization overlaps are empty");
    }

    std::vector<int> visit_count(forward_overlap_determinants.size(), 0);
    for (const std::vector<int>& string : strings.indices)
    {
        if (static_cast<int>(string.size()) != strings.points_per_string
            || string.front() != string.back())
        {
            throw std::invalid_argument("finite-field polarization k string is not closed");
        }
        for (int position = 0; position < strings.points_per_string - 1; ++position)
        {
            const int ik = string[position];
            if (ik < 0
                || ik >= static_cast<int>(forward_overlap_determinants.size()))
            {
                throw std::out_of_range("finite-field polarization k-point index is invalid");
            }
            ++visit_count[ik];
        }
    }
    for (const int count : visit_count)
    {
        if (count != 1)
        {
            throw std::invalid_argument(
                "finite-field polarization strings must cover every k point once");
        }
    }
}

} // namespace

FiniteFieldCartesianGeometry build_finite_field_cartesian_geometry(
    const std::array<ModuleBase::Vector3<double>, 3>& direct_lattice_vectors,
    const int cartesian_axis)
{
    if (cartesian_axis < 1 || cartesian_axis > 3)
    {
        throw std::invalid_argument(
            "finite-field Cartesian axis must be 1, 2, or 3");
    }

    const int axis = cartesian_axis - 1;
    FiniteFieldCartesianGeometry geometry;
    geometry.cartesian_axis = cartesian_axis;
    for (int direction = 0; direction < 3; ++direction)
    {
        const double period = direct_lattice_vectors[direction].norm();
        if (!std::isfinite(period) || period <= 0.0)
        {
            throw std::invalid_argument(
                "finite-field direct lattice vectors must be finite and nonzero");
        }
        geometry.lattice_periods[direction] = period;
        const double projection
            = vector_component(direct_lattice_vectors[direction], axis)
              / period;
        geometry.cartesian_projections[direction]
            = std::abs(projection) <= 1.0e-10 ? 0.0 : projection;
    }
    return geometry;
}

std::complex<double> finite_field_coupling(
    const double field_ry_au,
    const double lattice_period_bohr,
    const int kpoints_per_string)
{
    if (lattice_period_bohr <= 0.0 || kpoints_per_string <= 0)
    {
        throw std::invalid_argument(
            "finite-field k string dimensions are invalid");
    }
    const double delta_k = 2.0 * std::acos(-1.0)
                           / lattice_period_bohr / kpoints_per_string;
    return std::complex<double>(
        0.0, -field_ry_au * electron_charge_ry / (2.0 * delta_k));
}

double finite_field_polarization_quantum(const double lattice_period_bohr)
{
    if (lattice_period_bohr <= 0.0)
    {
        throw std::invalid_argument("finite-field polarization lattice period must be positive");
    }
    return electron_charge_ry * lattice_period_bohr;
}

FiniteFieldPolarization calculate_finite_field_polarization(
    const ModuleCell::KPointStrings& strings,
    const std::vector<std::complex<double>>& forward_overlap_determinants,
    const double lattice_period_bohr,
    const int spin_degeneracy,
    const std::vector<FiniteFieldIon>& ions)
{
    validate_strings(strings, forward_overlap_determinants);
    if (lattice_period_bohr <= 0.0)
    {
        throw std::invalid_argument("finite-field polarization lattice period must be positive");
    }
    if (spin_degeneracy <= 0)
    {
        throw std::invalid_argument("finite-field polarization spin degeneracy must be positive");
    }

    double phase_sum = 0.0;
    for (const std::vector<int>& string : strings.indices)
    {
        std::complex<double> string_product(1.0, 0.0);
        for (int position = 0; position < strings.points_per_string - 1; ++position)
        {
            const std::complex<double> determinant
                = forward_overlap_determinants[string[position]];
            if (std::abs(determinant) == 0.0)
            {
                throw std::runtime_error(
                    "finite-field polarization overlap determinant is zero");
            }
            string_product *= determinant;
        }
        phase_sum += std::arg(string_product);
    }

    const double pi = std::acos(-1.0);
    const double average_phase
        = phase_sum / static_cast<double>(strings.indices.size());
    const double dipole_per_radian
        = electron_charge_ry * lattice_period_bohr / (2.0 * pi);

    FiniteFieldPolarization polarization;
    polarization.berry_phase = average_phase;
    polarization.electronic_dipole
        = spin_degeneracy * average_phase * dipole_per_radian;
    polarization.polarization_quantum
        = finite_field_polarization_quantum(lattice_period_bohr);

    const int coordinate = strings.direction - 1;
    for (const FiniteFieldIon& ion : ions)
    {
        polarization.ionic_dipole
            += electron_charge_ry * lattice_period_bohr
               * ion.valence_charge * ion.direct_position[coordinate];
    }
    return polarization;
}

FiniteFieldPolarization finite_field_legacy_principal_branch(
    const FiniteFieldPolarization& polarization,
    const int lattice_direction,
    const std::vector<FiniteFieldIon>& ions)
{
    if (lattice_direction < 1 || lattice_direction > 3
        || polarization.polarization_quantum <= 0.0)
    {
        throw std::invalid_argument(
            "finite-field legacy principal branch arguments are invalid");
    }

    // Keep the nspin=1 convention of module_unk/berryphase.cpp exactly:
    // electron phase is modulo 2, i.e. its dipole is modulo one physical
    // polarization quantum.
    FiniteFieldPolarization constrained = polarization;
    const double quantum = polarization.polarization_quantum;
    constrained.electronic_dipole
        -= quantum * std::round(constrained.electronic_dipole / quantum);

    // The old module first constrains each ionic phase by the parity of the
    // ionic valence, then constrains their sum modulo 2 unless an odd-valence
    // species is present.  Preserve that historical output convention here.
    const int coordinate = lattice_direction - 1;
    bool has_odd_valence = false;
    double ionic_phase = 0.0;
    for (const FiniteFieldIon& ion : ions)
    {
        const int valence = static_cast<int>(ion.valence_charge);
        if (std::abs(ion.valence_charge - static_cast<double>(valence))
            > 1.0e-10)
        {
            throw std::invalid_argument(
                "finite-field ionic valence must be integral for legacy branch constraint");
        }
        const int modulus = std::abs(valence) % 2 == 1 ? 1 : 2;
        has_odd_valence = has_odd_valence || modulus == 1;
        double contribution
            = static_cast<double>(valence) * ion.direct_position[coordinate];
        contribution -= static_cast<double>(modulus)
                        * std::round(contribution / static_cast<double>(modulus));
        ionic_phase += contribution;
    }
    const int total_modulus = has_odd_valence ? 1 : 2;
    ionic_phase -= static_cast<double>(total_modulus)
                   * std::round(ionic_phase / static_cast<double>(total_modulus));
    constrained.ionic_dipole = quantum * ionic_phase;
    return constrained;
}

double finite_field_ionic_force(const double field_ry_au,
                                const double valence_charge)
{
    return electron_charge_ry * field_ry_au * valence_charge;
}

double finite_field_cartesian_ionic_dipole(
    const std::vector<FiniteFieldIon>& ions,
    const int cartesian_axis)
{
    if (cartesian_axis < 1 || cartesian_axis > 3)
    {
        throw std::invalid_argument(
            "finite-field Cartesian ionic-dipole axis must be 1, 2, or 3");
    }
    double ionic_dipole = 0.0;
    for (const FiniteFieldIon& ion : ions)
    {
        ionic_dipole += electron_charge_ry * ion.valence_charge
                        * ion.cartesian_position_bohr[cartesian_axis - 1];
    }
    return ionic_dipole;
}

FiniteFieldCartesianPolarization finite_field_cartesian_polarization(
    const std::array<FiniteFieldPolarization, 3>& lattice_polarizations,
    const std::array<std::array<double, 3>, 3>& lattice_unit_vectors,
    const std::vector<FiniteFieldIon>& ions,
    const double cell_volume_bohr3)
{
    if (!std::isfinite(cell_volume_bohr3) || cell_volume_bohr3 <= 0.0)
    {
        throw std::invalid_argument(
            "finite-field cell volume must be finite and positive");
    }

    FiniteFieldCartesianPolarization result;
    for (int direction = 0; direction < 3; ++direction)
    {
        for (int component = 0; component < 3; ++component)
        {
            result.electronic[component]
                += lattice_polarizations[direction].electronic_dipole
                   * lattice_unit_vectors[direction][component];
        }
    }
    for (const FiniteFieldIon& ion : ions)
    {
        for (int component = 0; component < 3; ++component)
        {
            result.ionic[component] += electron_charge_ry * ion.valence_charge
                                       * ion.cartesian_position_bohr[component];
        }
    }
    for (int component = 0; component < 3; ++component)
    {
        result.electronic[component] /= cell_volume_bohr3;
        result.ionic[component] /= cell_volume_bohr3;
        result.total[component] = result.electronic[component]
                                  + result.ionic[component];
    }
    return result;
}

FiniteFieldCartesianPolarization finite_field_legacy_reported_polarization(
    const std::array<FiniteFieldPolarization, 3>& lattice_polarizations,
    const std::array<std::array<double, 3>, 3>& lattice_unit_vectors,
    const std::vector<FiniteFieldIon>& ions,
    const double cell_volume_bohr3)
{
    if (!std::isfinite(cell_volume_bohr3) || cell_volume_bohr3 <= 0.0)
    {
        throw std::invalid_argument(
            "finite-field cell volume must be finite and positive");
    }

    FiniteFieldCartesianPolarization result;
    for (int direction = 0; direction < 3; ++direction)
    {
        const FiniteFieldPolarization constrained
            = finite_field_legacy_principal_branch(
                lattice_polarizations[direction], direction + 1, ions);
        for (int component = 0; component < 3; ++component)
        {
            result.electronic[component]
                += constrained.electronic_dipole
                   * lattice_unit_vectors[direction][component];
            result.ionic[component]
                += constrained.ionic_dipole
                   * lattice_unit_vectors[direction][component];
        }
    }
    for (int component = 0; component < 3; ++component)
    {
        result.electronic[component] /= cell_volume_bohr3;
        result.ionic[component] /= cell_volume_bohr3;
        result.total[component] = result.electronic[component]
                                  + result.ionic[component];
    }
    return result;
}

FiniteFieldCartesianPolarization finite_field_physical_polarization(
    const FiniteFieldCartesianPolarization& polarization)
{
    // The finite-field functional retains QE's Ry-a.u. convention internally
    // (electron charge = sqrt(e2)).  Its public polarization vector is
    // reported in physical elementary-charge units instead.
    const double inverse_electron_charge_ry = 1.0 / electron_charge_ry;
    FiniteFieldCartesianPolarization result;
    for (int component = 0; component < 3; ++component)
    {
        result.electronic[component]
            = polarization.electronic[component] * inverse_electron_charge_ry;
        result.ionic[component]
            = polarization.ionic[component] * inverse_electron_charge_ry;
        result.total[component]
            = polarization.total[component] * inverse_electron_charge_ry;
    }
    return result;
}

FiniteFieldElectricEnthalpy FiniteFieldPolarizationBranch::update(
    const FiniteFieldPolarization& principal,
    const double field_ry_au)
{
    if (principal.polarization_quantum <= 0.0)
    {
        throw std::invalid_argument("finite-field polarization quantum must be positive");
    }

    // Keep the electronic principal value in the old nspin=1 Berry-module
    // interval from the very first finite-field evaluation.  Ionic wrapping
    // is intentionally not used in this SCF path: the electric enthalpy must
    // retain the continuous ionic representative.
    FiniteFieldPolarization electronic_principal = principal;
    electronic_principal.electronic_dipole
        -= electronic_principal.polarization_quantum
           * std::round(electronic_principal.electronic_dipole
                        / electronic_principal.polarization_quantum);

    if (this->initialized)
    {
        const double principal_change
            = electronic_principal.electronic_dipole
              - this->previous_principal_dipole;
        const double half_quantum
            = 0.5 * electronic_principal.polarization_quantum;
        if (principal_change < -half_quantum)
        {
            this->branch_offset += electronic_principal.polarization_quantum;
        }
        else if (principal_change > half_quantum)
        {
            this->branch_offset -= electronic_principal.polarization_quantum;
        }
    }
    else
    {
        this->initialized = true;
    }
    this->previous_principal_dipole = electronic_principal.electronic_dipole;

    FiniteFieldElectricEnthalpy result;
    result.principal = electronic_principal;
    result.branch_offset = this->branch_offset;
    result.continuous_electronic_dipole
        = electronic_principal.electronic_dipole + this->branch_offset;
    result.electric_enthalpy
        = -field_ry_au
          * (result.continuous_electronic_dipole + principal.ionic_dipole);
    return result;
}

FiniteFieldPolarizationBranchState FiniteFieldPolarizationBranch::state() const
{
    FiniteFieldPolarizationBranchState result;
    result.initialized = this->initialized;
    result.previous_principal_dipole = this->previous_principal_dipole;
    result.branch_offset = this->branch_offset;
    return result;
}

void FiniteFieldPolarizationBranch::restore(
    const FiniteFieldPolarizationBranchState& state,
    const double expected_polarization_quantum)
{
    if (expected_polarization_quantum <= 0.0)
    {
        throw std::invalid_argument("finite-field polarization quantum must be positive");
    }
    if (!std::isfinite(state.previous_principal_dipole)
        || !std::isfinite(state.branch_offset))
    {
        throw std::invalid_argument("finite-field polarization branch state is not finite");
    }
    if (!state.initialized
        && (state.previous_principal_dipole != 0.0
            || state.branch_offset != 0.0))
    {
        throw std::invalid_argument("uninitialized finite-field branch state must be zero");
    }
    if (state.initialized)
    {
        const double nearest_quantum
            = std::round(state.branch_offset / expected_polarization_quantum);
        const double residual
            = state.branch_offset
              - nearest_quantum * expected_polarization_quantum;
        const double tolerance
            = 1.0e-10 * std::max(1.0, expected_polarization_quantum);
        if (std::abs(residual) > tolerance)
        {
            throw std::invalid_argument(
                "finite-field branch offset is incompatible with polarization quantum");
        }
    }

    this->initialized = state.initialized;
    // Phase files written before principal-branch normalization may retain a
    // valid but unwrapped previous value.  Normalize that representation while
    // compensating its offset, so the continuous enthalpy branch is unchanged.
    const double previous_shift = expected_polarization_quantum
                                  * std::round(
                                      state.previous_principal_dipole
                                      / expected_polarization_quantum);
    this->previous_principal_dipole
        = state.previous_principal_dipole - previous_shift;
    this->branch_offset = state.branch_offset + previous_shift;
}

void FiniteFieldPolarizationBranch::reset()
{
    this->initialized = false;
    this->previous_principal_dipole = 0.0;
    this->branch_offset = 0.0;
}

const char* finite_field_branch_state_filename()
{
    return "finite_field_branch_state.dat";
}

void validate_finite_field_branch_state(
    const FiniteFieldBranchState& state,
    const int expected_cartesian_axis,
    const std::array<bool, 3>& expected_active,
    const std::array<double, 3>& expected_lattice_period_bohr,
    const std::array<double, 3>& expected_polarization_quantum)
{
    if (expected_cartesian_axis < 1 || expected_cartesian_axis > 3
        || state.cartesian_axis != expected_cartesian_axis)
    {
        throw std::invalid_argument(
            "finite-field branch state has incompatible Cartesian axis");
    }
    for (int direction = 0; direction < 3; ++direction)
    {
        const FiniteFieldBranchDirectionState& direction_state
            = state.directions[direction];
        if (direction_state.active != expected_active[direction]
            || expected_lattice_period_bohr[direction] <= 0.0
            || expected_polarization_quantum[direction] <= 0.0)
        {
            throw std::invalid_argument(
                "finite-field branch state has incompatible active directions");
        }
        require_compatible(direction_state.lattice_period_bohr,
                           expected_lattice_period_bohr[direction],
                           "lattice period");
        require_compatible(direction_state.polarization_quantum,
                           expected_polarization_quantum[direction],
                           "polarization quantum");
        FiniteFieldPolarizationBranch branch;
        branch.restore(direction_state.branch,
                       expected_polarization_quantum[direction]);
        if (!direction_state.active && direction_state.branch.initialized)
        {
            throw std::invalid_argument(
                "inactive finite-field direction has initialized branch state");
        }
    }
}

void write_finite_field_branch_state(
    std::ostream& output,
    const FiniteFieldBranchState& state)
{
    if (!output)
    {
        throw std::runtime_error(
            "finite-field branch state output stream is not writable");
    }
    std::array<bool, 3> active;
    std::array<double, 3> periods;
    std::array<double, 3> quanta;
    for (int direction = 0; direction < 3; ++direction)
    {
        active[direction] = state.directions[direction].active;
        periods[direction] = state.directions[direction].lattice_period_bohr;
        quanta[direction] = state.directions[direction].polarization_quantum;
    }
    validate_finite_field_branch_state(
        state, state.cartesian_axis, active, periods, quanta);
    output << branch_format_header << " " << branch_format_version << "\n"
           << "cartesian_axis " << state.cartesian_axis << "\n"
           << std::setprecision(17);
    for (int direction = 0; direction < 3; ++direction)
    {
        const std::string prefix = "direction_" + std::to_string(direction + 1) + "_";
        const FiniteFieldBranchDirectionState& direction_state
            = state.directions[direction];
        output << prefix << "active " << (direction_state.active ? 1 : 0) << "\n"
               << prefix << "lattice_period_bohr "
               << direction_state.lattice_period_bohr << "\n"
               << prefix << "polarization_quantum "
               << direction_state.polarization_quantum << "\n"
               << prefix << "initialized "
               << (direction_state.branch.initialized ? 1 : 0) << "\n"
               << prefix << "previous_principal_dipole "
               << direction_state.branch.previous_principal_dipole << "\n"
               << prefix << "branch_offset "
               << direction_state.branch.branch_offset << "\n";
    }
    if (!output)
    {
        throw std::runtime_error("failed to write finite-field branch state");
    }
}

FiniteFieldBranchState read_finite_field_branch_state(std::istream& input)
{
    std::string header;
    int version = 0;
    if (!(input >> header >> version)
        || header != branch_format_header || version != branch_format_version)
    {
        throw std::runtime_error(
            "finite-field branch state has unsupported format");
    }
    std::string line;
    std::getline(input, line);
    std::map<std::string, std::string> values;
    while (std::getline(input, line))
    {
        if (line.empty())
        {
            continue;
        }
        std::istringstream parser(line);
        std::string key;
        std::string value;
        if (!(parser >> key >> value))
        {
            throw std::runtime_error(
                "finite-field branch state contains an invalid line");
        }
        parser >> std::ws;
        if (!parser.eof() || values.count(key) != 0)
        {
            throw std::runtime_error(
                "finite-field branch state contains invalid or duplicate keys");
        }
        values[key] = value;
    }
    if (!input.eof() || values.size() != 19)
    {
        throw std::runtime_error(
            "finite-field branch state contains unexpected keys");
    }

    FiniteFieldBranchState state;
    state.cartesian_axis = parse_value<int>(values, "cartesian_axis");
    std::array<bool, 3> active;
    std::array<double, 3> periods;
    std::array<double, 3> quanta;
    for (int direction = 0; direction < 3; ++direction)
    {
        const std::string prefix = "direction_" + std::to_string(direction + 1) + "_";
        const int active_value = parse_value<int>(values, prefix + "active");
        const int initialized = parse_value<int>(values, prefix + "initialized");
        if ((active_value != 0 && active_value != 1)
            || (initialized != 0 && initialized != 1))
        {
            throw std::runtime_error(
                "finite-field branch state Boolean flag is invalid");
        }
        FiniteFieldBranchDirectionState& direction_state
            = state.directions[direction];
        direction_state.active = active_value == 1;
        direction_state.lattice_period_bohr
            = parse_value<double>(values, prefix + "lattice_period_bohr");
        direction_state.polarization_quantum
            = parse_value<double>(values, prefix + "polarization_quantum");
        direction_state.branch.initialized = initialized == 1;
        direction_state.branch.previous_principal_dipole
            = parse_value<double>(values, prefix + "previous_principal_dipole");
        direction_state.branch.branch_offset
            = parse_value<double>(values, prefix + "branch_offset");
        active[direction] = direction_state.active;
        periods[direction] = direction_state.lattice_period_bohr;
        quanta[direction] = direction_state.polarization_quantum;
    }
    validate_finite_field_branch_state(
        state, state.cartesian_axis, active, periods, quanta);
    return state;
}

void write_finite_field_branch_state_file(
    const std::string& path,
    const FiniteFieldBranchState& state)
{
    const std::string temporary_path = path + ".tmp";
    {
        std::ofstream output(temporary_path.c_str());
        if (!output)
        {
            throw std::runtime_error(
                "cannot open finite-field branch state file for writing: "
                + temporary_path);
        }
        write_finite_field_branch_state(output, state);
        output.close();
        if (!output)
        {
            std::remove(temporary_path.c_str());
            throw std::runtime_error(
                "failed to close finite-field branch state file: "
                + temporary_path);
        }
    }
    if (std::rename(temporary_path.c_str(), path.c_str()) != 0)
    {
        std::remove(temporary_path.c_str());
        throw std::runtime_error(
            "cannot replace finite-field branch state file: " + path);
    }
}

FiniteFieldBranchState read_finite_field_branch_state_file(
    const std::string& path)
{
    std::ifstream input(path.c_str());
    if (!input)
    {
        throw std::runtime_error(
            "cannot open finite-field branch state file for reading: " + path);
    }
    return read_finite_field_branch_state(input);
}

} // namespace hamilt
