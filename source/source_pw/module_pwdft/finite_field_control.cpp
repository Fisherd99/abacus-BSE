#include "finite_field_control.h"

#include "source_hamilt/module_finite_field/finite_field_polarization.h"
#include "finite_field_pw.h"
#include "op_pw_finite_field.h"
#include "source_base/constants.h"
#include "source_base/parallel_common.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/kpoint_strings.h"
#include "source_cell/unitcell.h"
#include "source_psi/psi.h"

#include <cmath>
#include <exception>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace hamilt
{

namespace
{

class DirectionState
{
  public:
    using Wavefunctions = FiniteFieldPWController::Wavefunctions;

    DirectionState(const ModuleCell::KPointStrings& strings,
                   const ModulePW::PW_Basis_K& basis,
                   const int occupied_bands)
        : strings(strings),
          basis(basis),
          occupied_bands(occupied_bands)
    {
    }

    void prepare(const Wavefunctions& wavefunctions,
                 const std::complex<double>& factor,
                 FiniteFieldOperatorPW& finite_field_operator,
                 const bool append)
    {
        // Freeze the occupied manifold used by this Berry cycle. Every
        // lattice-direction term must see the same input wavefunctions.
        std::vector<FiniteFieldPreparedKPoint> prepared;
        prepare_finite_field_pw_data_from_abacus(this->strings,
                                                 this->basis,
                                                 wavefunctions,
                                                 this->occupied_bands,
                                                 prepared);
        if (prepared.size()
            != static_cast<std::size_t>(wavefunctions.get_nk()))
        {
            throw std::runtime_error(
                "finite-field PW preparation returned an invalid k-point count");
        }
        if (!append)
        {
            finite_field_operator.clear_kpoint_data();
        }
        for (int ik = 0; ik < wavefunctions.get_nk(); ++ik)
        {
            finite_field_operator.add_kpoint_data(
                ik,
                factor,
                prepared[ik].occupied,
                prepared[ik].dual_difference);
        }
    }

    const FiniteFieldElectricEnthalpy& evaluate(
        const Wavefunctions& wavefunctions,
        const double lattice_period_bohr,
        const std::vector<FiniteFieldIon>& ions,
        const double field_ry_au)
    {
        std::vector<FiniteFieldPreparedKPoint> prepared;
        prepare_finite_field_pw_data_from_abacus(this->strings,
                                                 this->basis,
                                                 wavefunctions,
                                                 this->occupied_bands,
                                                 prepared);
        if (prepared.size()
            != static_cast<std::size_t>(wavefunctions.get_nk()))
        {
            throw std::runtime_error(
                "finite-field polarization returned an invalid k-point count");
        }
        std::vector<std::complex<double>> determinants(prepared.size());
        for (std::size_t ik = 0; ik < prepared.size(); ++ik)
        {
            determinants[ik] = prepared[ik].forward_overlap_determinant;
        }
        const FiniteFieldPolarization principal
            = calculate_finite_field_polarization(this->strings,
                                                  determinants,
                                                  lattice_period_bohr,
                                                  2,
                                                  ions);
        this->enthalpy = this->branch.update(principal, field_ry_au);
        return this->enthalpy;
    }

    FiniteFieldPolarization principal(const Wavefunctions& wavefunctions,
                                      const double lattice_period_bohr,
                                      const std::vector<FiniteFieldIon>& ions) const
    {
        std::vector<FiniteFieldPreparedKPoint> prepared;
        prepare_finite_field_pw_data_from_abacus(this->strings,
                                                 this->basis,
                                                 wavefunctions,
                                                 this->occupied_bands,
                                                 prepared);
        std::vector<std::complex<double>> determinants(prepared.size());
        for (std::size_t ik = 0; ik < prepared.size(); ++ik)
        {
            determinants[ik] = prepared[ik].forward_overlap_determinant;
        }
        return finite_field_legacy_principal_branch(
            calculate_finite_field_polarization(this->strings,
                                                determinants,
                                                lattice_period_bohr,
                                                2,
                                                ions),
            this->strings.direction, ions);
    }

    FiniteFieldPolarizationBranchState branch_state() const
    {
        return this->branch.state();
    }

    void restore_branch(const FiniteFieldPolarizationBranchState& state,
                        const double polarization_quantum)
    {
        this->branch.restore(state, polarization_quantum);
    }

  private:
    ModuleCell::KPointStrings strings;
    const ModulePW::PW_Basis_K& basis;
    int occupied_bands = 0;
    FiniteFieldPolarizationBranch branch;
    FiniteFieldElectricEnthalpy enthalpy;
};

void broadcast_branch_state(FiniteFieldBranchState& state)
{
#ifdef __MPI
    Parallel_Common::bcast_int(state.cartesian_axis);
    for (int direction = 0; direction < 3; ++direction)
    {
        FiniteFieldBranchDirectionState& direction_state
            = state.directions[direction];
        Parallel_Common::bcast_bool(direction_state.active);
        Parallel_Common::bcast_double(direction_state.lattice_period_bohr);
        Parallel_Common::bcast_double(direction_state.polarization_quantum);
        Parallel_Common::bcast_bool(direction_state.branch.initialized);
        Parallel_Common::bcast_double(
            direction_state.branch.previous_principal_dipole);
        Parallel_Common::bcast_double(direction_state.branch.branch_offset);
    }
#else
    static_cast<void>(state);
#endif
}

void broadcast_io_status(bool& success, std::string& message)
{
#ifdef __MPI
    Parallel_Common::bcast_bool(success);
    if (!success)
    {
        Parallel_Common::bcast_string(message);
    }
#else
    static_cast<void>(success);
    static_cast<void>(message);
#endif
}

} // namespace

struct FiniteFieldPWController::Impl
{
    FiniteFieldPWConfig config;
    FiniteFieldOperatorPW* finite_field_operator = nullptr;
    std::array<std::unique_ptr<DirectionState>, 3> states;
    std::array<double, 3> periods_bohr{{0.0, 0.0, 0.0}};
    std::array<double, 3> projections{{0.0, 0.0, 0.0}};
    std::array<double, 3> component_amplitudes{{0.0, 0.0, 0.0}};
    std::array<bool, 3> active{{false, false, false}};
    std::array<std::array<double, 3>, 3> lattice_unit_vectors;
    std::array<std::complex<double>, 3> factors;
    std::vector<FiniteFieldIon> ions;

    std::array<double, 3> polarization_quanta() const
    {
        std::array<double, 3> quanta;
        for (int direction = 0; direction < 3; ++direction)
        {
            quanta[direction]
                = finite_field_polarization_quantum(this->periods_bohr[direction]);
        }
        return quanta;
    }

    void read_branch_state()
    {
        const std::string path
            = this->config.read_directory + finite_field_branch_state_filename();

        FiniteFieldBranchState branch_state;
        bool success = true;
        std::string message;
        if (this->config.root_rank)
        {
            try
            {
                branch_state = read_finite_field_branch_state_file(path);
                validate_finite_field_branch_state(
                    branch_state, this->config.cartesian_axis, this->active,
                    this->periods_bohr, this->polarization_quanta());
            }
            catch (const std::exception& error)
            {
                success = false;
                message = error.what();
            }
        }
        broadcast_io_status(success, message);
        if (!success)
        {
            throw std::runtime_error(message);
        }
        broadcast_branch_state(branch_state);
        const std::array<double, 3> quanta = this->polarization_quanta();
        for (int direction = 0; direction < 3; ++direction)
        {
            if (!this->active[direction]) continue;
            this->states[direction]->restore_branch(
                branch_state.directions[direction].branch, quanta[direction]);
        }
    }

    void write_branch_state() const
    {
        const std::string path
            = this->config.output_directory + finite_field_branch_state_filename();

        bool success = true;
        std::string message;
        if (this->config.root_rank)
        {
            FiniteFieldBranchState branch_state;
            branch_state.cartesian_axis = this->config.cartesian_axis;
            for (int direction = 0; direction < 3; ++direction)
            {
                FiniteFieldBranchDirectionState& direction_state
                    = branch_state.directions[direction];
                direction_state.active = this->active[direction];
                direction_state.lattice_period_bohr = this->periods_bohr[direction];
                direction_state.polarization_quantum
                    = finite_field_polarization_quantum(
                        this->periods_bohr[direction]);
                if (this->active[direction])
                {
                    direction_state.branch
                        = this->states[direction]->branch_state();
                }
            }
            try
            {
                write_finite_field_branch_state_file(path, branch_state);
            }
            catch (const std::exception& error)
            {
                success = false;
                message = error.what();
            }
        }
        broadcast_io_status(success, message);
        if (!success)
        {
            throw std::runtime_error(message);
        }
    }
};

FiniteFieldPWController::FiniteFieldPWController()
    : impl(new Impl())
{
}

FiniteFieldPWController::~FiniteFieldPWController() = default;

void FiniteFieldPWController::configure(
    const UnitCell& ucell,
    const ModulePW::PW_Basis_K& basis,
    FiniteFieldOperatorPW& finite_field_operator,
    const FiniteFieldPWConfig& config,
    std::ostream& log)
{
    if (config.cartesian_axis < 1 || config.cartesian_axis > 3
        || config.berry_cycles < 1 || config.occupied_bands < 1)
    {
        throw std::invalid_argument("finite-field controller configuration is invalid");
    }

    this->impl.reset(new Impl());
    this->impl->config = config;
    this->impl->finite_field_operator = &finite_field_operator;
    this->impl->factors.fill(std::complex<double>(0.0, 0.0));

    const std::array<ModuleBase::Vector3<double>, 3> lattice_vectors
        = {{ucell.a1, ucell.a2, ucell.a3}};
    const FiniteFieldCartesianGeometry geometry
        = build_finite_field_cartesian_geometry(lattice_vectors,
                                                config.cartesian_axis);

    // Match QE efield_cart: a Cartesian field is represented by its signed
    // projections onto all direct-lattice string directions.
    for (int direction = 0; direction < 3; ++direction)
    {
        const double projection = geometry.cartesian_projections[direction];
        this->impl->projections[direction] = projection;
        this->impl->periods_bohr[direction]
            = geometry.lattice_periods[direction] * ucell.lat0;
        this->impl->lattice_unit_vectors[direction]
            = {{lattice_vectors[direction].x / geometry.lattice_periods[direction],
                lattice_vectors[direction].y / geometry.lattice_periods[direction],
                lattice_vectors[direction].z / geometry.lattice_periods[direction]}};
        // Polarization reporting needs the Berry phase on all three
        // lattice strings, not only those coupled to the applied field.
        if (config.mesh[direction] >= 1)
        {
            const ModuleCell::KPointStrings strings
                = ModuleCell::build_periodic_kpoint_strings(config.mesh,
                                                            direction + 1,
                                                            1);
            this->impl->states[direction].reset(
                new DirectionState(strings, basis, config.occupied_bands));
        }
        if (projection == 0.0)
        {
            continue;
        }
        if (config.mesh[direction] < 1)
        {
            throw std::invalid_argument(
                "finite_field needs at least two k points along every "
                "lattice direction with a nonzero Cartesian field projection");
        }

        if (!this->impl->states[direction])
        {
            throw std::logic_error(
                "finite-field active direction has no k-string state");
        }
        this->impl->active[direction] = true;
        const double component_amplitude = config.amplitude * projection;
        this->impl->component_amplitudes[direction] = component_amplitude;
        this->impl->factors[direction]
            = finite_field_coupling(component_amplitude,
                                    this->impl->periods_bohr[direction],
                                    config.mesh[direction]);
    }
    for (int it = 0; it < ucell.ntype; ++it)
    {
        for (int ia = 0; ia < ucell.atoms[it].na; ++ia)
        {
            FiniteFieldIon ion;
            ion.valence_charge = ucell.atoms[it].ncpp.zv;
            ion.direct_position = {{ucell.atoms[it].taud[ia].x,
                                    ucell.atoms[it].taud[ia].y,
                                    ucell.atoms[it].taud[ia].z}};
            const ModuleBase::Vector3<double> cartesian_position
                = ucell.a1 * ion.direct_position[0]
                  + ucell.a2 * ion.direct_position[1]
                  + ucell.a3 * ion.direct_position[2];
            ion.cartesian_position_bohr
                = {{cartesian_position.x * ucell.lat0,
                    cartesian_position.y * ucell.lat0,
                    cartesian_position.z * ucell.lat0}};
            this->impl->ions.push_back(ion);
        }
    }

    if (config.branch_io == "read")
    {
        this->impl->read_branch_state();
    }

    log << " PERIODIC FINITE FIELD: cartesian_axis="
        << "xyz"[config.cartesian_axis - 1]
        << " direction=" << config.cartesian_axis
        << " amplitude=" << config.amplitude << " Ry a.u."
        << " nberrycyc=" << config.berry_cycles
        << " branch_io=" << config.branch_io
        << " occupied_bands=" << config.occupied_bands
        << " lattice_projections="
        << this->impl->projections[0] << ","
        << this->impl->projections[1] << ","
        << this->impl->projections[2] << std::endl;
}

void FiniteFieldPWController::prepare_cycle(
    const Wavefunctions& wavefunctions)
{
    bool append_operator_data = false;
    for (int direction = 0; direction < 3; ++direction)
    {
        if (!this->impl->active[direction])
        {
            continue;
        }
        this->impl->states[direction]->prepare(
            wavefunctions,
            this->impl->factors[direction],
            *this->impl->finite_field_operator,
            append_operator_data);
        append_operator_data = true;
    }
}

double FiniteFieldPWController::evaluate(
    const Wavefunctions& wavefunctions,
    std::ostream& log)
{
    double projected_phase_diagnostic = 0.0;
    double cartesian_electronic_dipole = 0.0;
    double cartesian_branch_offset = 0.0;
    double electronic_enthalpy = 0.0;
    const double cartesian_ionic_dipole
        = finite_field_cartesian_ionic_dipole(
            this->impl->ions,
            this->impl->config.cartesian_axis);

    for (int direction = 0; direction < 3; ++direction)
    {
        if (!this->impl->active[direction])
        {
            continue;
        }
        const FiniteFieldElectricEnthalpy& component
            = this->impl->states[direction]->evaluate(
                wavefunctions,
                this->impl->periods_bohr[direction],
                this->impl->ions,
                this->impl->component_amplitudes[direction]);
        const double projection = this->impl->projections[direction];
        projected_phase_diagnostic
            += projection * component.principal.berry_phase;
        cartesian_electronic_dipole
            += projection * component.principal.electronic_dipole;
        cartesian_branch_offset += projection * component.branch_offset;
        const double component_enthalpy
            = -this->impl->component_amplitudes[direction]
              * component.continuous_electronic_dipole;
        electronic_enthalpy += component_enthalpy;

        log << std::setprecision(15)
            << " Finite-field lattice direction=" << direction + 1
            << " projection=" << projection
            << " component_amplitude="
            << this->impl->component_amplitudes[direction]
            << " berry_phase=" << component.principal.berry_phase
            << " electronic_dipole="
            << component.principal.electronic_dipole
            << " ionic_dipole=" << component.principal.ionic_dipole
            << " branch_offset=" << component.branch_offset
            << " E_electronic_component=" << component_enthalpy
            << " Ry" << std::setprecision(6) << std::endl;
    }

    const double electric_enthalpy
        = electronic_enthalpy
          - this->impl->config.amplitude * cartesian_ionic_dipole;

    bool full_polarization_available = this->impl->config.cell_volume_bohr3 > 0.0;
    std::array<FiniteFieldPolarization, 3> lattice_polarizations;
    for (int direction = 0; direction < 3; ++direction)
    {
        if (!this->impl->states[direction])
        {
            full_polarization_available = false;
            continue;
        }
        lattice_polarizations[direction]
            = this->impl->states[direction]->principal(
                wavefunctions, this->impl->periods_bohr[direction], this->impl->ions);
    }
    if (full_polarization_available)
    {
        const FiniteFieldCartesianPolarization polarization_ry
            = finite_field_legacy_reported_polarization(
                lattice_polarizations, this->impl->lattice_unit_vectors,
                this->impl->ions, this->impl->config.cell_volume_bohr3);
        const FiniteFieldCartesianPolarization polarization
            = finite_field_physical_polarization(polarization_ry);
        log << std::setprecision(15)
            << " Finite-field polarization_e_per_bohr2 electronic="
            << polarization.electronic[0] << " "
            << polarization.electronic[1] << " "
            << polarization.electronic[2]
            << " total="
            << polarization.total[0] << " "
            << polarization.total[1] << " "
            << polarization.total[2]
            << std::setprecision(6) << std::endl;
    }
    else
    {
        log << " Finite-field polarization_e_per_bohr2 unavailable: "
            << "each lattice direction needs at least one k point" << std::endl;
    }
    if (this->impl->config.branch_io == "write")
    {
        this->impl->write_branch_state();
    }

    log << std::setprecision(15)
        << " Finite-field projected_phase_diagnostic="
        << projected_phase_diagnostic
        << " electronic_dipole=" << cartesian_electronic_dipole
        << " ionic_dipole=" << cartesian_ionic_dipole
        << " branch_offset=" << cartesian_branch_offset
        << " E_finite_field=" << electric_enthalpy
        << " Ry" << std::setprecision(6) << std::endl;
    return electric_enthalpy;
}

int FiniteFieldPWController::berry_cycles() const
{
    return this->impl->config.berry_cycles;
}

} // namespace hamilt
