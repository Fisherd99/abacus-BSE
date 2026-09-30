#include "finite_field_lcao.h"

#include "hamilt_lcao.h"
#include "module_finite_field/fr_overlap.h"
#include "source_base/constants.h"
#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/matrix.h"
#include "source_base/parallel_common.h"
#include "source_base/module_external/scalapack_connector.h"
#include "source_base/timer.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_cell/klist.h"
#include "source_cell/kpoint_strings.h"
#include "source_cell/unitcell.h"
#include "source_hamilt/module_finite_field/finite_field_matrix.h"
#include "source_hamilt/module_finite_field/finite_field_polarization.h"
#include "source_hamilt/module_hcontainer/hcontainer_funcs.h"
#include "source_psi/psi.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <vector>

#ifdef __MPI
#include <mpi.h>
#endif

namespace hamilt
{
namespace
{
using Complex = std::complex<double>;

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
    if (!success) Parallel_Common::bcast_string(message);
#else
    static_cast<void>(success);
    static_cast<void>(message);
#endif
}

std::vector<std::vector<Complex>> all_occupied_states(
    const FiniteFieldLCAOController::Wavefunctions& wavefunctions,
    const int nk,
    const Parallel_Orbitals& parallel_orbitals,
    const int occupied_bands)
{
    ModuleBase::timer::start("FiniteFieldLCAO", "occupied_gather");
    const int basis_size = parallel_orbitals.get_global_row_size();
    const std::size_t stride = static_cast<std::size_t>(basis_size) * occupied_bands;
    std::vector<Complex> packed(static_cast<std::size_t>(nk) * stride,
                                Complex(0.0, 0.0));
    for (int ik = 0; ik < nk; ++ik)
    {
        for (int local_band = 0; local_band < wavefunctions.get_nbands(); ++local_band)
        {
            const int band = parallel_orbitals.local2global_col(local_band);
            if (band >= occupied_bands) continue;
            for (int local_basis = 0; local_basis < wavefunctions.get_nbasis(); ++local_basis)
            {
                const int basis = parallel_orbitals.local2global_row(local_basis);
                packed[static_cast<std::size_t>(ik) * stride
                       + basis + basis_size * band]
                    = wavefunctions(ik, local_band, local_basis);
            }
        }
    }
#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE, packed.data(), packed.size(),
                  MPI_DOUBLE_COMPLEX, MPI_SUM, parallel_orbitals.comm());
#endif
    std::vector<std::vector<Complex>> result(nk);
    for (int ik = 0; ik < nk; ++ik)
    {
        const auto begin = packed.begin() + static_cast<std::size_t>(ik) * stride;
        result[ik].assign(begin, begin + stride);
    }
    ModuleBase::timer::end("FiniteFieldLCAO", "occupied_gather");
    return result;
}

std::vector<Complex> distributed_matrix_to_owner(
    const Complex* local, const Parallel_Orbitals& parallel_orbitals,
    const int owner)
{
    const int rows = parallel_orbitals.get_global_row_size();
    const int columns = parallel_orbitals.get_global_col_size();
    std::vector<Complex> full(static_cast<std::size_t>(rows) * columns,
                              Complex(0.0, 0.0));
    for (int local_column = 0; local_column < parallel_orbitals.get_col_size();
         ++local_column)
    {
        const int column = parallel_orbitals.local2global_col(local_column);
        for (int local_row = 0; local_row < parallel_orbitals.get_row_size();
             ++local_row)
        {
            const int row = parallel_orbitals.local2global_row(local_row);
            full[row + rows * column]
                = local[local_row
                        + parallel_orbitals.get_row_size() * local_column];
        }
    }
#ifdef __MPI
    int rank = 0;
    MPI_Comm_rank(parallel_orbitals.comm(), &rank);
    if (rank == owner)
    {
        MPI_Reduce(MPI_IN_PLACE, full.data(), full.size(), MPI_DOUBLE_COMPLEX,
                   MPI_SUM, owner, parallel_orbitals.comm());
    }
    else
    {
        MPI_Reduce(full.data(), nullptr, full.size(), MPI_DOUBLE_COMPLEX,
                   MPI_SUM, owner, parallel_orbitals.comm());
        std::vector<Complex>().swap(full);
    }
#else
    (void)owner;
#endif
    return full;
}

struct DirectionData
{
    ModuleCell::KPointStrings strings;
    // AO links retain the Hamiltonian's two-dimensional block-cyclic layout.
    std::vector<std::vector<Complex>> plus_links;
    std::vector<std::vector<Complex>> minus_links;
    std::vector<std::vector<std::vector<Complex>>> plus_link_derivatives;
    std::vector<Complex> forward_determinants;
    std::vector<std::vector<Complex>> forward_inverses;
    FiniteFieldPolarizationBranch branch;
    double max_adjoint_error = 0.0;
    double max_interior_adjoint_error = 0.0;
    double max_closure_adjoint_error = 0.0;
};

std::array<int, 3> reciprocal_shift(const int direction, const int sign)
{
    std::array<int, 3> shift{{0, 0, 0}};
    shift[direction] = sign;
    return shift;
}

ModuleBase::Vector3<double> shifted_k(
    const ModuleBase::Vector3<double>& k, const std::array<int, 3>& shift)
{
    return ModuleBase::Vector3<double>(k.x + shift[0],
                                       k.y + shift[1],
                                       k.z + shift[2]);
}

void build_links_for_direction(
    DirectionData& data,
    const int direction,
    const UnitCell& ucell,
    const Grid_Driver& grid,
    const K_Vectors& kpoints,
    const LCAO_Orbitals& orbitals,
    const RadialCollection& radial_orbitals,
    const Parallel_Orbitals& parallel_orbitals,
    const bool calculate_forces,
    const std::string& overlap_backend,
    const int rayleigh_lmax)
{
    ModuleBase::timer::start("FiniteFieldLCAO", "ao_links");
    const int basis_size = parallel_orbitals.get_global_row_size();
    const int count = kpoints.get_nks();
    data.plus_links.assign(count, std::vector<Complex>());
    data.minus_links.assign(count, std::vector<Complex>());
    if (calculate_forces)
    {
        data.plus_link_derivatives.assign(
            3 * ucell.nat, std::vector<std::vector<Complex>>(count));
    }

    const double step
        = 1.0 / static_cast<double>(data.strings.points_per_string - 1);
    ModuleBase::Vector3<double> plus_step;
    if (direction == 0) plus_step.x = step;
    else if (direction == 1) plus_step.y = step;
    else plus_step.z = step;
    const ModuleBase::Vector3<double> plus_delta
        = (plus_step * ucell.G) * ucell.tpiba;
    const auto make_integral = [&](const ModuleBase::Vector3<double>& delta) {
        std::unique_ptr<FiniteFieldFROverlap<Complex>> integral(
            new FiniteFieldFROverlap<Complex>());
        if (overlap_backend == "taylor_first_order")
        {
            integral->set_first_order_parameters(
                &ucell, &orbitals, &grid, &parallel_orbitals, delta,
                calculate_forces);
        }
        else if (overlap_backend == "rayleigh_expansion")
        {
            integral->set_two_center_parameters(
                &ucell, &orbitals, &radial_orbitals, &grid,
                &parallel_orbitals, delta, rayleigh_lmax, 0, 0.0,
                calculate_forces);
        }
        else
        {
            integral->set_parameters(
                [delta](const ModuleBase::Vector3<double> r) {
                    return std::exp(-ModuleBase::IMAG_UNIT * (delta * r));
                },
                // The plane-wave phase has an infinite angular expansion.
                // 770 angular points converges semicore-rich links such as Mg.
                &ucell, &orbitals, &grid, &parallel_orbitals, 140, 770,
                calculate_forces, overlap_backend == "analytic_gradient",
                delta);
        }
        integral->calculate_FR();
        return integral;
    };
    const std::unique_ptr<FiniteFieldFROverlap<Complex>> plus_integral
        = make_integral(plus_delta);
    int communicator_size = 1;
#ifdef __MPI
    MPI_Comm_size(parallel_orbitals.comm(), &communicator_size);
#endif

    for (const std::vector<int>& string : data.strings.indices)
    {
        const int unique_points = data.strings.points_per_string - 1;
        for (int position = 0; position < unique_points; ++position)
        {
            const int current = string[position];
            const int plus = string[(position + 1) % unique_points];
            const std::array<int, 3> plus_shift
                = position == unique_points - 1
                      ? reciprocal_shift(direction, 1)
                      : std::array<int, 3>{{0, 0, 0}};

            const ModuleBase::Vector3<double> plus_k
                = shifted_k(kpoints.kvec_d[plus], plus_shift);
            const auto fold_link = [&](const hamilt::HContainer<Complex>& integral,
                                       const ModuleBase::Vector3<double>& neighbor_k) {
                std::vector<Complex> local_link(
                    parallel_orbitals.get_local_size(), Complex(0.0, 0.0));
                folding_HR(integral, local_link.data(), neighbor_k,
                           parallel_orbitals.get_row_size(), 1);
                return local_link;
            };
            data.plus_links[current]
                = fold_link(*plus_integral->get_FR_pointer(), plus_k);
        }
    }
    // Stream the real-space center derivatives one atom at a time.  Keeping
    // 3*nat HContainers alive makes the setup memory scale with system size
    // even though each folded derivative is immediately reduced to one owner.
    for (int atom = 0; calculate_forces && atom < ucell.nat; ++atom)
    {
        plus_integral->calculate_center_gradient(atom);
        for (const std::vector<int>& string : data.strings.indices)
        {
            const int unique_points = data.strings.points_per_string - 1;
            for (int position = 0; position < unique_points; ++position)
            {
                const int current = string[position];
                const int plus = string[(position + 1) % unique_points];
                const std::array<int, 3> plus_shift
                    = position == unique_points - 1
                          ? reciprocal_shift(direction, 1)
                          : std::array<int, 3>{{0, 0, 0}};
                const ModuleBase::Vector3<double> plus_k
                    = shifted_k(kpoints.kvec_d[plus], plus_shift);
                for (int component = 0; component < 3; ++component)
                {
                    const hamilt::HContainer<Complex>* derivative
                        = plus_integral->get_dFR_pointer(atom, component);
                    const int owner = (3 * atom + component) % communicator_size;
                    std::vector<Complex> local_link(
                        parallel_orbitals.get_local_size(), Complex(0.0, 0.0));
                    folding_HR(*derivative, local_link.data(), plus_k,
                               parallel_orbitals.get_row_size(), 1);
                    data.plus_link_derivatives[3 * atom + component][current]
                        = distributed_matrix_to_owner(
                            local_link.data(), parallel_orbitals, owner);
                }
            }
        }
    }

    // Construct the backward link from the adjoint forward link.  Computing
    // the +/- momentum-transfer integrals independently introduces a finite
    // quadrature mismatch and breaks the exact discrete identity
    // T_-(k) = T_+(k-dk)^H required by the Berry functional.
    for (const std::vector<int>& string : data.strings.indices)
    {
        const int unique_points = data.strings.points_per_string - 1;
        for (int position = 0; position < unique_points; ++position)
        {
            const int current = string[position];
            const int minus
                = string[(position + unique_points - 1) % unique_points];
            data.minus_links[current].assign(parallel_orbitals.get_local_size(),
                                             Complex(0.0, 0.0));
#ifdef __MPI
            ScalapackConnector::tranc(
                basis_size, basis_size, Complex(1.0, 0.0),
                data.plus_links[minus].data(), 1, 1,
                parallel_orbitals.get_desc(), Complex(0.0, 0.0),
                data.minus_links[current].data(), 1, 1,
                parallel_orbitals.get_desc());
#else
            for (int column = 0; column < basis_size; ++column)
            {
                for (int row = 0; row < basis_size; ++row)
                {
                    data.minus_links[current][row + basis_size * column]
                        = std::conj(data.plus_links[minus]
                                       [column + basis_size * row]);
                }
            }
#endif
        }
    }

    data.max_adjoint_error = 0.0;
    data.max_interior_adjoint_error = 0.0;
    data.max_closure_adjoint_error = 0.0;
    for (const std::vector<int>& string : data.strings.indices)
    {
        const int unique_points = data.strings.points_per_string - 1;
        for (int position = 0; position < unique_points; ++position)
        {
            const int current = string[position];
            const int plus = string[(position + 1) % unique_points];
            // The backward matrix was produced by PZTRANC from this exact
            // forward matrix, so its distributed adjoint error is identically
            // zero without another all-to-all transpose.
        }
    }
    ModuleBase::timer::end("FiniteFieldLCAO", "ao_links");
}

} // namespace

struct FiniteFieldLCAOController::Impl
{
    FiniteFieldLCAOConfig config;
    const UnitCell* ucell = nullptr;
    const Grid_Driver* grid = nullptr;
    const K_Vectors* kpoints = nullptr;
    const LCAO_Orbitals* orbitals = nullptr;
    const RadialCollection* radial_orbitals = nullptr;
    const Parallel_Orbitals* parallel_orbitals = nullptr;
    HamiltLCAO<Complex, double>* hamiltonian = nullptr;
    int basis_size = 0;
    std::array<std::unique_ptr<DirectionData>, 3> directions;
    std::array<double, 3> periods{{0.0, 0.0, 0.0}};
    std::array<double, 3> projections{{0.0, 0.0, 0.0}};
    std::array<double, 3> amplitudes{{0.0, 0.0, 0.0}};
    std::array<bool, 3> active{{false, false, false}};
    std::array<std::array<double, 3>, 3> lattice_unit_vectors;
    std::vector<FiniteFieldIon> ions;
    std::vector<std::vector<Complex>> final_states;
    Parallel_2D occupied_state_layout;
    Parallel_2D occupied_link_layout;
    bool prepared = false;
    bool final_cache_valid = false;
    double final_enthalpy = 0.0;

    std::unique_ptr<DirectionData> build_direction(const int direction,
                                                   const bool calculate_forces,
                                                   std::ostream& log) const
    {
        std::unique_ptr<DirectionData> data(new DirectionData());
        data->strings = ModuleCell::build_periodic_kpoint_strings(
            this->config.mesh, direction + 1, 1);
        build_links_for_direction(*data, direction, *this->ucell, *this->grid,
                                  *this->kpoints, *this->orbitals,
                                  *this->radial_orbitals,
                                  *this->parallel_orbitals, calculate_forces,
                                  this->config.overlap_backend,
                                  this->config.rayleigh_lmax);
        log << " Finite-field LCAO AO-link adjoint_error["
            << "xyz"[direction] << "]=" << std::setprecision(15)
            << data->max_adjoint_error << " interior="
            << data->max_interior_adjoint_error << " closure="
            << data->max_closure_adjoint_error << std::setprecision(6)
            << std::endl;
        return data;
    }

    std::array<double, 3> polarization_quanta() const
    {
        std::array<double, 3> quanta;
        for (int direction = 0; direction < 3; ++direction)
        {
            quanta[direction]
                = finite_field_polarization_quantum(this->periods[direction]);
        }
        return quanta;
    }

    void read_branch_state()
    {
        FiniteFieldBranchState state;
        bool success = true;
        std::string message;
        if (this->config.root_rank)
        {
            try
            {
                state = read_finite_field_branch_state_file(
                    this->config.read_directory
                    + finite_field_branch_state_filename());
                validate_finite_field_branch_state(
                    state, this->config.cartesian_axis, this->active,
                    this->periods, this->polarization_quanta());
            }
            catch (const std::exception& error)
            {
                success = false;
                message = error.what();
            }
        }
        broadcast_io_status(success, message);
        if (!success) throw std::runtime_error(message);
        broadcast_branch_state(state);
        const std::array<double, 3> quanta = this->polarization_quanta();
        for (int direction = 0; direction < 3; ++direction)
        {
            if (!this->active[direction]) continue;
            this->directions[direction]->branch.restore(
                state.directions[direction].branch, quanta[direction]);
        }
    }

    void write_branch_state() const
    {
        FiniteFieldBranchState state;
        state.cartesian_axis = this->config.cartesian_axis;
        for (int direction = 0; direction < 3; ++direction)
        {
            FiniteFieldBranchDirectionState& direction_state
                = state.directions[direction];
            direction_state.active = this->active[direction];
            direction_state.lattice_period_bohr = this->periods[direction];
            direction_state.polarization_quantum
                = finite_field_polarization_quantum(this->periods[direction]);
            if (this->active[direction])
            {
                direction_state.branch
                    = this->directions[direction]->branch.state();
            }
        }
        bool success = true;
        std::string message;
        if (this->config.root_rank)
        {
            try
            {
                write_finite_field_branch_state_file(
                    this->config.output_directory
                        + finite_field_branch_state_filename(),
                    state);
            }
            catch (const std::exception& error)
            {
                success = false;
                message = error.what();
            }
        }
        broadcast_io_status(success, message);
        if (!success) throw std::runtime_error(message);
    }

    void prepare_direction(DirectionData& direction,
                           const Wavefunctions& wavefunctions,
                           const Complex factor,
                           std::vector<std::vector<Complex>>* field_matrices)
    {
        ModuleBase::timer::start("FiniteFieldLCAO", "dense_links");
        direction.forward_determinants.assign(this->kpoints->get_nks(),
                                               Complex(0.0, 0.0));
        direction.forward_inverses.assign(this->kpoints->get_nks(),
                                           std::vector<Complex>());
        for (const std::vector<int>& string : direction.strings.indices)
        {
            const int unique_points = direction.strings.points_per_string - 1;
            for (int position = 0; position < unique_points; ++position)
            {
                const int current = string[position];
                const int plus = string[(position + 1) % unique_points];
                const int minus = string[(position + unique_points - 1) % unique_points];
                const Complex* current_states = &wavefunctions(current, 0, 0);
                const FiniteFieldDistributedLink plus_data
                    = finite_field_distributed_link(
                        current_states, direction.plus_links[current],
                        &wavefunctions(plus, 0, 0), this->basis_size,
                        this->config.occupied_bands, *this->parallel_orbitals,
                        this->occupied_state_layout, this->occupied_link_layout
#ifdef __MPI
                        , this->parallel_orbitals->desc_wfc
                        , this->parallel_orbitals->comm()
#endif
                        );
                const FiniteFieldDistributedLink minus_data
                    = finite_field_distributed_link(
                        current_states, direction.minus_links[current],
                        &wavefunctions(minus, 0, 0), this->basis_size,
                        this->config.occupied_bands, *this->parallel_orbitals,
                        this->occupied_state_layout, this->occupied_link_layout
#ifdef __MPI
                        , this->parallel_orbitals->desc_wfc
                        , this->parallel_orbitals->comm()
#endif
                        );
                direction.forward_determinants[current] = plus_data.determinant;
                direction.forward_inverses[current] = plus_data.inverse;
                if (field_matrices != nullptr)
                {
                    this->hamiltonian->updateSk(current, 1);
                    finite_field_distributed_hamiltonian(
                        this->hamiltonian->getSk(), current_states,
                        minus_data.local_dual, plus_data.local_dual,
                        this->basis_size, this->config.occupied_bands, factor,
                        *this->parallel_orbitals, this->occupied_state_layout
#ifdef __MPI
                        , this->parallel_orbitals->desc_wfc
#endif
                        ,
                        (*field_matrices)[current]);
                }
            }
        }
        ModuleBase::timer::end("FiniteFieldLCAO", "dense_links");
    }
};

FiniteFieldLCAOController::FiniteFieldLCAOController() : impl(new Impl()) {}
FiniteFieldLCAOController::~FiniteFieldLCAOController() = default;

void FiniteFieldLCAOController::configure(
    const UnitCell& ucell, const Grid_Driver& grid, const K_Vectors& kpoints,
    const LCAO_Orbitals& orbitals, const RadialCollection& radial_orbitals,
    const Parallel_Orbitals& parallel_orbitals,
    HamiltLCAO<Complex, double>& hamiltonian,
    const FiniteFieldLCAOConfig& config, std::ostream& log)
{
    if (config.cartesian_axis < 1 || config.cartesian_axis > 3
        || config.occupied_bands <= 0 || config.berry_cycles <= 0
        || config.rayleigh_lmax < 0)
    {
        throw std::invalid_argument("finite-field LCAO configuration is invalid");
    }
    this->impl.reset(new Impl());
    this->impl->config = config;
    this->impl->ucell = &ucell;
    this->impl->grid = &grid;
    this->impl->kpoints = &kpoints;
    this->impl->orbitals = &orbitals;
    this->impl->radial_orbitals = &radial_orbitals;
    this->impl->parallel_orbitals = &parallel_orbitals;
    this->impl->hamiltonian = &hamiltonian;
    this->impl->basis_size = parallel_orbitals.get_global_row_size();
#ifdef __MPI
    this->impl->occupied_state_layout.set(
        this->impl->basis_size, config.occupied_bands,
        parallel_orbitals.get_block_size(), parallel_orbitals.blacs_ctxt);
    this->impl->occupied_link_layout.set(
        config.occupied_bands, config.occupied_bands,
        parallel_orbitals.get_block_size(), parallel_orbitals.blacs_ctxt);
#else
    this->impl->occupied_state_layout.set_serial(this->impl->basis_size,
                                                 config.occupied_bands);
    this->impl->occupied_link_layout.set_serial(config.occupied_bands,
                                                config.occupied_bands);
#endif
    const std::array<ModuleBase::Vector3<double>, 3> vectors
        = {{ucell.a1, ucell.a2, ucell.a3}};
    const FiniteFieldCartesianGeometry geometry
        = build_finite_field_cartesian_geometry(vectors, config.cartesian_axis);
    for (int direction = 0; direction < 3; ++direction)
    {
        this->impl->periods[direction]
            = geometry.lattice_periods[direction] * ucell.lat0;
        this->impl->projections[direction]
            = geometry.cartesian_projections[direction];
        this->impl->amplitudes[direction]
            = config.amplitude * this->impl->projections[direction];
        this->impl->lattice_unit_vectors[direction]
            = {{vectors[direction].x / geometry.lattice_periods[direction],
                vectors[direction].y / geometry.lattice_periods[direction],
                vectors[direction].z / geometry.lattice_periods[direction]}};
        const FiniteFieldLCAODirectionRole role
            = finite_field_lcao_direction_role(
                this->impl->projections[direction], config.calculate_forces);
        this->impl->active[direction] = role.couple_field;
        if (!role.couple_field) continue;
        this->impl->directions[direction]
            = this->impl->build_direction(direction,
                                          role.calculate_force_derivatives,
                                          log);
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
            const ModuleBase::Vector3<double> position
                = vectors[0] * ion.direct_position[0]
                  + vectors[1] * ion.direct_position[1]
                  + vectors[2] * ion.direct_position[2];
            ion.cartesian_position_bohr
                = {{position.x * ucell.lat0, position.y * ucell.lat0,
                    position.z * ucell.lat0}};
            this->impl->ions.push_back(ion);
        }
    }
    if (config.branch_io == "read") this->impl->read_branch_state();
    log << " PERIODIC FINITE FIELD (LCAO stage 1): cartesian_axis="
        << "xyz"[config.cartesian_axis - 1] << " amplitude="
        << config.amplitude << " Ry a.u. occupied_bands="
        << config.occupied_bands << " distributed_Berry=exact_exp(-i*dk*r)"
        << " overlap_backend="
        << config.overlap_backend
        << " rayleigh_lmax=" << config.rayleigh_lmax
        << " branch_io=" << config.branch_io
        << std::endl;
}

void FiniteFieldLCAOController::prepare_cycle(const Wavefunctions& wavefunctions)
{
    std::vector<std::vector<Complex>> matrices(
        this->impl->kpoints->get_nks(),
        std::vector<Complex>(this->impl->parallel_orbitals->get_local_size(),
                             Complex(0.0, 0.0)));
    for (int direction = 0; direction < 3; ++direction)
    {
        if (!this->impl->active[direction]) continue;
        this->impl->prepare_direction(
            *this->impl->directions[direction], wavefunctions,
            finite_field_coupling(this->impl->amplitudes[direction],
                                  this->impl->periods[direction],
                                  this->impl->config.mesh[direction]),
            &matrices);
    }
    this->impl->hamiltonian->set_finite_field_hk(std::move(matrices));
    this->impl->prepared = true;
    this->impl->final_cache_valid = false;
}

double FiniteFieldLCAOController::evaluate(
    const Wavefunctions& wavefunctions, std::ostream& log)
{
    double electronic_enthalpy = 0.0;
    for (int direction = 0; direction < 3; ++direction)
    {
        if (!this->impl->directions[direction]) continue;
        DirectionData& data = *this->impl->directions[direction];
        this->impl->prepare_direction(data, wavefunctions,
                                      Complex(0.0, 0.0), nullptr);
        const FiniteFieldPolarization lattice
            = calculate_finite_field_polarization(
            data.strings, data.forward_determinants, this->impl->periods[direction],
            2, this->impl->ions);
        const FiniteFieldElectricEnthalpy component
            = data.branch.update(lattice, this->impl->amplitudes[direction]);
        electronic_enthalpy -= this->impl->amplitudes[direction]
                               * component.continuous_electronic_dipole;
    }
    const double ionic_dipole = finite_field_cartesian_ionic_dipole(
        this->impl->ions, this->impl->config.cartesian_axis);
    const double enthalpy = electronic_enthalpy
                           - this->impl->config.amplitude * ionic_dipole;
    this->impl->final_enthalpy = enthalpy;
    if (this->impl->config.branch_io == "write")
    {
        this->impl->write_branch_state();
    }
    this->impl->final_cache_valid = false;
    return enthalpy;
}

void FiniteFieldLCAOController::report_polarization(
    const Wavefunctions& wavefunctions, std::ostream& log)
{
    std::array<FiniteFieldPolarization, 3> reported_lattice{};
    for (int direction = 0; direction < 3; ++direction)
    {
        std::unique_ptr<DirectionData> temporary;
        DirectionData* data = this->impl->directions[direction].get();
        if (data == nullptr)
        {
            temporary = this->impl->build_direction(direction, false, log);
            data = temporary.get();
        }
        this->impl->prepare_direction(*data, wavefunctions,
                                      Complex(0.0, 0.0),
                                      nullptr);
        const FiniteFieldPolarization lattice
            = calculate_finite_field_polarization(
                data->strings, data->forward_determinants,
                this->impl->periods[direction], 2, this->impl->ions);
        reported_lattice[direction] = finite_field_legacy_principal_branch(
            lattice, direction + 1, this->impl->ions);
    }
    const FiniteFieldCartesianPolarization physical
        = finite_field_physical_polarization(finite_field_cartesian_polarization(
            reported_lattice, this->impl->lattice_unit_vectors, this->impl->ions,
            this->impl->config.cell_volume_bohr3));
    log << std::setprecision(15)
        << " Finite-field LCAO polarization_e_per_bohr2 electronic="
        << physical.electronic[0] << " " << physical.electronic[1] << " "
        << physical.electronic[2] << " total=" << physical.total[0] << " "
        << physical.total[1] << " " << physical.total[2]
        << " E_finite_field=" << this->impl->final_enthalpy << " Ry"
        << std::setprecision(6) << std::endl;
    if (this->impl->config.calculate_forces)
    {
        this->impl->final_states = all_occupied_states(
            wavefunctions, this->impl->kpoints->get_nks(),
            *this->impl->parallel_orbitals, this->impl->config.occupied_bands);
        this->impl->final_cache_valid = true;
    }
}

int FiniteFieldLCAOController::berry_cycles() const
{
    return this->impl->config.berry_cycles;
}

bool FiniteFieldLCAOController::needs_bootstrap() const
{
    return !this->impl->prepared;
}

void FiniteFieldLCAOController::add_force(
    const Wavefunctions& wavefunctions, ModuleBase::matrix& force,
    std::ostream& log)
{
    if (this->impl->config.amplitude == 0.0)
    {
        log << " Finite-field LCAO force skipped for zero field" << std::endl;
        return;
    }
    if (!this->impl->config.calculate_forces)
    {
        throw std::logic_error(
            "finite-field LCAO force derivatives were not initialized");
    }
    if (force.nr != this->impl->ucell->nat || force.nc != 3)
    {
        throw std::invalid_argument(
            "finite-field LCAO force matrix has inconsistent dimensions");
    }
    if (!this->impl->final_cache_valid)
    {
        this->impl->final_states = all_occupied_states(
            wavefunctions, this->impl->kpoints->get_nks(),
            *this->impl->parallel_orbitals, this->impl->config.occupied_bands);
        for (int direction = 0; direction < 3; ++direction)
        {
            if (!this->impl->active[direction]) continue;
            this->impl->prepare_direction(
                *this->impl->directions[direction], wavefunctions,
                Complex(0.0, 0.0), nullptr);
        }
        this->impl->final_cache_valid = true;
    }
    ModuleBase::timer::start("FiniteFieldLCAO", "force_contraction");
    ModuleBase::matrix berry_force(this->impl->ucell->nat, 3);
    const int spin_degeneracy = 2;
    const double pi = std::acos(-1.0);
    int communicator_rank = 0;
    int communicator_size = 1;
#ifdef __MPI
    MPI_Comm_rank(this->impl->parallel_orbitals->comm(), &communicator_rank);
    MPI_Comm_size(this->impl->parallel_orbitals->comm(), &communicator_size);
#endif

    for (int direction = 0; direction < 3; ++direction)
    {
        if (!this->impl->active[direction]) continue;
        DirectionData& data = *this->impl->directions[direction];
        const double coefficient
            = this->impl->amplitudes[direction] * spin_degeneracy
              * std::sqrt(2.0) * this->impl->periods[direction]
              / (2.0 * pi * data.strings.indices.size());
        for (const std::vector<int>& string : data.strings.indices)
        {
            const int unique_points = data.strings.points_per_string - 1;
            for (int position = 0; position < unique_points; ++position)
            {
                const int current = string[position];
                const int plus = string[(position + 1) % unique_points];
                const std::vector<Complex>& current_states
                    = this->impl->final_states[current];
                const std::vector<Complex>& neighbor_states
                    = this->impl->final_states[plus];
                for (int atom = 0; atom < this->impl->ucell->nat; ++atom)
                {
                    for (int component = 0; component < 3; ++component)
                    {
                        if ((3 * atom + component) % communicator_size
                            != communicator_rank)
                        {
                            continue;
                        }
                        const Complex derivative
                            = finite_field_dense_logdet_derivative(
                                current_states,
                                data.plus_link_derivatives[3 * atom + component][current],
                                neighbor_states, data.forward_inverses[current],
                                this->impl->basis_size,
                                this->impl->config.occupied_bands);
                        berry_force(atom, component)
                            += coefficient * derivative.imag();
                    }
                }
            }
        }
    }
#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE, berry_force.c, berry_force.nr * berry_force.nc,
                  MPI_DOUBLE, MPI_SUM, this->impl->parallel_orbitals->comm());
#endif

    const int field_component = this->impl->config.cartesian_axis - 1;
    int atom = 0;
    for (int it = 0; it < this->impl->ucell->ntype; ++it)
    {
        for (int ia = 0; ia < this->impl->ucell->atoms[it].na; ++ia, ++atom)
        {
            berry_force(atom, field_component)
                += finite_field_ionic_force(
                    this->impl->config.amplitude,
                    this->impl->ucell->atoms[it].ncpp.zv);
        }
    }
    for (int iat = 0; iat < force.nr; ++iat)
    {
        for (int component = 0; component < 3; ++component)
        {
            force(iat, component) += berry_force(iat, component);
        }
    }
    log << " Finite-field LCAO force added (Ry/bohr): Berry-basis plus ionic"
        << std::endl;
    ModuleBase::timer::end("FiniteFieldLCAO", "force_contraction");
}

} // namespace hamilt
