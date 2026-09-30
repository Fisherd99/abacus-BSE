#include "finite_field_pw.h"

#include "source_base/module_external/blas_connector.h"
#include "source_base/module_external/lapack_connector.h"
#include "source_base/parallel_comm.h"
#include "source_base/parallel_global.h"
#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_psi/psi.h"

#include <cmath>
#include <map>
#include <stdexcept>

namespace hamilt
{

namespace
{

std::size_t matrix_size(const int occupied_bands)
{
    return static_cast<std::size_t>(occupied_bands) * occupied_bands;
}

std::size_t state_size(const int stride, const int occupied_bands)
{
    return static_cast<std::size_t>(stride) * occupied_bands;
}

void reduce_in_pool(std::complex<double>* values, const int count)
{
    Parallel_Reduce::reduce_pool(values, count);
}

int integer_component(const double component)
{
    const double rounded = std::round(component);
    if (std::abs(component - rounded) > 1.0e-10)
    {
        throw std::runtime_error("finite-field PW G label is not integral");
    }
    return static_cast<int>(rounded);
}

FiniteFieldGLabel closure_shift(const int direction, const int sign)
{
    FiniteFieldGLabel shift{{0, 0, 0}};
    shift[direction - 1] = sign;
    return shift;
}

FiniteFieldGLabel shifted_label(const FiniteFieldGLabel& label,
                                const FiniteFieldGLabel& shift)
{
    return FiniteFieldGLabel{{label[0] + shift[0],
                              label[1] + shift[1],
                              label[2] + shift[2]}};
}

std::size_t grid_size(const std::array<int, 3>& reciprocal_grid)
{
    std::size_t size = 1;
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        if (reciprocal_grid[dimension] <= 0)
        {
            throw std::invalid_argument(
                "finite-field reciprocal grid dimensions must be positive");
        }
        size *= reciprocal_grid[dimension];
    }
    return size;
}

int centered_grid_index(const int label, const int dimension)
{
    const int lower_bound = -(dimension - 1) / 2;
    const int upper_bound = dimension / 2;
    if (label < lower_bound || label > upper_bound)
    {
        return -1;
    }
    return label < 0 ? label + dimension : label;
}

int linear_grid_index(const FiniteFieldGLabel& label,
                      const std::array<int, 3>& reciprocal_grid)
{
    const int ix = centered_grid_index(label[0], reciprocal_grid[0]);
    const int iy = centered_grid_index(label[1], reciprocal_grid[1]);
    const int iz = centered_grid_index(label[2], reciprocal_grid[2]);
    return ix < 0 || iy < 0 || iz < 0
               ? -1
               : iz + reciprocal_grid[2] * (iy + reciprocal_grid[1] * ix);
}

void validate_states(const ModuleCell::KPointStrings& strings,
                     const std::vector<FiniteFieldPWState>& states,
                     const int occupied_bands,
                     const int state_stride)
{
    if (strings.direction < 1 || strings.direction > 3
        || strings.points_per_string < 2 || strings.indices.empty()
        || states.empty() || occupied_bands <= 0 || state_stride <= 0)
    {
        throw std::invalid_argument(
            "finite-field PW preparation dimensions are invalid");
    }

    std::vector<int> visit_count(states.size(), 0);
    for (const std::vector<int>& string : strings.indices)
    {
        if (static_cast<int>(string.size()) != strings.points_per_string
            || string.front() != string.back())
        {
            throw std::invalid_argument("finite-field k string is not closed");
        }
        for (int position = 0; position < strings.points_per_string - 1;
             ++position)
        {
            const int index = string[position];
            if (index < 0 || index >= static_cast<int>(states.size()))
            {
                throw std::out_of_range(
                    "finite-field k string index is invalid");
            }
            ++visit_count[index];
        }
    }
    for (const int count : visit_count)
    {
        if (count != 1)
        {
            throw std::invalid_argument(
                "finite-field k strings must cover each k point exactly once");
        }
    }
    for (const FiniteFieldPWState& state : states)
    {
        if (state.labels.empty()
            || state_stride < static_cast<int>(state.labels.size())
            || state.occupied.size()
                   != state_size(state_stride, occupied_bands))
        {
            throw std::invalid_argument(
                "finite-field occupied PW state dimensions are inconsistent");
        }
    }
}

void map_neighbor_to_current(
    const FiniteFieldPWState& current,
    const FiniteFieldPWState& neighbor,
    const int state_stride,
    const int occupied_bands,
    const std::array<int, 3>& reciprocal_grid,
    const FiniteFieldGLabel& shift,
    const FiniteFieldReducer& reducer,
    std::vector<std::complex<double>>& mapped)
{
    // Gather only the occupied coefficients which actually exist on a rank.
    // The earlier dense FFT-grid allreduce needed
    //   nx * ny * nz * occupied_bands * sizeof(complex<double>)
    // bytes per rank (about 456 MiB for the 72-atom SiO2 validation case),
    // regardless of the number of plane waves below cutoff.  Besides being
    // wasteful, that made a pool-parallel finite-field SCF impractical.
    std::vector<FiniteFieldGLabel> all_labels = neighbor.labels;
    std::vector<std::complex<double>> all_coefficients;
    all_coefficients.reserve(neighbor.labels.size() * occupied_bands);
    for (std::size_t local_pw = 0; local_pw < neighbor.labels.size(); ++local_pw)
    {
        for (int band = 0; band < occupied_bands; ++band)
        {
            all_coefficients.push_back(
                neighbor.occupied[band * state_stride + local_pw]);
        }
    }

#ifdef __MPI
    if (GlobalV::NPROC_IN_POOL > 1)
    {
        const int local_count = static_cast<int>(neighbor.labels.size());
        std::vector<int> counts(GlobalV::NPROC_IN_POOL, 0);
        MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, POOL_WORLD);
        std::vector<int> label_counts(GlobalV::NPROC_IN_POOL, 0);
        std::vector<int> label_displacements(GlobalV::NPROC_IN_POOL, 0);
        int total_count = 0;
        for (int rank = 0; rank < GlobalV::NPROC_IN_POOL; ++rank)
        {
            label_counts[rank] = 3 * counts[rank];
            label_displacements[rank] = 3 * total_count;
            total_count += counts[rank];
        }
        std::vector<int> local_label_data(3 * local_count);
        for (int local_pw = 0; local_pw < local_count; ++local_pw)
        {
            for (int component = 0; component < 3; ++component)
            {
                local_label_data[3 * local_pw + component]
                    = neighbor.labels[local_pw][component];
            }
        }
        std::vector<int> global_label_data(3 * total_count);
        MPI_Allgatherv(local_label_data.data(),
                       3 * local_count,
                       MPI_INT,
                       global_label_data.data(),
                       label_counts.data(),
                       label_displacements.data(),
                       MPI_INT,
                       POOL_WORLD);
        all_labels.resize(total_count);
        for (int global_pw = 0; global_pw < total_count; ++global_pw)
        {
            for (int component = 0; component < 3; ++component)
            {
                all_labels[global_pw][component]
                    = global_label_data[3 * global_pw + component];
            }
        }

        std::vector<int> coefficient_counts(GlobalV::NPROC_IN_POOL, 0);
        std::vector<int> coefficient_displacements(GlobalV::NPROC_IN_POOL, 0);
        int total_coefficients = 0;
        for (int rank = 0; rank < GlobalV::NPROC_IN_POOL; ++rank)
        {
            coefficient_counts[rank] = counts[rank] * occupied_bands;
            coefficient_displacements[rank] = total_coefficients;
            total_coefficients += coefficient_counts[rank];
        }
        std::vector<std::complex<double>> global_coefficients(total_coefficients);
        MPI_Allgatherv(all_coefficients.data(),
                       local_count * occupied_bands,
                       MPI_DOUBLE_COMPLEX,
                       global_coefficients.data(),
                       coefficient_counts.data(),
                       coefficient_displacements.data(),
                       MPI_DOUBLE_COMPLEX,
                       POOL_WORLD);
        all_coefficients.swap(global_coefficients);
    }
#endif

    std::map<FiniteFieldGLabel, int> coefficient_index;
    for (std::size_t global_pw = 0; global_pw < all_labels.size(); ++global_pw)
    {
        const FiniteFieldGLabel shifted = shifted_label(all_labels[global_pw], shift);
        if (linear_grid_index(shifted, reciprocal_grid) < 0)
        {
            continue;
        }
        if (!coefficient_index.insert(std::make_pair(shifted, static_cast<int>(global_pw))).second)
        {
            throw std::invalid_argument(
                "finite-field shifted PW labels must be unique");
        }
    }

    const int active_size = static_cast<int>(current.labels.size());
    mapped.assign(state_size(active_size, occupied_bands),
                  std::complex<double>(0.0, 0.0));
    for (int local_pw = 0; local_pw < active_size; ++local_pw)
    {
        if (linear_grid_index(current.labels[local_pw], reciprocal_grid) < 0)
        {
            throw std::invalid_argument(
                "finite-field current PW label is outside the reciprocal grid");
        }
        const std::map<FiniteFieldGLabel, int>::const_iterator found
            = coefficient_index.find(current.labels[local_pw]);
        if (found == coefficient_index.end())
        {
            // At the kinetic-cutoff boundary a current G vector can have no
            // shifted neighbour representation.  The former dense FFT-grid
            // implementation naturally returned zero in this case, so retain
            // that finite-basis convention here.
            continue;
        }
        for (int band = 0; band < occupied_bands; ++band)
        {
            mapped[band * active_size + local_pw]
                = all_coefficients[found->second * occupied_bands + band];
        }
    }
}

void copy_active_states(const FiniteFieldPWState& source,
                        const int occupied_bands,
                        const int state_stride,
                        std::vector<std::complex<double>>& active)
{
    const int active_size = static_cast<int>(source.labels.size());
    active.assign(state_size(active_size, occupied_bands),
                  std::complex<double>(0.0, 0.0));
    for (int band = 0; band < occupied_bands; ++band)
    {
        for (int basis = 0; basis < active_size; ++basis)
        {
            active[band * active_size + basis]
                = source.occupied[band * state_stride + basis];
        }
    }
}

void compute_overlap(const std::vector<std::complex<double>>& current,
                     const std::vector<std::complex<double>>& neighbor,
                     const int basis_size,
                     const int occupied_bands,
                     const FiniteFieldReducer& reducer,
                     std::vector<std::complex<double>>& overlap)
{
    overlap.assign(matrix_size(occupied_bands),
                   std::complex<double>(0.0, 0.0));
    // The wavefunctions are row-major [band][PW]. Interpreting the same
    // storage as column-major gives [PW][band], so this is exactly
    // current^H * neighbor.  This BLAS-3 formulation replaces the previous
    // O(Nocc^2 * Npw) scalar triple loop in every Berry-link evaluation.
    const char conjugate_transpose = 'C';
    const char no_transpose = 'N';
    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);
    zgemm_(&conjugate_transpose,
           &no_transpose,
           &occupied_bands,
           &occupied_bands,
           &basis_size,
           &one,
           current.data(),
           &basis_size,
           neighbor.data(),
           &basis_size,
           &zero,
           overlap.data(),
           &occupied_bands);
    // The overlap is another pool-distributed scalar contraction.
    reducer(overlap.data(), static_cast<int>(overlap.size()));
}

std::complex<double> determinant(
    const std::vector<std::complex<double>>& overlap,
    const int occupied_bands)
{
    std::vector<std::complex<double>> factorized = overlap;
    std::vector<int> pivots(occupied_bands);
    int info = 0;
    zgetrf_(&occupied_bands,
            &occupied_bands,
            factorized.data(),
            &occupied_bands,
            pivots.data(),
            &info);
    if (info != 0)
    {
        throw std::runtime_error("finite-field occupied overlap is singular");
    }

    std::complex<double> value(1.0, 0.0);
    for (int band = 0; band < occupied_bands; ++band)
    {
        value *= pivots[band] == band + 1 ? factorized[band * (occupied_bands + 1)]
                                         : -factorized[band * (occupied_bands + 1)];
    }
    return value;
}

void build_dual(const std::vector<std::complex<double>>& neighbor,
                const int basis_size,
                const int occupied_bands,
                std::vector<std::complex<double>> overlap,
                std::vector<std::complex<double>>& dual)
{
    std::vector<int> pivots(occupied_bands);
    int info = 0;
    zgetrf_(&occupied_bands,
            &occupied_bands,
            overlap.data(),
            &occupied_bands,
            pivots.data(),
            &info);
    if (info != 0)
    {
        throw std::runtime_error("finite-field occupied overlap is singular");
    }
    const int workspace_size = occupied_bands * 64;
    std::vector<std::complex<double>> workspace(workspace_size);
    zgetri_(&occupied_bands,
            overlap.data(),
            &occupied_bands,
            pivots.data(),
            workspace.data(),
            &workspace_size,
            &info);
    if (info != 0)
    {
        throw std::runtime_error(
            "finite-field occupied overlap inversion failed");
    }

    dual.assign(state_size(basis_size, occupied_bands),
                std::complex<double>(0.0, 0.0));
    // In the same column-major view, neighbour is [PW][band] and overlap is
    // [neighbour-band][dual-band], so neighbour * overlap produces the
    // desired [PW][dual-band] dual-state matrix.
    const char no_transpose = 'N';
    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);
    zgemm_(&no_transpose,
           &no_transpose,
           &basis_size,
           &occupied_bands,
           &occupied_bands,
           &one,
           neighbor.data(),
           &basis_size,
           overlap.data(),
           &occupied_bands,
           &zero,
           dual.data(),
           &basis_size);
}

void build_mapped_dual(
    const FiniteFieldPWState& current,
    const FiniteFieldPWState& neighbor,
    const FiniteFieldGLabel& shift,
    const int occupied_bands,
    const int state_stride,
    const std::array<int, 3>& reciprocal_grid,
    const FiniteFieldReducer& reducer,
    std::vector<std::complex<double>>& dual,
    std::complex<double>* overlap_determinant)
{
    std::vector<std::complex<double>> mapped_neighbor;
    map_neighbor_to_current(current,
                            neighbor,
                            state_stride,
                            occupied_bands,
                            reciprocal_grid,
                            shift,
                            reducer,
                            mapped_neighbor);
    std::vector<std::complex<double>> current_active;
    copy_active_states(current,
                       occupied_bands,
                       state_stride,
                       current_active);
    std::vector<std::complex<double>> overlap;
    compute_overlap(current_active,
                    mapped_neighbor,
                    static_cast<int>(current.labels.size()),
                    occupied_bands,
                    reducer,
                    overlap);
    if (overlap_determinant != nullptr)
    {
        *overlap_determinant = determinant(overlap, occupied_bands);
    }
    build_dual(mapped_neighbor,
               static_cast<int>(current.labels.size()),
               occupied_bands,
               overlap,
               dual);
}

void store_dual_difference(
    const std::vector<std::complex<double>>& minus,
    const std::vector<std::complex<double>>& plus,
    const int active_size,
    const int occupied_bands,
    const int state_stride,
    std::vector<std::complex<double>>& padded)
{
    if (minus.size() != plus.size())
    {
        throw std::invalid_argument(
            "finite-field dual dimensions are inconsistent");
    }
    padded.assign(state_size(state_stride, occupied_bands),
                  std::complex<double>(0.0, 0.0));
    for (int band = 0; band < occupied_bands; ++band)
    {
        for (int basis = 0; basis < active_size; ++basis)
        {
            const int active_index = band * active_size + basis;
            padded[band * state_stride + basis]
                = minus[active_index] - plus[active_index];
        }
    }
}

std::vector<FiniteFieldPWState> collect_abacus_states(
    const ModulePW::PW_Basis_K& basis,
    const psi::Psi<std::complex<double>, base_device::DEVICE_CPU>& wavefunctions,
    const int occupied_bands)
{
    if (basis.nks <= 0 || basis.npwk == nullptr || basis.gcar == nullptr
        || wavefunctions.get_nk() != basis.nks
        || wavefunctions.get_nbasis() != basis.npwk_max
        || occupied_bands <= 0
        || occupied_bands > wavefunctions.get_nbands())
    {
        throw std::invalid_argument(
            "finite-field PW basis and wavefunctions are inconsistent");
    }

    std::vector<FiniteFieldPWState> states(basis.nks);
    for (int ik = 0; ik < basis.nks; ++ik)
    {
        FiniteFieldPWState& state = states[ik];
        state.labels.reserve(basis.npwk[ik]);
        for (int local_pw = 0; local_pw < basis.npwk[ik]; ++local_pw)
        {
            const ModuleBase::Vector3<double> label
                = basis.getgdirect(ik, local_pw);
            state.labels.push_back(FiniteFieldGLabel{{
                integer_component(label.x),
                integer_component(label.y),
                integer_component(label.z)}});
        }
        state.occupied.assign(state_size(basis.npwk_max, occupied_bands),
                              std::complex<double>(0.0, 0.0));
        for (int band = 0; band < occupied_bands; ++band)
        {
            for (int local_pw = 0; local_pw < basis.npwk[ik]; ++local_pw)
            {
                state.occupied[band * basis.npwk_max + local_pw]
                    = wavefunctions(ik, band, local_pw);
            }
        }
    }
    return states;
}

} // namespace

void prepare_finite_field_pw_data(
    const ModuleCell::KPointStrings& strings,
    const std::vector<FiniteFieldPWState>& states,
    const int occupied_bands,
    const int state_stride,
    const std::array<int, 3>& reciprocal_grid,
    const FiniteFieldReducer& reducer,
    std::vector<FiniteFieldPreparedKPoint>& prepared)
{
    validate_states(strings, states, occupied_bands, state_stride);
    if (!reducer)
    {
        throw std::invalid_argument(
            "finite-field PW preparation needs a pool reducer");
    }

    prepared.assign(states.size(), FiniteFieldPreparedKPoint());
    const FiniteFieldGLabel no_shift{{0, 0, 0}};
    const FiniteFieldGLabel minus_closure
        = closure_shift(strings.direction, 1);
    const FiniteFieldGLabel plus_closure
        = closure_shift(strings.direction, -1);

    for (const std::vector<int>& string : strings.indices)
    {
        const int unique_points = strings.points_per_string - 1;
        for (int position = 0; position < unique_points; ++position)
        {
            const int current = string[position];
            const int minus
                = string[(position + unique_points - 1) % unique_points];
            const int plus = string[(position + 1) % unique_points];

            std::vector<std::complex<double>> dual_minus;
            std::vector<std::complex<double>> dual_plus;
            build_mapped_dual(states[current],
                              states[minus],
                              position == 0 ? minus_closure : no_shift,
                              occupied_bands,
                              state_stride,
                              reciprocal_grid,
                              reducer,
                              dual_minus,
                              nullptr);
            build_mapped_dual(
                states[current],
                states[plus],
                position == unique_points - 1 ? plus_closure : no_shift,
                occupied_bands,
                state_stride,
                reciprocal_grid,
                reducer,
                dual_plus,
                &prepared[current].forward_overlap_determinant);

            prepared[current].occupied = states[current].occupied;
            store_dual_difference(
                dual_minus,
                dual_plus,
                static_cast<int>(states[current].labels.size()),
                occupied_bands,
                state_stride,
                prepared[current].dual_difference);
        }
    }
}

void prepare_finite_field_pw_data_from_abacus(
    const ModuleCell::KPointStrings& strings,
    const ModulePW::PW_Basis_K& basis,
    const psi::Psi<std::complex<double>, base_device::DEVICE_CPU>& wavefunctions,
    const int occupied_bands,
    std::vector<FiniteFieldPreparedKPoint>& prepared)
{
    const std::vector<FiniteFieldPWState> states
        = collect_abacus_states(basis, wavefunctions, occupied_bands);
    prepare_finite_field_pw_data(
        strings,
        states,
        occupied_bands,
        basis.npwk_max,
        std::array<int, 3>{{basis.nx, basis.ny, basis.nz}},
        reduce_in_pool,
        prepared);
}

} // namespace hamilt
