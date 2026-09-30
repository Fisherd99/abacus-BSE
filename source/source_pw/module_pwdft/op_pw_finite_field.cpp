#include "op_pw_finite_field.h"

#include "source_base/parallel_reduce.h"

#include <algorithm>
#include <stdexcept>

namespace hamilt
{

namespace
{

std::size_t state_size(const int occupied_bands, const int state_stride)
{
    return static_cast<std::size_t>(occupied_bands) * state_stride;
}

std::size_t checked_kpoint_count(const int kpoint_count)
{
    if (kpoint_count <= 0)
    {
        throw std::invalid_argument("finite-field PW operator needs k points");
    }
    return static_cast<std::size_t>(kpoint_count);
}

void reduce_projections_in_pool(std::complex<double>* projections, const int count)
{
    Parallel_Reduce::reduce_pool(projections, count);
}

void apply_term(const std::complex<double>* occupied,
                const std::complex<double>* dual_difference,
                const std::complex<double>* wavefunctions,
                const int active_basis_size,
                const int state_stride,
                const int wavefunction_stride,
                const int occupied_bands,
                const int input_bands,
                const std::complex<double>& factor,
                const FiniteFieldOperatorPW::ProjectionReducer& reduce,
                std::complex<double>* output)
{
    const std::size_t projection_count
        = static_cast<std::size_t>(occupied_bands) * input_bands;
    std::vector<std::complex<double>> occupied_projections(
        projection_count, std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> dual_projections(
        projection_count, std::complex<double>(0.0, 0.0));

    // Each rank contracts only its local plane waves. The pool reduction
    // produces the global band projections before the local output is formed.
    for (int input_band = 0; input_band < input_bands; ++input_band)
    {
        const std::complex<double>* wavefunction
            = wavefunctions + input_band * wavefunction_stride;
        for (int occupied_band = 0; occupied_band < occupied_bands;
             ++occupied_band)
        {
            const std::complex<double>* occupied_state
                = occupied + occupied_band * state_stride;
            const std::complex<double>* dual_state
                = dual_difference + occupied_band * state_stride;
            const int projection_index
                = occupied_band + occupied_bands * input_band;
            for (int basis = 0; basis < active_basis_size; ++basis)
            {
                occupied_projections[projection_index]
                    += std::conj(occupied_state[basis]) * wavefunction[basis];
                dual_projections[projection_index]
                    += std::conj(dual_state[basis]) * wavefunction[basis];
            }
        }
    }
    reduce(occupied_projections.data(),
           static_cast<int>(occupied_projections.size()));
    reduce(dual_projections.data(),
           static_cast<int>(dual_projections.size()));

    const std::complex<double> adjoint_factor = std::conj(factor);
    for (int input_band = 0; input_band < input_bands; ++input_band)
    {
        std::complex<double>* output_band
            = output + input_band * wavefunction_stride;
        for (int occupied_band = 0; occupied_band < occupied_bands;
             ++occupied_band)
        {
            const std::complex<double>* occupied_state
                = occupied + occupied_band * state_stride;
            const std::complex<double>* dual_state
                = dual_difference + occupied_band * state_stride;
            const int projection_index
                = occupied_band + occupied_bands * input_band;
            const std::complex<double> occupied_coefficient
                = factor * occupied_projections[projection_index];
            const std::complex<double> dual_coefficient
                = adjoint_factor * dual_projections[projection_index];
            for (int basis = 0; basis < active_basis_size; ++basis)
            {
                output_band[basis]
                    += dual_state[basis] * occupied_coefficient
                       + occupied_state[basis] * dual_coefficient;
            }
        }
    }
}

} // namespace

FiniteFieldOperatorPW::FiniteFieldOperatorPW(const int kpoint_count,
                                             const int occupied_bands,
                                             const int state_stride)
    : FiniteFieldOperatorPW(kpoint_count,
                            occupied_bands,
                            state_stride,
                            reduce_projections_in_pool)
{
}

FiniteFieldOperatorPW::FiniteFieldOperatorPW(
    const int kpoint_count,
    const int occupied_bands,
    const int state_stride,
    const ProjectionReducer& projection_reducer)
    : occupied_bands(occupied_bands),
      state_stride(state_stride),
      projection_reducer(projection_reducer),
      kpoint_data(checked_kpoint_count(kpoint_count))
{
    if (occupied_bands <= 0)
    {
        throw std::invalid_argument("finite-field PW operator needs occupied bands");
    }
    if (state_stride <= 0)
    {
        throw std::invalid_argument("finite-field PW operator state stride must be positive");
    }
    if (!projection_reducer)
    {
        throw std::invalid_argument("finite-field PW operator needs a projection reducer");
    }

    this->classname = "FiniteField";
    this->cal_type = calculation_type::pw_finite_field;
}

void FiniteFieldOperatorPW::set_kpoint_data(
    const int ik,
    const std::complex<double>& factor,
    const std::vector<std::complex<double>>& occupied,
    const std::vector<std::complex<double>>& dual_difference)
{
    if (ik < 0 || ik >= static_cast<int>(this->kpoint_data.size()))
    {
        throw std::out_of_range("finite-field PW operator k-point index is invalid");
    }
    this->kpoint_data[ik].clear();
    this->add_kpoint_data(ik, factor, occupied, dual_difference);
}

void FiniteFieldOperatorPW::add_kpoint_data(
    const int ik,
    const std::complex<double>& factor,
    const std::vector<std::complex<double>>& occupied,
    const std::vector<std::complex<double>>& dual_difference)
{
    if (ik < 0 || ik >= static_cast<int>(this->kpoint_data.size()))
    {
        throw std::out_of_range("finite-field PW operator k-point index is invalid");
    }
    const std::size_t expected_size = state_size(this->occupied_bands, this->state_stride);
    if (occupied.size() != expected_size || dual_difference.size() != expected_size)
    {
        throw std::invalid_argument("finite-field PW operator state dimensions are inconsistent");
    }

    KPointTerm term;
    term.factor = factor;
    term.occupied = occupied;
    term.dual_difference = dual_difference;
    this->kpoint_data[ik].push_back(term);
}

void FiniteFieldOperatorPW::clear_kpoint_data()
{
    for (std::vector<KPointTerm>& terms : this->kpoint_data)
    {
        terms.clear();
    }
}

bool FiniteFieldOperatorPW::has_kpoint_data(const int ik) const
{
    return ik >= 0
           && ik < static_cast<int>(this->kpoint_data.size())
           && !this->kpoint_data[ik].empty();
}

void FiniteFieldOperatorPW::act(const int nbands,
                                const int nbasis,
                                const int npol,
                                const std::complex<double>* wavefunctions,
                                std::complex<double>* output,
                                const int active_basis_size,
                                const bool is_first_node) const
{
    if (nbands <= 0 || wavefunctions == nullptr || output == nullptr)
    {
        throw std::invalid_argument("finite-field PW operator input is invalid");
    }
    if (npol != 1)
    {
        throw std::invalid_argument("finite-field PW operator supports scalar wavefunctions only");
    }
    if (!this->has_kpoint_data(this->ik))
    {
        throw std::logic_error("finite-field PW operator data are not initialized for this k point");
    }
    const int active_size = active_basis_size > 0 ? active_basis_size : nbasis;
    if (is_first_node)
    {
        std::fill(output,
                  output + static_cast<std::size_t>(nbands) * nbasis,
                  std::complex<double>(0.0, 0.0));
    }

    for (const KPointTerm& term : this->kpoint_data[this->ik])
    {
        apply_term(
            term.occupied.data(),
            term.dual_difference.data(),
            wavefunctions,
            active_size,
            this->state_stride,
            nbasis,
            this->occupied_bands,
            nbands,
            term.factor,
            this->projection_reducer,
            output);
    }
}

} // namespace hamilt
