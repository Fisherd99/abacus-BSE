#include "finite_field_matrix.h"

#include "source_base/module_external/blas_connector.h"
#include "source_base/module_external/lapack_connector.h"
#include "source_base/module_external/scalapack_connector.h"
#include "source_base/parallel_2d.h"

#include <stdexcept>

namespace hamilt
{
namespace
{
void interpolation_values(const double* values, const int size, const int index,
                          double& v0, double& v1, double& v2, double& v3)
{
    v0 = v1 = v2 = v3 = 0.0;
    if (index < 0 || index >= size) return;
    v0 = values[index];
    if (index + 1 < size) v1 = values[index + 1];
    if (index + 2 < size) v2 = values[index + 2];
    if (index + 3 < size) v3 = values[index + 3];
}
}

double finite_field_cubic_interpolate(const double* values, const int size,
                                      const double spacing, const double x)
{
    if (values == nullptr || size <= 0 || spacing <= 0.0 || x < 0.0) return 0.0;
    const double position = x / spacing;
    const int index = static_cast<int>(position);
    double v0, v1, v2, v3;
    interpolation_values(values, size, index, v0, v1, v2, v3);
    const double u = position - index;
    return v0 * (1.0 - u) * (2.0 - u) * (3.0 - u) / 6.0
           + v1 * u * (2.0 - u) * (3.0 - u) / 2.0
           - v2 * (1.0 - u) * u * (3.0 - u) / 2.0
           + v3 * (1.0 - u) * (2.0 - u) * u / 6.0;
}

double finite_field_cubic_interpolate_derivative(
    const double* values, const int size, const double spacing, const double x)
{
    if (values == nullptr || size <= 0 || spacing <= 0.0 || x < 0.0) return 0.0;
    const double position = x / spacing;
    const int index = static_cast<int>(position);
    double v0, v1, v2, v3;
    interpolation_values(values, size, index, v0, v1, v2, v3);
    const double u = position - index;
    return (v0 * (-11.0 + 12.0 * u - 3.0 * u * u) / 6.0
            + v1 * (3.0 - 5.0 * u + 1.5 * u * u)
            + v2 * (-1.5 + 4.0 * u - 1.5 * u * u)
            + v3 * (2.0 - 6.0 * u + 3.0 * u * u) / 6.0)
           / spacing;
}

namespace
{

std::size_t matrix_size(const int rows, const int columns)
{
    if (rows <= 0 || columns <= 0)
    {
        throw std::invalid_argument("finite-field dense matrix dimensions must be positive");
    }
    return static_cast<std::size_t>(rows) * columns;
}

void require_size(const std::vector<std::complex<double>>& matrix,
                  const int rows,
                  const int columns,
                  const char* name)
{
    if (matrix.size() != matrix_size(rows, columns))
    {
        throw std::invalid_argument(std::string("finite-field " ) + name
                                    + " dimensions are inconsistent");
    }
}

std::complex<double> invert_and_determinant(
    std::vector<std::complex<double>>& matrix, const int size)
{
    std::vector<int> pivots(size);
    int info = 0;
    zgetrf_(&size, &size, matrix.data(), &size, pivots.data(), &info);
    if (info != 0)
    {
        throw std::runtime_error("finite-field occupied link is singular");
    }
    std::complex<double> determinant(1.0, 0.0);
    for (int index = 0; index < size; ++index)
    {
        determinant *= pivots[index] == index + 1
                           ? matrix[index * (size + 1)]
                           : -matrix[index * (size + 1)];
    }
    const int workspace_size = 64 * size;
    std::vector<std::complex<double>> workspace(workspace_size);
    zgetri_(&size, matrix.data(), &size, pivots.data(), workspace.data(),
            &workspace_size, &info);
    if (info != 0)
    {
        throw std::runtime_error("finite-field occupied link inversion failed");
    }
    return determinant;
}

} // namespace

FiniteFieldDenseLink finite_field_dense_link(
    const std::vector<std::complex<double>>& current,
    const std::vector<std::complex<double>>& link,
    const std::vector<std::complex<double>>& neighbor,
    const int basis_size,
    const int occupied_bands)
{
    require_size(current, basis_size, occupied_bands, "current states");
    require_size(neighbor, basis_size, occupied_bands, "neighbor states");
    require_size(link, basis_size, basis_size, "AO link");

    const char n = 'N';
    const char c = 'C';
    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);
    std::vector<std::complex<double>> mapped(matrix_size(basis_size, occupied_bands));
    zgemm_(&n, &n, &basis_size, &occupied_bands, &basis_size, &one,
           link.data(), &basis_size, neighbor.data(), &basis_size, &zero,
           mapped.data(), &basis_size);

    std::vector<std::complex<double>> occupied_link(
        matrix_size(occupied_bands, occupied_bands));
    zgemm_(&c, &n, &occupied_bands, &occupied_bands, &basis_size, &one,
           current.data(), &basis_size, mapped.data(), &basis_size, &zero,
           occupied_link.data(), &occupied_bands);

    FiniteFieldDenseLink result;
    result.determinant = invert_and_determinant(occupied_link, occupied_bands);
    result.inverse = occupied_link;
    result.dual.resize(matrix_size(basis_size, occupied_bands));
    zgemm_(&n, &n, &basis_size, &occupied_bands, &occupied_bands, &one,
           mapped.data(), &basis_size, occupied_link.data(), &occupied_bands,
           &zero, result.dual.data(), &basis_size);
    return result;
}

FiniteFieldDistributedLink finite_field_distributed_link(
    const std::complex<double>* current,
    const std::vector<std::complex<double>>& local_link,
    const std::complex<double>* neighbor,
    const int basis_size,
    const int occupied_bands,
    const Parallel_2D& matrix_layout,
    const Parallel_2D& occupied_state_layout,
    const Parallel_2D& occupied_link_layout
#ifdef __MPI
    , const int* wavefunction_descriptor
    , MPI_Comm communicator
#endif
    )
{
    if (current == nullptr || neighbor == nullptr
        || local_link.size() != static_cast<std::size_t>(matrix_layout.get_local_size()))
    {
        throw std::invalid_argument("finite-field distributed link input is inconsistent");
    }
#ifdef __MPI
    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);
    std::vector<std::complex<double>> mapped(
        occupied_state_layout.get_local_size(), zero);
    ScalapackConnector::gemm(
        'N', 'N', basis_size, occupied_bands, basis_size, one,
        local_link.data(), 1, 1, matrix_layout.get_desc(), neighbor, 1, 1,
        wavefunction_descriptor, zero, mapped.data(), 1, 1,
        occupied_state_layout.get_desc());

    std::vector<std::complex<double>> occupied_local(
        occupied_link_layout.get_local_size(), zero);
    ScalapackConnector::gemm(
        'C', 'N', occupied_bands, occupied_bands, basis_size, one, current,
        1, 1, wavefunction_descriptor, mapped.data(), 1, 1,
        occupied_state_layout.get_desc(), zero, occupied_local.data(), 1, 1,
        occupied_link_layout.get_desc());

    std::vector<std::complex<double>> occupied_full(
        matrix_size(occupied_bands, occupied_bands), zero);
    for (int local_column = 0; local_column < occupied_link_layout.get_col_size();
         ++local_column)
    {
        const int column = occupied_link_layout.local2global_col(local_column);
        for (int local_row = 0; local_row < occupied_link_layout.get_row_size();
             ++local_row)
        {
            const int row = occupied_link_layout.local2global_row(local_row);
            occupied_full[row + occupied_bands * column]
                = occupied_local[local_row
                                 + occupied_link_layout.get_row_size()
                                       * local_column];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, occupied_full.data(), occupied_full.size(),
                  MPI_DOUBLE_COMPLEX, MPI_SUM, communicator);

    FiniteFieldDistributedLink result;
    result.determinant
        = invert_and_determinant(occupied_full, occupied_bands);
    result.inverse = occupied_full;
    std::vector<std::complex<double>> inverse_local(
        occupied_link_layout.get_local_size(), zero);
    for (int local_column = 0; local_column < occupied_link_layout.get_col_size();
         ++local_column)
    {
        const int column = occupied_link_layout.local2global_col(local_column);
        for (int local_row = 0; local_row < occupied_link_layout.get_row_size();
             ++local_row)
        {
            const int row = occupied_link_layout.local2global_row(local_row);
            inverse_local[local_row
                          + occupied_link_layout.get_row_size() * local_column]
                = result.inverse[row + occupied_bands * column];
        }
    }
    result.local_dual.assign(occupied_state_layout.get_local_size(), zero);
    ScalapackConnector::gemm(
        'N', 'N', basis_size, occupied_bands, occupied_bands, one,
        mapped.data(), 1, 1, occupied_state_layout.get_desc(),
        inverse_local.data(), 1, 1, occupied_link_layout.get_desc(), zero,
        result.local_dual.data(), 1, 1, occupied_state_layout.get_desc());
    return result;
#else
    const std::vector<std::complex<double>> current_full(
        current, current + matrix_size(basis_size, occupied_bands));
    const std::vector<std::complex<double>> neighbor_full(
        neighbor, neighbor + matrix_size(basis_size, occupied_bands));
    const FiniteFieldDenseLink dense = finite_field_dense_link(
        current_full, local_link, neighbor_full, basis_size, occupied_bands);
    return {dense.dual, dense.inverse, dense.determinant};
#endif
}

void finite_field_distributed_hamiltonian(
    const std::complex<double>* local_overlap,
    const std::complex<double>* occupied,
    const std::vector<std::complex<double>>& dual_minus,
    const std::vector<std::complex<double>>& dual_plus,
    const int basis_size,
    const int occupied_bands,
    const std::complex<double>& gamma,
    const Parallel_2D& matrix_layout,
    const Parallel_2D& occupied_state_layout
#ifdef __MPI
    , const int* wavefunction_descriptor
#endif
    ,
    std::vector<std::complex<double>>& local_hamiltonian)
{
#ifdef __MPI
    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);
    std::vector<std::complex<double>> sc(
        occupied_state_layout.get_local_size(), zero);
    ScalapackConnector::gemm(
        'N', 'N', basis_size, occupied_bands, basis_size, one, local_overlap,
        1, 1, matrix_layout.get_desc(), occupied, 1, 1,
        wavefunction_descriptor, zero, sc.data(), 1, 1,
        occupied_state_layout.get_desc());
    std::vector<std::complex<double>> difference(dual_minus.size());
    for (std::size_t index = 0; index < difference.size(); ++index)
    {
        difference[index] = dual_minus[index] - dual_plus[index];
    }
    if (local_hamiltonian.size()
        != static_cast<std::size_t>(matrix_layout.get_local_size()))
    {
        local_hamiltonian.assign(matrix_layout.get_local_size(), zero);
    }
    ScalapackConnector::gemm(
        'N', 'C', basis_size, basis_size, occupied_bands, gamma,
        difference.data(), 1, 1, occupied_state_layout.get_desc(), sc.data(),
        1, 1, occupied_state_layout.get_desc(), one, local_hamiltonian.data(),
        1, 1, matrix_layout.get_desc());
    ScalapackConnector::gemm(
        'N', 'C', basis_size, basis_size, occupied_bands, std::conj(gamma),
        sc.data(), 1, 1, occupied_state_layout.get_desc(), difference.data(),
        1, 1, occupied_state_layout.get_desc(), one, local_hamiltonian.data(),
        1, 1, matrix_layout.get_desc());
#else
    const std::vector<std::complex<double>> overlap(
        local_overlap, local_overlap + matrix_size(basis_size, basis_size));
    const std::vector<std::complex<double>> states(
        occupied, occupied + matrix_size(basis_size, occupied_bands));
    const std::vector<std::complex<double>> dense = finite_field_dense_hamiltonian(
        overlap, states, dual_minus, dual_plus, basis_size, occupied_bands, gamma);
    if (local_hamiltonian.empty()) local_hamiltonian.resize(dense.size(), zero);
    for (std::size_t index = 0; index < dense.size(); ++index)
    {
        local_hamiltonian[index] += dense[index];
    }
#endif
}

std::vector<std::complex<double>> finite_field_dense_hamiltonian_block(
    const std::vector<std::complex<double>>& overlap,
    const std::vector<std::complex<double>>& occupied,
    const std::vector<std::complex<double>>& dual_minus,
    const std::vector<std::complex<double>>& dual_plus,
    const int basis_size,
    const int occupied_bands,
    const std::complex<double> gamma,
    const std::vector<int>& local_rows,
    const std::vector<int>& local_columns)
{
    require_size(overlap, basis_size, basis_size, "AO overlap");
    require_size(occupied, basis_size, occupied_bands, "occupied states");
    require_size(dual_minus, basis_size, occupied_bands, "minus dual states");
    require_size(dual_plus, basis_size, occupied_bands, "plus dual states");

    const char n = 'N';
    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);
    std::vector<std::complex<double>> sc(matrix_size(basis_size, occupied_bands));
    zgemm_(&n, &n, &basis_size, &occupied_bands, &basis_size, &one,
           overlap.data(), &basis_size, occupied.data(), &basis_size, &zero,
           sc.data(), &basis_size);

    std::vector<std::complex<double>> result(
        static_cast<std::size_t>(local_rows.size()) * local_columns.size(), zero);
    const std::complex<double> adjoint_gamma = std::conj(gamma);
    for (std::size_t local_column = 0; local_column < local_columns.size(); ++local_column)
    {
        const int column = local_columns[local_column];
        for (std::size_t local_row = 0; local_row < local_rows.size(); ++local_row)
        {
            const int row = local_rows[local_row];
            std::complex<double> value(0.0, 0.0);
            for (int band = 0; band < occupied_bands; ++band)
            {
                const std::complex<double> difference_row
                    = dual_minus[row + basis_size * band]
                      - dual_plus[row + basis_size * band];
                const std::complex<double> difference_column
                    = dual_minus[column + basis_size * band]
                      - dual_plus[column + basis_size * band];
                value += gamma * difference_row
                             * std::conj(sc[column + basis_size * band])
                         + adjoint_gamma * sc[row + basis_size * band]
                             * std::conj(difference_column);
            }
            result[local_row + local_rows.size() * local_column] = value;
        }
    }
    return result;
}

std::complex<double> finite_field_dense_logdet_derivative(
    const std::vector<std::complex<double>>& current,
    const std::vector<std::complex<double>>& link_derivative,
    const std::vector<std::complex<double>>& neighbor,
    const std::vector<std::complex<double>>& occupied_link_inverse,
    const int basis_size,
    const int occupied_bands)
{
    require_size(current, basis_size, occupied_bands, "current states");
    require_size(neighbor, basis_size, occupied_bands, "neighbor states");
    require_size(link_derivative, basis_size, basis_size, "AO-link derivative");
    require_size(occupied_link_inverse, occupied_bands, occupied_bands,
                 "occupied-link inverse");

    const char n = 'N';
    const char c = 'C';
    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);
    std::vector<std::complex<double>> mapped(matrix_size(basis_size, occupied_bands));
    zgemm_(&n, &n, &basis_size, &occupied_bands, &basis_size, &one,
           link_derivative.data(), &basis_size, neighbor.data(), &basis_size,
           &zero, mapped.data(), &basis_size);
    std::vector<std::complex<double>> derivative(
        matrix_size(occupied_bands, occupied_bands));
    zgemm_(&c, &n, &occupied_bands, &occupied_bands, &basis_size, &one,
           current.data(), &basis_size, mapped.data(), &basis_size, &zero,
           derivative.data(), &occupied_bands);

    std::complex<double> trace(0.0, 0.0);
    for (int row = 0; row < occupied_bands; ++row)
    {
        for (int column = 0; column < occupied_bands; ++column)
        {
            trace += occupied_link_inverse[row + occupied_bands * column]
                     * derivative[column + occupied_bands * row];
        }
    }
    return trace;
}

std::vector<std::complex<double>> finite_field_dense_hamiltonian(
    const std::vector<std::complex<double>>& overlap,
    const std::vector<std::complex<double>>& occupied,
    const std::vector<std::complex<double>>& dual_minus,
    const std::vector<std::complex<double>>& dual_plus,
    const int basis_size,
    const int occupied_bands,
    const std::complex<double>& gamma)
{
    require_size(overlap, basis_size, basis_size, "AO overlap");
    require_size(occupied, basis_size, occupied_bands, "occupied states");
    require_size(dual_minus, basis_size, occupied_bands, "minus dual states");
    require_size(dual_plus, basis_size, occupied_bands, "plus dual states");

    const char n = 'N';
    const char c = 'C';
    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> zero(0.0, 0.0);
    std::vector<std::complex<double>> sc(matrix_size(basis_size, occupied_bands));
    zgemm_(&n, &n, &basis_size, &occupied_bands, &basis_size, &one,
           overlap.data(), &basis_size, occupied.data(), &basis_size, &zero,
           sc.data(), &basis_size);

    std::vector<std::complex<double>> difference(dual_minus.size());
    for (std::size_t index = 0; index < difference.size(); ++index)
    {
        difference[index] = dual_minus[index] - dual_plus[index];
    }

    std::vector<std::complex<double>> result(matrix_size(basis_size, basis_size));
    zgemm_(&n, &c, &basis_size, &basis_size, &occupied_bands, &gamma,
           difference.data(), &basis_size, sc.data(), &basis_size, &zero,
           result.data(), &basis_size);
    const std::complex<double> adjoint_gamma = std::conj(gamma);
    zgemm_(&n, &c, &basis_size, &basis_size, &occupied_bands,
           &adjoint_gamma, sc.data(), &basis_size, difference.data(),
           &basis_size, &one, result.data(), &basis_size);
    return result;
}

} // namespace hamilt
