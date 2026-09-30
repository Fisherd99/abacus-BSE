#ifndef MODULE_FINITE_FIELD_MATRIX_H
#define MODULE_FINITE_FIELD_MATRIX_H

#include <complex>
#include <vector>

#ifdef __MPI
#include <mpi.h>
#endif

class Parallel_2D;

namespace hamilt
{

/** Four-point cubic interpolation used by finite-field radial tables. */
double finite_field_cubic_interpolate(const double* values,
                                      int size,
                                      double spacing,
                                      double x);
double finite_field_cubic_interpolate_derivative(const double* values,
                                                 int size,
                                                 double spacing,
                                                 double x);

/** Dense, column-major occupied-space data for one finite-field k point. */
struct FiniteFieldDenseLink
{
    std::vector<std::complex<double>> dual;
    std::vector<std::complex<double>> inverse;
    std::complex<double> determinant = std::complex<double>(0.0, 0.0);
};

/** Distributed counterpart of FiniteFieldDenseLink.  The AO link, occupied
 * coefficients, mapped coefficients, dual coefficients, and resulting AO
 * Hamiltonian use compatible two-dimensional block-cyclic layouts.  Only the
 * small occupied-space inverse is replicated. */
struct FiniteFieldDistributedLink
{
    std::vector<std::complex<double>> local_dual;
    std::vector<std::complex<double>> inverse;
    std::complex<double> determinant = std::complex<double>(0.0, 0.0);
};

FiniteFieldDistributedLink finite_field_distributed_link(
    const std::complex<double>* current,
    const std::vector<std::complex<double>>& local_link,
    const std::complex<double>* neighbor,
    int basis_size,
    int occupied_bands,
    const Parallel_2D& matrix_layout,
    const Parallel_2D& occupied_state_layout,
    const Parallel_2D& occupied_link_layout
#ifdef __MPI
    , const int* wavefunction_descriptor
    , MPI_Comm communicator
#endif
    );

void finite_field_distributed_hamiltonian(
    const std::complex<double>* local_overlap,
    const std::complex<double>* occupied,
    const std::vector<std::complex<double>>& dual_minus,
    const std::vector<std::complex<double>>& dual_plus,
    int basis_size,
    int occupied_bands,
    const std::complex<double>& gamma,
    const Parallel_2D& matrix_layout,
    const Parallel_2D& occupied_state_layout
#ifdef __MPI
    , const int* wavefunction_descriptor
#endif
    ,
    std::vector<std::complex<double>>& local_hamiltonian);

/** Build M=C_current^H T C_neighbor and Q=T C_neighbor M^{-1}. */
FiniteFieldDenseLink finite_field_dense_link(
    const std::vector<std::complex<double>>& current,
    const std::vector<std::complex<double>>& link,
    const std::vector<std::complex<double>>& neighbor,
    int basis_size,
    int occupied_bands);

/** Return Tr[M^-1 C_current^H (dT) C_neighbor], the derivative of
 * log(det(M)) at fixed occupied coefficients. */
std::complex<double> finite_field_dense_logdet_derivative(
    const std::vector<std::complex<double>>& current,
    const std::vector<std::complex<double>>& link_derivative,
    const std::vector<std::complex<double>>& neighbor,
    const std::vector<std::complex<double>>& occupied_link_inverse,
    int basis_size,
    int occupied_bands);

/** Build the Hermitian AO matrix
 * gamma D C^H S + gamma^* S C D^H, where D=Q_- - Q_+. */
std::vector<std::complex<double>> finite_field_dense_hamiltonian(
    const std::vector<std::complex<double>>& overlap,
    const std::vector<std::complex<double>>& occupied,
    const std::vector<std::complex<double>>& dual_minus,
    const std::vector<std::complex<double>>& dual_plus,
    int basis_size,
    int occupied_bands,
    const std::complex<double>& gamma);

/** Build only selected rows and columns of the Hermitian AO matrix. */
std::vector<std::complex<double>> finite_field_dense_hamiltonian_block(
    const std::vector<std::complex<double>>& overlap,
    const std::vector<std::complex<double>>& occupied,
    const std::vector<std::complex<double>>& dual_minus,
    const std::vector<std::complex<double>>& dual_plus,
    int basis_size,
    int occupied_bands,
    std::complex<double> gamma,
    const std::vector<int>& local_rows,
    const std::vector<int>& local_columns);

} // namespace hamilt

#endif
