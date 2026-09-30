#include "source_hamilt/module_finite_field/finite_field_matrix.h"

#include <gtest/gtest.h>

#include <complex>
#include <vector>

namespace
{
using Complex = std::complex<double>;

double max_antihermitian(
    const std::vector<Complex>& matrix, const int size)
{
    double error = 0.0;
    for (int column = 0; column < size; ++column)
    {
        for (int row = 0; row < size; ++row)
        {
            error = std::max(error, std::abs(
                matrix[row + size * column]
                - std::conj(matrix[column + size * row])));
        }
    }
    return error;
}
} // namespace

TEST(FiniteFieldMatrix, CubicRadialDerivativeMatchesCenteredDifference)
{
    const std::vector<double> values{0.3, 0.9, -0.2, 1.7, 0.4, -0.1};
    const double spacing = 0.2;
    for (double x : {0.07, 0.31, 0.58})
    {
        const double h = 1.0e-7;
        const double numerical
            = (hamilt::finite_field_cubic_interpolate(
                   values.data(), values.size(), spacing, x + h)
               - hamilt::finite_field_cubic_interpolate(
                   values.data(), values.size(), spacing, x - h))
              / (2.0 * h);
        EXPECT_NEAR(hamilt::finite_field_cubic_interpolate_derivative(
                        values.data(), values.size(), spacing, x),
                    numerical, 2.0e-8);
    }
}

TEST(FiniteFieldMatrix, ZeroTransferReducesToMetricOverlap)
{
    const int nao = 2;
    const int nocc = 1;
    // Column-major Hermitian positive-definite S and S-normalized C.
    const std::vector<Complex> overlap{Complex(1.2, 0.0), Complex(0.1, -0.2),
                                       Complex(0.1, 0.2), Complex(1.4, 0.0)};
    std::vector<Complex> occupied{Complex(0.5, 0.1), Complex(-0.2, 0.6)};
    Complex norm(0.0, 0.0);
    for (int i = 0; i < nao; ++i)
    {
        for (int j = 0; j < nao; ++j)
        {
            norm += std::conj(occupied[i]) * overlap[i + nao * j] * occupied[j];
        }
    }
    for (Complex& value : occupied)
    {
        value /= std::sqrt(norm.real());
    }

    const hamilt::FiniteFieldDenseLink link
        = hamilt::finite_field_dense_link(occupied, overlap, occupied, nao, nocc);
    EXPECT_NEAR(link.determinant.real(), 1.0, 1.0e-12);
    EXPECT_NEAR(link.determinant.imag(), 0.0, 1.0e-12);
    std::vector<Complex> metric_occupied(nao);
    for (int row = 0; row < nao; ++row)
    {
        for (int column = 0; column < nao; ++column)
        {
            metric_occupied[row]
                += overlap[row + nao * column] * occupied[column];
        }
        EXPECT_NEAR(link.dual[row].real(), metric_occupied[row].real(), 1.0e-12);
        EXPECT_NEAR(link.dual[row].imag(), metric_occupied[row].imag(), 1.0e-12);
    }
}

TEST(FiniteFieldMatrix, HermitianLowRankHamiltonian)
{
    const int nao = 3;
    const int nocc = 2;
    const std::vector<Complex> overlap{
        {1.2, 0.0}, {0.1, -0.1}, {0.0, 0.2},
        {0.1, 0.1}, {1.1, 0.0}, {-0.1, 0.0},
        {0.0, -0.2}, {-0.1, 0.0}, {1.3, 0.0}};
    const std::vector<Complex> occupied{
        {0.7, 0.1}, {0.2, -0.3}, {-0.1, 0.4},
        {0.1, -0.2}, {0.6, 0.3}, {0.2, 0.1}};
    const std::vector<Complex> minus{
        {0.4, 0.2}, {-0.2, 0.3}, {0.5, -0.1},
        {-0.1, 0.5}, {0.3, 0.2}, {0.2, -0.4}};
    const std::vector<Complex> plus{
        {0.1, -0.3}, {0.2, 0.1}, {-0.2, 0.2},
        {0.4, 0.1}, {-0.1, 0.2}, {0.3, 0.3}};
    const std::vector<Complex> field
        = hamilt::finite_field_dense_hamiltonian(
            overlap, occupied, minus, plus, nao, nocc, Complex(0.0, -0.07));
    EXPECT_LT(max_antihermitian(field, nao), 1.0e-12);

    const std::vector<int> rows{2, 0};
    const std::vector<int> columns{1, 2};
    const std::vector<Complex> block
        = hamilt::finite_field_dense_hamiltonian_block(
            overlap, occupied, minus, plus, nao, nocc, Complex(0.0, -0.07),
            rows, columns);
    for (std::size_t j = 0; j < columns.size(); ++j)
    {
        for (std::size_t i = 0; i < rows.size(); ++i)
        {
            EXPECT_NEAR(std::abs(block[i + rows.size() * j]
                                 - field[rows[i] + nao * columns[j]]),
                        0.0, 1.0e-12);
        }
    }
}

TEST(FiniteFieldMatrix, OccupiedGaugeChangesOnlyLinkPhase)
{
    const int nao = 2;
    const int nocc = 2;
    const std::vector<Complex> identity{
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
    const std::vector<Complex> current = identity;
    const std::vector<Complex> neighbor{
        {0.8, 0.0}, {0.0, 0.6}, {0.0, 0.6}, {0.8, 0.0}};
    const double theta = 0.37;
    std::vector<Complex> gauged = neighbor;
    const Complex phase = std::exp(Complex(0.0, theta));
    gauged[0] *= phase;
    gauged[1] *= phase;

    const hamilt::FiniteFieldDenseLink original
        = hamilt::finite_field_dense_link(current, identity, neighbor, nao, nocc);
    const hamilt::FiniteFieldDenseLink transformed
        = hamilt::finite_field_dense_link(current, identity, gauged, nao, nocc);
    EXPECT_NEAR(std::abs(original.determinant),
                std::abs(transformed.determinant), 1.0e-12);
    for (std::size_t index = 0; index < original.dual.size(); ++index)
    {
        EXPECT_NEAR(std::abs(original.dual[index] - transformed.dual[index]),
                    0.0, 1.0e-12);
    }
}

TEST(FiniteFieldMatrix, LogDetDerivativeMatchesCenteredDifference)
{
    const int nao = 3;
    const int nocc = 2;
    const std::vector<Complex> current{
        {0.8, 0.1}, {0.1, -0.2}, {-0.2, 0.3},
        {0.2, -0.1}, {0.7, 0.2}, {0.1, 0.4}};
    const std::vector<Complex> neighbor{
        {0.6, -0.2}, {0.2, 0.1}, {0.1, 0.3},
        {-0.1, 0.2}, {0.8, -0.1}, {0.3, 0.1}};
    const std::vector<Complex> link{
        {1.1, 0.0}, {0.1, -0.2}, {0.0, 0.1},
        {-0.1, 0.1}, {0.9, 0.0}, {0.2, -0.1},
        {0.0, -0.1}, {-0.2, -0.1}, {1.2, 0.0}};
    const std::vector<Complex> derivative{
        {0.03, -0.02}, {-0.01, 0.04}, {0.02, 0.01},
        {0.00, -0.03}, {-0.02, 0.01}, {0.01, 0.02},
        {-0.01, 0.00}, {0.04, -0.02}, {0.02, 0.03}};

    const hamilt::FiniteFieldDenseLink reference
        = hamilt::finite_field_dense_link(
            current, link, neighbor, nao, nocc);
    const Complex analytic = hamilt::finite_field_dense_logdet_derivative(
        current, derivative, neighbor, reference.inverse, nao, nocc);

    const double step = 1.0e-6;
    std::vector<Complex> plus = link;
    std::vector<Complex> minus = link;
    for (std::size_t index = 0; index < link.size(); ++index)
    {
        plus[index] += step * derivative[index];
        minus[index] -= step * derivative[index];
    }
    const Complex log_plus = std::log(
        hamilt::finite_field_dense_link(current, plus, neighbor, nao, nocc)
            .determinant);
    const Complex log_minus = std::log(
        hamilt::finite_field_dense_link(current, minus, neighbor, nao, nocc)
            .determinant);
    const Complex numerical = (log_plus - log_minus) / (2.0 * step);

    EXPECT_NEAR(analytic.real(), numerical.real(), 1.0e-9);
    EXPECT_NEAR(analytic.imag(), numerical.imag(), 1.0e-9);
}
