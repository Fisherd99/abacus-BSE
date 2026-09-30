#include "gtest/gtest.h"

#include "source_pw/module_pwdft/op_pw_finite_field.h"

#include <complex>
#include <stdexcept>
#include <vector>

namespace
{

using Complex = std::complex<double>;

TEST(FiniteFieldOperatorPWTest, AppliesPreparedKPointDataWithPadding)
{
    std::vector<int> reduction_counts;
    const hamilt::FiniteFieldOperatorPW::ProjectionReducer reducer
        = [&reduction_counts](Complex*, const int count) {
              reduction_counts.push_back(count);
          };
    hamilt::FiniteFieldOperatorPW finite_field(2, 1, 3, reducer);
    const std::vector<Complex> occupied = {
        Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(90.0, -40.0)};
    const std::vector<Complex> dual_difference = {
        Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(-30.0, 70.0)};
    finite_field.set_kpoint_data(0,
                                 Complex(0.0, -0.5),
                                 occupied,
                                 dual_difference);
    finite_field.init(0);

    const std::vector<Complex> wavefunction = {
        Complex(2.0, 3.0), Complex(5.0, 7.0), Complex(100.0, 200.0)};
    std::vector<Complex> output(3, Complex(9.0, -8.0));
    finite_field.act(1,
                     3,
                     1,
                     wavefunction.data(),
                     output.data(),
                     2,
                     true);

    ASSERT_EQ(reduction_counts.size(), 2);
    EXPECT_EQ(reduction_counts[0], 1);
    EXPECT_EQ(reduction_counts[1], 1);
    EXPECT_EQ(output[0], Complex(-3.5, 2.5));
    EXPECT_EQ(output[1], Complex(1.5, -1.0));
    EXPECT_EQ(output[2], Complex(0.0, 0.0));
}

TEST(FiniteFieldOperatorPWTest, AccumulatesWhenNotFirstNode)
{
    const hamilt::FiniteFieldOperatorPW::ProjectionReducer reducer
        = [](Complex*, int) {};
    hamilt::FiniteFieldOperatorPW finite_field(1, 1, 2, reducer);
    finite_field.set_kpoint_data(0,
                                 Complex(0.0, -0.5),
                                 {Complex(1.0, 0.0), Complex(0.0, 0.0)},
                                 {Complex(0.0, 0.0), Complex(1.0, 0.0)});
    finite_field.init(0);

    const std::vector<Complex> wavefunction = {
        Complex(2.0, 3.0), Complex(5.0, 7.0)};
    std::vector<Complex> output(2, Complex(1.0, -2.0));
    finite_field.act(1,
                     2,
                     1,
                     wavefunction.data(),
                     output.data(),
                     2,
                     false);

    EXPECT_EQ(output[0], Complex(-2.5, 0.5));
    EXPECT_EQ(output[1], Complex(2.5, -3.0));
}

TEST(FiniteFieldOperatorPWTest, AddsMultipleLatticeDirectionTerms)
{
    const hamilt::FiniteFieldOperatorPW::ProjectionReducer reducer
        = [](Complex*, int) {};
    hamilt::FiniteFieldOperatorPW finite_field(1, 1, 2, reducer);
    finite_field.clear_kpoint_data();
    finite_field.add_kpoint_data(
        0,
        Complex(0.0, -0.5),
        {Complex(1.0, 0.0), Complex(0.0, 0.0)},
        {Complex(0.0, 0.0), Complex(1.0, 0.0)});
    finite_field.add_kpoint_data(
        0,
        Complex(0.0, -0.25),
        {Complex(1.0, 0.0), Complex(0.0, 0.0)},
        {Complex(0.0, 0.0), Complex(1.0, 0.0)});
    finite_field.init(0);

    const std::vector<Complex> wavefunction = {
        Complex(2.0, 3.0), Complex(5.0, 7.0)};
    std::vector<Complex> output(2, Complex(9.0, -8.0));
    finite_field.act(1,
                     2,
                     1,
                     wavefunction.data(),
                     output.data(),
                     2,
                     true);

    EXPECT_EQ(output[0], Complex(-5.25, 3.75));
    EXPECT_EQ(output[1], Complex(2.25, -1.5));
}

TEST(FiniteFieldOperatorPWTest, SupportsCompactIterativeSolverBuffers)
{
    const hamilt::FiniteFieldOperatorPW::ProjectionReducer reducer
        = [](Complex*, int) {};
    hamilt::FiniteFieldOperatorPW finite_field(1, 1, 3, reducer);
    finite_field.set_kpoint_data(0,
                                 Complex(0.0, -0.5),
                                 {Complex(1.0, 0.0),
                                  Complex(0.0, 0.0),
                                  Complex(90.0, -40.0)},
                                 {Complex(0.0, 0.0),
                                  Complex(1.0, 0.0),
                                  Complex(-30.0, 70.0)});
    finite_field.init(0);

    const std::vector<Complex> compact_wavefunction = {
        Complex(2.0, 3.0), Complex(5.0, 7.0)};
    std::vector<Complex> compact_output(2, Complex(9.0, -8.0));
    finite_field.act(1,
                     2,
                     1,
                     compact_wavefunction.data(),
                     compact_output.data(),
                     2,
                     true);

    EXPECT_EQ(compact_output[0], Complex(-3.5, 2.5));
    EXPECT_EQ(compact_output[1], Complex(1.5, -1.0));
}

TEST(FiniteFieldOperatorPWTest, RejectsUnpreparedAndSpinorPaths)
{
    const hamilt::FiniteFieldOperatorPW::ProjectionReducer reducer
        = [](Complex*, int) {};
    hamilt::FiniteFieldOperatorPW finite_field(1, 1, 2, reducer);
    finite_field.init(0);
    const std::vector<Complex> wavefunction(2, Complex(0.0, 0.0));
    std::vector<Complex> output(2, Complex(0.0, 0.0));

    EXPECT_THROW(finite_field.act(1,
                                  2,
                                  1,
                                  wavefunction.data(),
                                  output.data(),
                                  2,
                                  false),
                 std::logic_error);

    finite_field.set_kpoint_data(0,
                                 Complex(0.0, -0.5),
                                 wavefunction,
                                 wavefunction);
    EXPECT_THROW(finite_field.act(1,
                                  2,
                                  2,
                                  wavefunction.data(),
                                  output.data(),
                                  2,
                                  false),
                 std::invalid_argument);
}

} // namespace
