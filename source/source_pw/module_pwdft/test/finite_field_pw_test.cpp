#include "gtest/gtest.h"

#include "source_pw/module_pwdft/finite_field_pw.h"

#include <array>
#include <complex>
#include <stdexcept>
#include <vector>

namespace
{

using Complex = std::complex<double>;
using Label = hamilt::FiniteFieldGLabel;

hamilt::FiniteFieldPWState make_state(const std::vector<Label>& labels,
                                      const double coefficient,
                                      const int stride)
{
    hamilt::FiniteFieldPWState state;
    state.labels = labels;
    state.occupied.assign(stride, Complex(90.0, -40.0));
    for (std::size_t basis = 0; basis < labels.size(); ++basis)
    {
        state.occupied[basis] = Complex(coefficient, 0.0);
    }
    return state;
}

TEST(FiniteFieldPWTest, PreparesClosedStringWithPadding)
{
    ModuleCell::KPointStrings strings;
    strings.direction = 1;
    strings.points_per_string = 4;
    strings.indices = {{0, 1, 2, 0}};
    const std::vector<Label> labels = {
        Label{{-1, 0, 0}}, Label{{0, 0, 0}}, Label{{1, 0, 0}}};
    const int stride = 4;
    const std::vector<hamilt::FiniteFieldPWState> states = {
        make_state(labels, 1.0, stride),
        make_state(labels, 2.0, stride),
        make_state(labels, 3.0, stride)};
    int reductions = 0;

    std::vector<hamilt::FiniteFieldPreparedKPoint> prepared;
    hamilt::prepare_finite_field_pw_data(
        strings,
        states,
        1,
        stride,
        std::array<int, 3>{{5, 3, 3}},
        [&reductions](Complex*, int) { ++reductions; },
        prepared);

    ASSERT_EQ(prepared.size(), 3);
    // The sparse-label implementation gathers neighbour coefficients and
    // reduces only the two occupied overlap matrices per k point.
    EXPECT_EQ(reductions, 6);
    for (int ik = 0; ik < 3; ++ik)
    {
        ASSERT_EQ(prepared[ik].occupied.size(), stride);
        ASSERT_EQ(prepared[ik].dual_difference.size(), stride);
        EXPECT_EQ(prepared[ik].occupied[3], Complex(90.0, -40.0));
        EXPECT_EQ(prepared[ik].dual_difference[3], Complex(0.0, 0.0));
    }

    EXPECT_NEAR(std::abs(prepared[0].dual_difference[0]
                         - Complex(-1.0 / 3.0, 0.0)),
                0.0,
                1.0e-14);
    EXPECT_NEAR(std::abs(prepared[0].dual_difference[1]
                         - Complex(1.0 / 6.0, 0.0)),
                0.0,
                1.0e-14);
    EXPECT_NEAR(std::abs(prepared[2].dual_difference[2]
                         - Complex(1.0 / 9.0, 0.0)),
                0.0,
                1.0e-14);
}

TEST(FiniteFieldPWTest, PreparesGammaOnlyClosureWithOppositeReciprocalShifts)
{
    ModuleCell::KPointStrings strings;
    strings.direction = 1;
    strings.points_per_string = 2;
    strings.indices = {{0, 0}};
    const std::vector<Label> labels = {
        Label{{-1, 0, 0}}, Label{{0, 0, 0}}, Label{{1, 0, 0}}};
    std::vector<hamilt::FiniteFieldPWState> states = {
        make_state(labels, 1.0, 3)};
    states[0].occupied[0] = Complex(1.0, 0.0);
    states[0].occupied[1] = Complex(2.0, 0.0);
    states[0].occupied[2] = Complex(3.0, 0.0);

    std::vector<hamilt::FiniteFieldPreparedKPoint> prepared;
    hamilt::prepare_finite_field_pw_data(
        strings,
        states,
        1,
        3,
        std::array<int, 3>{{7, 3, 3}},
        [](Complex*, int) {},
        prepared);

    ASSERT_EQ(prepared.size(), 1);
    EXPECT_NE(0.0, std::abs(prepared[0].forward_overlap_determinant));
    EXPECT_NEAR(std::abs(prepared[0].dual_difference[0] - Complex(-0.25, 0.0)),
                0.0,
                1.0e-14);
    EXPECT_NEAR(std::abs(prepared[0].dual_difference[1] - Complex(-0.25, 0.0)),
                0.0,
                1.0e-14);
    EXPECT_NEAR(std::abs(prepared[0].dual_difference[2] - Complex(0.25, 0.0)),
                0.0,
                1.0e-14);
}

TEST(FiniteFieldPWTest, RejectsIncompleteKPointCoverage)
{
    ModuleCell::KPointStrings strings;
    strings.direction = 1;
    strings.points_per_string = 2;
    strings.indices = {{0, 0}};
    const std::vector<hamilt::FiniteFieldPWState> states = {
        make_state({Label{{0, 0, 0}}}, 1.0, 1),
        make_state({Label{{0, 0, 0}}}, 1.0, 1)};
    std::vector<hamilt::FiniteFieldPreparedKPoint> prepared;

    EXPECT_THROW(
        hamilt::prepare_finite_field_pw_data(
            strings,
            states,
            1,
            1,
            std::array<int, 3>{{3, 3, 3}},
            [](Complex*, int) {},
            prepared),
        std::invalid_argument);
}

} // namespace
