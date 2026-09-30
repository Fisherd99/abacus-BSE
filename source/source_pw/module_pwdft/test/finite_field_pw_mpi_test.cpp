#include "gtest/gtest.h"

#include "source_pw/module_pwdft/finite_field_pw.h"
#include "source_base/global_variable.h"
#include "source_base/parallel_comm.h"

#include "mpi.h"

#include <array>
#include <cmath>
#include <complex>
#include <vector>

namespace
{

using Complex = std::complex<double>;
using Label = hamilt::FiniteFieldGLabel;

int world_rank = 0;
int world_size = 1;

void allreduce(Complex* values, const int count)
{
    MPI_Allreduce(MPI_IN_PLACE,
                  values,
                  count,
                  MPI_CXX_DOUBLE_COMPLEX,
                  MPI_SUM,
                  MPI_COMM_WORLD);
}

hamilt::FiniteFieldPWState local_state(const Label& label,
                                      const Complex& coefficient)
{
    hamilt::FiniteFieldPWState state;
    state.labels = {label};
    state.occupied = {coefficient};
    return state;
}

TEST(FiniteFieldPWMPI, ReducesRemoteCoefficientsAndOverlaps)
{
    if (world_size != 2)
    {
        GTEST_SKIP() << "requires exactly two MPI ranks";
    }

    ModuleCell::KPointStrings strings;
    strings.direction = 1;
    strings.points_per_string = 3;
    strings.indices = {{0, 1, 0}};
    const Label minus_one{{-1, 0, 0}};
    const Label zero{{0, 0, 0}};

    // Each k point is distributed across both ranks with complementary G labels.
    const std::vector<hamilt::FiniteFieldPWState> states = {
        local_state(world_rank == 0 ? minus_one : zero,
                    world_rank == 0 ? Complex(1.0, 0.0)
                                    : Complex(2.0, 0.0)),
        local_state(world_rank == 0 ? zero : minus_one,
                    world_rank == 0 ? Complex(3.0, 0.0)
                                    : Complex(4.0, 0.0))};

    std::vector<hamilt::FiniteFieldPreparedKPoint> prepared;
    hamilt::prepare_finite_field_pw_data(
        strings,
        states,
        1,
        1,
        std::array<int, 3>{{5, 3, 3}},
        allreduce,
        prepared);

    ASSERT_EQ(prepared.size(), 2);
    for (const hamilt::FiniteFieldPreparedKPoint& point : prepared)
    {
        ASSERT_EQ(point.dual_difference.size(), 1);
        EXPECT_TRUE(std::isfinite(point.dual_difference[0].real()));
        EXPECT_TRUE(std::isfinite(point.forward_overlap_determinant.real()));
    }

    const double expected_first_dual = world_rank == 0 ? -0.4 : 0.2;
    const double expected_second_dual = world_rank == 0 ? 0.2 : -0.15;
    EXPECT_NEAR(std::abs(prepared[0].dual_difference[0]
                         - Complex(expected_first_dual, 0.0)),
                0.0,
                1.0e-14);
    EXPECT_NEAR(std::abs(prepared[1].dual_difference[0]
                         - Complex(expected_second_dual, 0.0)),
                0.0,
                1.0e-14);
    EXPECT_NEAR(std::abs(prepared[0].forward_overlap_determinant
                         - Complex(10.0, 0.0)),
                0.0,
                1.0e-14);
    EXPECT_NEAR(std::abs(prepared[1].forward_overlap_determinant
                         - Complex(8.0, 0.0)),
                0.0,
                1.0e-14);
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Production initialisation assigns the PW-pool communicator and size.
    // This focused unit test bypasses that driver, so mirror its contract.
    POOL_WORLD = MPI_COMM_WORLD;
    GlobalV::NPROC_IN_POOL = world_size;
    GlobalV::RANK_IN_POOL = world_rank;
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
