#include "gtest/gtest.h"

#include "source_cell/kpoint_strings.h"

#include <array>
#include <stdexcept>
#include <vector>

namespace
{

using ModuleCell::KPointStrings;

TEST(KPointStringsTest, BuildsPeriodicStringsAlongX)
{
    const std::array<int, 3> mesh = {{2, 3, 2}};
    const KPointStrings strings = ModuleCell::build_periodic_kpoint_strings(mesh, 1, 1);

    EXPECT_EQ(strings.direction, 1);
    EXPECT_EQ(strings.points_per_string, 3);
    ASSERT_EQ(strings.indices.size(), 6);
    EXPECT_EQ(strings.indices[0], (std::vector<int>{0, 1, 0}));
    EXPECT_EQ(strings.indices[1], (std::vector<int>{2, 3, 2}));
    EXPECT_EQ(strings.indices[5], (std::vector<int>{10, 11, 10}));
}

TEST(KPointStringsTest, BuildsPeriodicStringsAlongY)
{
    const std::array<int, 3> mesh = {{2, 3, 2}};
    const KPointStrings strings = ModuleCell::build_periodic_kpoint_strings(mesh, 2, 1);

    EXPECT_EQ(strings.direction, 2);
    EXPECT_EQ(strings.points_per_string, 4);
    ASSERT_EQ(strings.indices.size(), 4);
    EXPECT_EQ(strings.indices[0], (std::vector<int>{0, 2, 4, 0}));
    EXPECT_EQ(strings.indices[1], (std::vector<int>{1, 3, 5, 1}));
    EXPECT_EQ(strings.indices[3], (std::vector<int>{7, 9, 11, 7}));
}

TEST(KPointStringsTest, BuildsPeriodicStringsAlongZ)
{
    const std::array<int, 3> mesh = {{2, 3, 2}};
    const KPointStrings strings = ModuleCell::build_periodic_kpoint_strings(mesh, 3, 1);

    EXPECT_EQ(strings.direction, 3);
    EXPECT_EQ(strings.points_per_string, 3);
    ASSERT_EQ(strings.indices.size(), 6);
    EXPECT_EQ(strings.indices[0], (std::vector<int>{0, 6, 0}));
    EXPECT_EQ(strings.indices[1], (std::vector<int>{1, 7, 1}));
    EXPECT_EQ(strings.indices[5], (std::vector<int>{5, 11, 5}));
}

TEST(KPointStringsTest, AppendsSpinDownStrings)
{
    const std::array<int, 3> mesh = {{2, 3, 2}};
    const KPointStrings strings = ModuleCell::build_periodic_kpoint_strings(mesh, 2, 2);

    ASSERT_EQ(strings.indices.size(), 8);
    for (int istring = 0; istring < 4; ++istring)
    {
        ASSERT_EQ(strings.indices[istring].size(), strings.indices[istring + 4].size());
        for (std::size_t ipoint = 0; ipoint < strings.indices[istring].size(); ++ipoint)
        {
            EXPECT_EQ(strings.indices[istring + 4][ipoint], strings.indices[istring][ipoint] + 12);
        }
    }
}

TEST(KPointStringsTest, RejectsInvalidTopology)
{
    EXPECT_THROW(ModuleCell::build_periodic_kpoint_strings({{2, 0, 2}}, 1, 1), std::invalid_argument);
    EXPECT_THROW(ModuleCell::build_periodic_kpoint_strings({{2, 2, 2}}, 0, 1), std::invalid_argument);
    EXPECT_THROW(ModuleCell::build_periodic_kpoint_strings({{2, 2, 2}}, 1, 4), std::invalid_argument);
}

} // namespace
