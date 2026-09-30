#include <gtest/gtest.h>

#include "source_lcao/finite_field_lcao.h"

TEST(FiniteFieldLCAODirectionRole, ReportsTransversePolarization)
{
    const hamilt::FiniteFieldLCAODirectionRole role
        = hamilt::finite_field_lcao_direction_role(0.0, true);

    EXPECT_TRUE(role.report_polarization);
    EXPECT_FALSE(role.couple_field);
    EXPECT_FALSE(role.calculate_force_derivatives);
}

TEST(FiniteFieldLCAODirectionRole, ActivatesFieldAndForceTogether)
{
    const hamilt::FiniteFieldLCAODirectionRole force_role
        = hamilt::finite_field_lcao_direction_role(-0.25, true);
    EXPECT_TRUE(force_role.report_polarization);
    EXPECT_TRUE(force_role.couple_field);
    EXPECT_TRUE(force_role.calculate_force_derivatives);

    const hamilt::FiniteFieldLCAODirectionRole energy_role
        = hamilt::finite_field_lcao_direction_role(0.25, false);
    EXPECT_TRUE(energy_role.report_polarization);
    EXPECT_TRUE(energy_role.couple_field);
    EXPECT_FALSE(energy_role.calculate_force_derivatives);
}
