#include <gtest/gtest.h>

#include <type_traits>

#include "source_io/module_wannier/fr_overlap.h"
#include "source_lcao/module_finite_field/fr_overlap.h"

static_assert(
    !std::is_same<FR_overlap<double>, FiniteFieldFROverlap<double>>::value,
    "finite-field and Wannier overlap implementations must have distinct types");

TEST(FiniteFieldFROverlapType, DoesNotAliasWannierOverlap)
{
    EXPECT_FALSE((std::is_same<FR_overlap<double>,
                               FiniteFieldFROverlap<double>>::value));
}
