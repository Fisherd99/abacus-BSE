#include "../blas_threading.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace
{
using ModuleBase::BlasThreading::ScopedThreadLimit;
using ModuleBase::BlasThreading::get_num_threads;
using ModuleBase::BlasThreading::is_control_available;
using ModuleBase::BlasThreading::set_num_threads;
}

TEST(BlasThreading, RejectsNonPositiveThreadCount)
{
    EXPECT_THROW(ScopedThreadLimit guard(0), std::invalid_argument);
    EXPECT_THROW(set_num_threads(0), std::invalid_argument);
}

TEST(BlasThreading, ScopedLimitRestoresPreviousValue)
{
    if (!is_control_available())
    {
        GTEST_SKIP() << "linked BLAS has no supported thread-control API";
    }
    const int original = get_num_threads();
    const int temporary = original == 1 ? 2 : 1;
    {
        ScopedThreadLimit guard(temporary);
        EXPECT_EQ(get_num_threads(), temporary);
    }
    EXPECT_EQ(get_num_threads(), original);
}

TEST(BlasThreading, NestedLimitsRestoreInStackOrder)
{
    if (!is_control_available())
    {
        GTEST_SKIP() << "linked BLAS has no supported thread-control API";
    }
    const int original = get_num_threads();
    {
        ScopedThreadLimit outer(1);
        EXPECT_EQ(get_num_threads(), 1);
        {
            ScopedThreadLimit inner(2);
            EXPECT_EQ(get_num_threads(), 2);
        }
        EXPECT_EQ(get_num_threads(), 1);
    }
    EXPECT_EQ(get_num_threads(), original);
}
