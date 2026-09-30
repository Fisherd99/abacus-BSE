#ifndef KPOINT_STRINGS_H
#define KPOINT_STRINGS_H

#include <array>
#include <vector>

namespace ModuleCell
{

struct KPointStrings
{
    int direction = 0;
    int points_per_string = 0;
    std::vector<std::vector<int>> indices;
};

KPointStrings build_periodic_kpoint_strings(const std::array<int, 3>& mesh,
                                            int direction,
                                            int spin_channels);

} // namespace ModuleCell

#endif
