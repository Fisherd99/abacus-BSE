#include "kpoint_strings.h"

#include <stdexcept>

namespace ModuleCell
{

namespace
{

int linear_index(const std::array<int, 3>& mesh, int ix, int iy, int iz)
{
    return ix + mesh[0] * iy + mesh[0] * mesh[1] * iz;
}

} // namespace

KPointStrings build_periodic_kpoint_strings(const std::array<int, 3>& mesh,
                                            const int direction,
                                            const int spin_channels)
{
    if (direction < 1 || direction > 3)
    {
        throw std::invalid_argument("k-point string direction must be 1, 2, or 3");
    }
    if (spin_channels != 1 && spin_channels != 2)
    {
        throw std::invalid_argument("k-point strings support one or two spin channels");
    }
    for (int dimension = 0; dimension < 3; ++dimension)
    {
        if (mesh[dimension] <= 0)
        {
            throw std::invalid_argument("k-point mesh dimensions must be positive");
        }
    }

    const int string_axis = direction - 1;
    const int points_per_string = mesh[string_axis];
    const int kpoints_per_spin = mesh[0] * mesh[1] * mesh[2];
    const int strings_per_spin = kpoints_per_spin / points_per_string;

    KPointStrings result;
    result.direction = direction;
    result.points_per_string = points_per_string + 1;
    result.indices.reserve(strings_per_spin * spin_channels);

    if (direction == 1)
    {
        for (int iz = 0; iz < mesh[2]; ++iz)
        {
            for (int iy = 0; iy < mesh[1]; ++iy)
            {
                std::vector<int> string;
                string.reserve(result.points_per_string);
                for (int ix = 0; ix < mesh[0]; ++ix)
                {
                    string.push_back(linear_index(mesh, ix, iy, iz));
                }
                string.push_back(string.front());
                result.indices.push_back(string);
            }
        }
    }
    else if (direction == 2)
    {
        for (int iz = 0; iz < mesh[2]; ++iz)
        {
            for (int ix = 0; ix < mesh[0]; ++ix)
            {
                std::vector<int> string;
                string.reserve(result.points_per_string);
                for (int iy = 0; iy < mesh[1]; ++iy)
                {
                    string.push_back(linear_index(mesh, ix, iy, iz));
                }
                string.push_back(string.front());
                result.indices.push_back(string);
            }
        }
    }
    else
    {
        for (int iy = 0; iy < mesh[1]; ++iy)
        {
            for (int ix = 0; ix < mesh[0]; ++ix)
            {
                std::vector<int> string;
                string.reserve(result.points_per_string);
                for (int iz = 0; iz < mesh[2]; ++iz)
                {
                    string.push_back(linear_index(mesh, ix, iy, iz));
                }
                string.push_back(string.front());
                result.indices.push_back(string);
            }
        }
    }

    if (spin_channels == 2)
    {
        for (int istring = 0; istring < strings_per_spin; ++istring)
        {
            std::vector<int> spin_down = result.indices[istring];
            for (std::vector<int>::iterator index = spin_down.begin(); index != spin_down.end(); ++index)
            {
                *index += kpoints_per_spin;
            }
            result.indices.push_back(spin_down);
        }
    }

    return result;
}

} // namespace ModuleCell
