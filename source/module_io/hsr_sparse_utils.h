#ifndef MODULE_IO_HSR_SPARSE_UTILS_H
#define MODULE_IO_HSR_SPARSE_UTILS_H

#include "module_base/vector3.h"
#include "module_basis/module_ao/parallel_orbitals.h"
#include "module_hamilt_lcao/module_hcontainer/hcontainer.h"
#include "module_parameter/parameter.h"

#include <algorithm>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <tuple>
#include <utility>
#include <vector>

namespace ModuleIO
{
namespace detail
{
template <typename T>
struct SparseRBlock
{
    ModuleBase::Vector3<int> r;
    std::vector<T> values;
    std::vector<int> columns;
    std::vector<int> row_ptr;
};

template <typename T>
void write_sparse_value(std::ostream& output, const T& value)
{
    output << " " << std::scientific << std::setprecision(8) << value;
}

template <>
inline void write_sparse_value(std::ostream& output, const std::complex<double>& value)
{
    output << " (" << std::scientific << std::setprecision(8) << value.real() << "," << value.imag() << ")";
}

template <typename T>
void write_serial_matrix(const std::vector<SparseRBlock<T>>& blocks,
                         const std::vector<bool>& output_blocks,
                         const std::string& filename,
                         const std::string& matrix_name,
                         const int step,
                         const bool binary,
                         const bool append)
{
    const int nlocal = PARAM.globalv.nlocal;
    std::ios_base::openmode mode = binary ? std::ios::binary : std::ios::out;
    if (append)
    {
        mode |= std::ios::app;
    }
    std::ofstream output(filename, mode);
    const int block_count = static_cast<int>(std::count(output_blocks.begin(), output_blocks.end(), true));
    if (binary)
    {
        output.write(reinterpret_cast<const char*>(&step), sizeof(int));
        output.write(reinterpret_cast<const char*>(&nlocal), sizeof(int));
        output.write(reinterpret_cast<const char*>(&block_count), sizeof(int));
    }
    else
    {
        output << "STEP: " << step << "\n";
        output << "Matrix Dimension of " << matrix_name << "(R): " << nlocal << "\n";
        output << "Matrix number of " << matrix_name << "(R): " << block_count << "\n";
    }

    for (size_t ir = 0; ir < blocks.size(); ++ir)
    {
        if (!output_blocks[ir])
        {
            continue;
        }
        const auto& block = blocks[ir];
        const int header[4] = {block.r.x, block.r.y, block.r.z, static_cast<int>(block.values.size())};
        if (binary)
        {
            output.write(reinterpret_cast<const char*>(header), sizeof(header));
            if (!block.values.empty())
            {
                output.write(reinterpret_cast<const char*>(block.values.data()), block.values.size() * sizeof(T));
                output.write(reinterpret_cast<const char*>(block.columns.data()), block.columns.size() * sizeof(int));
                output.write(reinterpret_cast<const char*>(block.row_ptr.data()), (nlocal + 1) * sizeof(int));
            }
        }
        else
        {
            output << header[0] << " " << header[1] << " " << header[2] << " " << header[3] << "\n";
            if (block.values.empty())
            {
                continue;
            }
            for (const auto& value: block.values)
            {
                write_sparse_value(output, value);
            }
            output << "\n";
            for (const int column: block.columns)
            {
                output << " " << column;
            }
            output << "\n";
            for (const int offset: block.row_ptr)
            {
                output << " " << offset;
            }
            output << "\n";
        }
    }
}

inline bool less_R(const ModuleBase::Vector3<int>& lhs, const ModuleBase::Vector3<int>& rhs)
{
    if (lhs.x != rhs.x)
    {
        return lhs.x < rhs.x;
    }
    if (lhs.y != rhs.y)
    {
        return lhs.y < rhs.y;
    }
    return lhs.z < rhs.z;
}

inline bool equal_R(const ModuleBase::Vector3<int>& lhs, const ModuleBase::Vector3<int>& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

inline std::vector<ModuleBase::Vector3<int>> merge_R_vectors(
    const std::vector<ModuleBase::Vector3<int>>& lhs,
    const std::vector<ModuleBase::Vector3<int>>& rhs)
{
    std::vector<ModuleBase::Vector3<int>> result;
    result.reserve(lhs.size() + rhs.size());
    std::set_union(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(result), less_R);
    return result;
}

template <typename T>
std::vector<ModuleBase::Vector3<int>> collect_local_R(const hamilt::HContainer<T>& hR)
{
    std::vector<ModuleBase::Vector3<int>> r_vectors;
    const size_t nr = hR.size_R_loop();
    r_vectors.reserve(nr);
    for (size_t ir = 0; ir < nr; ++ir)
    {
        int rx = 0;
        int ry = 0;
        int rz = 0;
        hR.loop_R(ir, rx, ry, rz);
        r_vectors.emplace_back(rx, ry, rz);
    }
    std::sort(r_vectors.begin(), r_vectors.end(), less_R);
    r_vectors.erase(std::unique(r_vectors.begin(), r_vectors.end(), equal_R), r_vectors.end());
    return r_vectors;
}

// Visit every locally stored matrix element that survives the sparse cutoff.
// Consumers decide whether to assemble local CSR rows or send the entry to
// the MPI rank that owns its global row.
template <typename T, typename Visitor>
void for_each_sparse_entry(const hamilt::HContainer<T>& hR,
                           const Parallel_Orbitals& pv,
                           const std::vector<ModuleBase::Vector3<int>>& r_vectors,
                           const double sparse_thr,
                           Visitor&& visitor)
{
    for (int iap = 0; iap < hR.size_atom_pairs(); ++iap)
    {
        const auto& atom_pair = hR.get_atom_pair(iap);
        for (int local_ir = 0; local_ir < atom_pair.get_R_size(); ++local_ir)
        {
            const auto r = atom_pair.get_R_index(local_ir);
            const auto r_iter = std::lower_bound(r_vectors.begin(), r_vectors.end(), r, less_R);
            if (r_iter == r_vectors.end() || !equal_R(*r_iter, r))
            {
                continue;
            }
            const int ir = static_cast<int>(std::distance(r_vectors.begin(), r_iter));
            const auto matrix_info = atom_pair.get_matrix_values(local_ir);
            const auto& indexes = std::get<0>(matrix_info);
            const T* data = std::get<1>(matrix_info);
            for (int local_row = 0; local_row < indexes[1]; ++local_row)
            {
                const int row = pv.local2global_row(indexes[0] + local_row);
                for (int local_col = 0; local_col < indexes[3]; ++local_col)
                {
                    const T value = data[local_row * indexes[3] + local_col];
                    if (std::abs(value) > sparse_thr)
                    {
                        visitor(ir, row, pv.local2global_col(indexes[2] + local_col), value);
                    }
                }
            }
        }
    }
}

template <typename T>
std::vector<SparseRBlock<T>> make_sparse_blocks(const hamilt::HContainer<T>& hR,
                                                const Parallel_Orbitals& pv,
                                                const std::vector<ModuleBase::Vector3<int>>& r_vectors,
                                                const int nlocal,
                                                const double sparse_thr)
{
    std::vector<SparseRBlock<T>> blocks(r_vectors.size());
    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        blocks[ir].r = r_vectors[ir];
        blocks[ir].row_ptr.assign(nlocal + 1, 0);
    }
    // rows[R index][global row] stores the nonzeros in that row as
    // (global column, matrix value) pairs before they are flattened to CSR.
    std::vector<std::vector<std::vector<std::pair<int, T>>>> rows(r_vectors.size(),
                                                                  std::vector<std::vector<std::pair<int, T>>>(nlocal));
    for_each_sparse_entry(hR,
                          pv,
                          r_vectors,
                          sparse_thr,
                          [&](const int ir, const int row, const int column, const T& value) {
                              rows[ir][row].emplace_back(column, value);
                          });
    for (auto& block_rows: rows)
    {
        for (auto& row: block_rows)
        {
            std::sort(row.begin(), row.end(), [](const std::pair<int, T>& lhs, const std::pair<int, T>& rhs) {
                return lhs.first < rhs.first;
            });
        }
    }

    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        auto& block = blocks[ir];
        block.row_ptr.clear();
        block.row_ptr.push_back(0);
        for (const auto& row: rows[ir])
        {
            for (const auto& entry: row)
            {
                block.columns.push_back(entry.first);
                block.values.push_back(entry.second);
            }
            block.row_ptr.push_back(static_cast<int>(block.values.size()));
        }
    }
    return blocks;
}

inline std::vector<SparseRBlock<std::complex<double>>> make_td_sparse_blocks(
    const hamilt::HContainer<double>& hR,
    const hamilt::HContainer<std::complex<double>>& td_hR,
    const Parallel_Orbitals& pv,
    const std::vector<ModuleBase::Vector3<int>>& r_vectors,
    const int nlocal,
    const double sparse_thr)
{
    const auto base_blocks = make_sparse_blocks(hR, pv, r_vectors, nlocal, sparse_thr);
    // Preserve the threshold used when velocity-gauge HR was cached.
    const auto td_blocks = make_sparse_blocks(td_hR, pv, r_vectors, nlocal, 1.0e-10);
    std::vector<SparseRBlock<std::complex<double>>> result(r_vectors.size());
    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        auto& output = result[ir];
        output.r = r_vectors[ir];
        output.row_ptr.push_back(0);
        for (int row = 0; row < nlocal; ++row)
        {
            int base_pos = base_blocks[ir].row_ptr[row];
            const int base_end = base_blocks[ir].row_ptr[row + 1];
            int td_pos = td_blocks[ir].row_ptr[row];
            const int td_end = td_blocks[ir].row_ptr[row + 1];
            while (base_pos < base_end || td_pos < td_end)
            {
                int column = 0;
                std::complex<double> value = 0.0;
                if (td_pos == td_end
                    || (base_pos < base_end
                        && base_blocks[ir].columns[base_pos] < td_blocks[ir].columns[td_pos]))
                {
                    column = base_blocks[ir].columns[base_pos];
                    value = base_blocks[ir].values[base_pos++];
                }
                else if (base_pos == base_end
                         || td_blocks[ir].columns[td_pos] < base_blocks[ir].columns[base_pos])
                {
                    column = td_blocks[ir].columns[td_pos];
                    value = td_blocks[ir].values[td_pos++];
                }
                else
                {
                    column = base_blocks[ir].columns[base_pos];
                    value = base_blocks[ir].values[base_pos++] + td_blocks[ir].values[td_pos++];
                }
                if (std::abs(value) > sparse_thr)
                {
                    output.columns.push_back(column);
                    output.values.push_back(value);
                }
            }
            output.row_ptr.push_back(static_cast<int>(output.values.size()));
        }
    }
    return result;
}
} // namespace detail
} // namespace ModuleIO

#endif // MODULE_IO_HSR_SPARSE_UTILS_H
