#include "write_HS_R.h"

#include "module_parameter/parameter.h"
#include "module_base/timer.h"
#include "module_hamilt_lcao/hamilt_lcaodft/LCAO_HS_arrays.hpp"
#include "module_hamilt_lcao/hamilt_lcaodft/spar_dh.h"
#include "module_hamilt_lcao/hamilt_lcaodft/spar_hsr.h"
#include "module_hamilt_lcao/hamilt_lcaodft/spar_st.h"
#include "module_hamilt_lcao/module_dftu/dftu.h"
#include "module_hamilt_lcao/module_hcontainer/transfer.h"
#include "module_hamilt_lcao/module_tddft/td_velocity.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "write_HS_sparse.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <iomanip>
#include <iterator>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace
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
void write_sparse_value(std::ostream& ofs, const T& value)
{
    ofs << " " << std::scientific << std::setprecision(8) << value;
}

template <>
void write_sparse_value(std::ostream& ofs, const std::complex<double>& value)
{
    ofs << " (" << std::scientific << std::setprecision(8) << value.real() << "," << value.imag()
        << ")";
}

bool less_R(const ModuleBase::Vector3<int>& lhs, const ModuleBase::Vector3<int>& rhs)
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

bool equal_R(const ModuleBase::Vector3<int>& lhs, const ModuleBase::Vector3<int>& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

std::vector<ModuleBase::Vector3<int>> merge_R_vectors(
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

template <typename T>
std::vector<SparseRBlock<T>> make_sparse_blocks(
    const hamilt::HContainer<T>& hR,
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
    std::vector<std::vector<std::vector<std::pair<int, T>>>> rows(
        r_vectors.size(),
        std::vector<std::vector<std::pair<int, T>>>(nlocal));
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
            const size_t ir = std::distance(r_vectors.begin(), r_iter);
            const auto matrix_info = atom_pair.get_matrix_values(local_ir);
            const auto& indexes = std::get<0>(matrix_info);
            const T* data = std::get<1>(matrix_info);
            for (int irow = 0; irow < indexes[1]; ++irow)
            {
                const int row_index = pv.local2global_row(indexes[0] + irow);
                auto& row = rows[ir][row_index];
                for (int icol = 0; icol < indexes[3]; ++icol)
                {
                    const T value = data[irow * indexes[3] + icol];
                    if (std::abs(value) > sparse_thr)
                    {
                        row.emplace_back(pv.local2global_col(indexes[2] + icol), value);
                    }
                }
            }
        }
    }
    for (auto& block_rows : rows)
    {
        for (auto& row : block_rows)
        {
            std::sort(row.begin(), row.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.first < rhs.first;
            });
        }
    }

    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        auto& block = blocks[ir];
        block.row_ptr.clear();
        block.row_ptr.push_back(0);
        for (auto& row : rows[ir])
        {
            for (const auto& entry : row)
            {
                block.columns.push_back(entry.first);
                block.values.push_back(entry.second);
            }
            block.row_ptr.push_back(static_cast<int>(block.values.size()));
        }
    }
    return blocks;
}

std::vector<SparseRBlock<std::complex<double>>> make_td_sparse_blocks(
    const hamilt::HContainer<double>& hR,
    const hamilt::HContainer<std::complex<double>>& td_hR,
    const Parallel_Orbitals& pv,
    const std::vector<ModuleBase::Vector3<int>>& r_vectors,
    const int nlocal,
    const double sparse_thr)
{
    const auto base_blocks = make_sparse_blocks(hR, pv, r_vectors, nlocal, sparse_thr);
    // Preserve the threshold used when velocity-gauge HR was cached in
    // TD_Velocity::HR_sparse_td_vel. The combined value is filtered again
    // with the user-selected sparse threshold below.
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
                    || (base_pos < base_end && base_blocks[ir].columns[base_pos] < td_blocks[ir].columns[td_pos]))
                {
                    column = base_blocks[ir].columns[base_pos];
                    value = base_blocks[ir].values[base_pos++];
                }
                else if (base_pos == base_end || td_blocks[ir].columns[td_pos] < base_blocks[ir].columns[base_pos])
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
    std::ofstream ofs(filename, mode);
    const int block_count = static_cast<int>(std::count(output_blocks.begin(), output_blocks.end(), true));
    if (binary)
    {
        ofs.write(reinterpret_cast<const char*>(&step), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&nlocal), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&block_count), sizeof(int));
    }
    else
    {
        ofs << "STEP: " << step << "\n";
        ofs << "Matrix Dimension of " << matrix_name << "(R): " << nlocal << "\n";
        ofs << "Matrix number of " << matrix_name << "(R): " << block_count << "\n";
    }

    for (size_t ir = 0; ir < blocks.size(); ++ir)
    {
        if (!output_blocks[ir])
        {
            continue;
        }
        const auto& block = blocks[ir];
        const int rx = block.r.x;
        const int ry = block.r.y;
        const int rz = block.r.z;
        const int nnz = static_cast<int>(block.values.size());
        if (binary)
        {
            ofs.write(reinterpret_cast<const char*>(&rx), sizeof(int));
            ofs.write(reinterpret_cast<const char*>(&ry), sizeof(int));
            ofs.write(reinterpret_cast<const char*>(&rz), sizeof(int));
            ofs.write(reinterpret_cast<const char*>(&nnz), sizeof(int));
            if (nnz > 0)
            {
                ofs.write(reinterpret_cast<const char*>(block.values.data()), nnz * sizeof(T));
                ofs.write(reinterpret_cast<const char*>(block.columns.data()), nnz * sizeof(int));
                ofs.write(reinterpret_cast<const char*>(block.row_ptr.data()),
                          (nlocal + 1) * sizeof(int));
            }
        }
        else
        {
            ofs << rx << " " << ry << " " << rz << " " << nnz << "\n";
            if (nnz == 0)
            {
                continue;
            }
            for (const auto& value : block.values)
            {
                write_sparse_value(ofs, value);
            }
            ofs << "\n";
            for (const int column : block.columns)
            {
                ofs << " " << column;
            }
            ofs << "\n";
            for (const int offset : block.row_ptr)
            {
                ofs << " " << offset;
            }
            ofs << "\n";
        }
    }
}

#ifdef __MPI
struct RowRange
{
    int begin;
    int end;
};

RowRange get_row_range(const int nlocal, const int rank, const int nproc)
{
    const int quotient = nlocal / nproc;
    const int remainder = nlocal % nproc;
    const int begin = rank * quotient + std::min(rank, remainder);
    return {begin, begin + quotient + (rank < remainder ? 1 : 0)};
}

int get_row_owner(const int row, const int nlocal, const int nproc)
{
    const int quotient = nlocal / nproc;
    const int remainder = nlocal % nproc;
    const int long_rows = (quotient + 1) * remainder;
    if (row < long_rows)
    {
        return row / (quotient + 1);
    }
    return remainder + (row - long_rows) / quotient;
}

template <typename T>
std::vector<ModuleBase::Vector3<int>> collect_global_R(const hamilt::HContainer<T>& hR)
{
    int nproc = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    std::vector<int> local_R;
    const size_t nr = hR.size_R_loop();
    local_R.reserve(3 * nr);
    for (size_t ir = 0; ir < nr; ++ir)
    {
        int rx = 0;
        int ry = 0;
        int rz = 0;
        hR.loop_R(ir, rx, ry, rz);
        local_R.insert(local_R.end(), {rx, ry, rz});
    }

    const int local_size = static_cast<int>(local_R.size());
    std::vector<int> counts(nproc);
    MPI_Allgather(&local_size, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> displs(nproc, 0);
    for (int ip = 1; ip < nproc; ++ip)
    {
        displs[ip] = displs[ip - 1] + counts[ip - 1];
    }
    std::vector<int> all_R(displs.back() + counts.back());
    MPI_Allgatherv(local_R.data(),
                   local_size,
                   MPI_INT,
                   all_R.data(),
                   counts.data(),
                   displs.data(),
                   MPI_INT,
                   MPI_COMM_WORLD);

    std::vector<ModuleBase::Vector3<int>> result;
    result.reserve(all_R.size() / 3);
    for (size_t i = 0; i < all_R.size(); i += 3)
    {
        result.emplace_back(all_R[i], all_R[i + 1], all_R[i + 2]);
    }
    std::sort(result.begin(), result.end(), less_R);
    result.erase(std::unique(result.begin(), result.end(), equal_R), result.end());
    return result;
}

template <typename T>
struct DistributedEntry
{
    int ir;
    int row;
    int column;
    T value;
};

template <typename T>
struct DistributedRBlock
{
    std::vector<T> values;
    std::vector<int> columns;
    std::vector<int> row_counts;
};

template <typename T>
std::vector<DistributedRBlock<T>> redistribute_entries(
    const std::vector<std::vector<DistributedEntry<T>>>& outgoing,
    const std::vector<ModuleBase::Vector3<int>>& r_vectors,
    const RowRange row_range)
{
    int nproc = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    std::vector<int> send_counts(nproc), recv_counts(nproc), send_displs(nproc, 0), recv_displs(nproc, 0);
    for (int ip = 0; ip < nproc; ++ip)
    {
        send_counts[ip] = static_cast<int>(outgoing[ip].size());
    }
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int ip = 1; ip < nproc; ++ip)
    {
        send_displs[ip] = send_displs[ip - 1] + send_counts[ip - 1];
        recv_displs[ip] = recv_displs[ip - 1] + recv_counts[ip - 1];
    }
    const int send_size = send_displs.back() + send_counts.back();
    const int recv_size = recv_displs.back() + recv_counts.back();
    std::vector<int> send_meta(3 * send_size), recv_meta(3 * recv_size);
    std::vector<T> send_values(send_size), recv_values(recv_size);
    for (int ip = 0; ip < nproc; ++ip)
    {
        int index = send_displs[ip];
        for (const auto& entry : outgoing[ip])
        {
            send_meta[3 * index] = entry.ir;
            send_meta[3 * index + 1] = entry.row;
            send_meta[3 * index + 2] = entry.column;
            send_values[index] = entry.value;
            ++index;
        }
    }
    std::vector<int> send_counts_meta(nproc), recv_counts_meta(nproc), send_displs_meta(nproc), recv_displs_meta(nproc);
    for (int ip = 0; ip < nproc; ++ip)
    {
        send_counts_meta[ip] = 3 * send_counts[ip];
        recv_counts_meta[ip] = 3 * recv_counts[ip];
        send_displs_meta[ip] = 3 * send_displs[ip];
        recv_displs_meta[ip] = 3 * recv_displs[ip];
    }
    MPI_Alltoallv(send_meta.data(),
                  send_counts_meta.data(),
                  send_displs_meta.data(),
                  MPI_INT,
                  recv_meta.data(),
                  recv_counts_meta.data(),
                  recv_displs_meta.data(),
                  MPI_INT,
                  MPI_COMM_WORLD);
    MPI_Alltoallv(send_values.data(),
                  send_counts.data(),
                  send_displs.data(),
                  MPITraits<T>::datatype(),
                  recv_values.data(),
                  recv_counts.data(),
                  recv_displs.data(),
                  MPITraits<T>::datatype(),
                  MPI_COMM_WORLD);

    std::vector<DistributedEntry<T>> received(recv_size);
    for (int i = 0; i < recv_size; ++i)
    {
        received[i] = {recv_meta[3 * i], recv_meta[3 * i + 1], recv_meta[3 * i + 2], recv_values[i]};
    }
    std::stable_sort(received.begin(), received.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.ir != rhs.ir)
        {
            return lhs.ir < rhs.ir;
        }
        if (lhs.row != rhs.row)
        {
            return lhs.row < rhs.row;
        }
        return lhs.column < rhs.column;
    });

    std::vector<DistributedRBlock<T>> blocks(r_vectors.size());
    for (auto& block : blocks)
    {
        block.row_counts.assign(row_range.end - row_range.begin, 0);
    }
    for (const auto& entry : received)
    {
        auto& block = blocks[entry.ir];
        block.values.push_back(entry.value);
        block.columns.push_back(entry.column);
        ++block.row_counts[entry.row - row_range.begin];
    }
    return blocks;
}

template <typename T>
std::vector<DistributedRBlock<T>> redistribute_sparse_rows(
    const hamilt::HContainer<T>& hR,
    const Parallel_Orbitals& pv,
    const std::vector<ModuleBase::Vector3<int>>& r_vectors,
    const int nlocal,
    const double sparse_thr,
    const RowRange row_range)
{
    int nproc = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    std::vector<std::vector<DistributedEntry<T>>> outgoing(nproc);

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
                const int owner = get_row_owner(row, nlocal, nproc);
                for (int local_col = 0; local_col < indexes[3]; ++local_col)
                {
                    const T value = data[local_row * indexes[3] + local_col];
                    if (std::abs(value) > sparse_thr)
                    {
                        const int column = pv.local2global_col(indexes[2] + local_col);
                        outgoing[owner].push_back({ir, row, column, value});
                    }
                }
            }
        }
    }

    return redistribute_entries(outgoing, r_vectors, row_range);
}

struct FileFragments
{
    std::vector<char> data;
    std::vector<int> lengths;
    std::vector<MPI_Aint> offsets;

    void append(const MPI_Offset offset, const void* source, const size_t size)
    {
        if (size == 0)
        {
            return;
        }
        if (size > INT_MAX || data.size() + size > INT_MAX)
        {
            throw std::runtime_error("Distributed CSR output exceeds the MPI count limit");
        }
        const size_t old_size = data.size();
        data.resize(old_size + size);
        std::memcpy(data.data() + old_size, source, size);
        lengths.push_back(static_cast<int>(size));
        offsets.push_back(static_cast<MPI_Aint>(offset));
    }

    void append(const MPI_Offset offset, const std::string& source)
    {
        append(offset, source.data(), source.size());
    }
};

void write_MPI_fragments(const std::string& filename,
                         const MPI_Offset file_size,
                         const FileFragments& fragments,
                         const bool append = false)
{
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD,
                  const_cast<char*>(filename.c_str()),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL,
                  &file);
    MPI_Offset base_offset = 0;
    if (append)
    {
        MPI_File_get_size(file, &base_offset);
    }
    MPI_File_set_size(file, base_offset + file_size);

    MPI_Datatype filetype = MPI_DATATYPE_NULL;
    if (fragments.lengths.empty())
    {
        MPI_File_set_view(file, 0, MPI_BYTE, MPI_BYTE, const_cast<char*>("native"), MPI_INFO_NULL);
    }
    else
    {
        std::vector<MPI_Aint> offsets = fragments.offsets;
        for (auto& offset : offsets)
        {
            offset += base_offset;
        }
        MPI_Type_create_hindexed(static_cast<int>(fragments.lengths.size()),
                                 fragments.lengths.data(),
                                 offsets.data(),
                                 MPI_BYTE,
                                 &filetype);
        MPI_Type_commit(&filetype);
        MPI_File_set_view(file, 0, MPI_BYTE, filetype, const_cast<char*>("native"), MPI_INFO_NULL);
    }
    MPI_File_write_all(file,
                       fragments.data.empty() ? nullptr : fragments.data.data(),
                       static_cast<int>(fragments.data.size()),
                       MPI_BYTE,
                       MPI_STATUS_IGNORE);
    if (filetype != MPI_DATATYPE_NULL)
    {
        MPI_Type_free(&filetype);
    }
    MPI_File_close(&file);
}

template <typename T>
void write_distributed_binary_matrix(const std::vector<ModuleBase::Vector3<int>>& r_vectors,
                                     const std::vector<DistributedRBlock<T>>& blocks,
                                     const std::vector<int>& global_nnz,
                                     const std::vector<int>& nnz_by_rank,
                                     const std::vector<bool>& output_blocks,
                                     const RowRange row_range,
                                     const int nlocal,
                                     const int rank,
                                     const int nproc,
                                     const int step,
                                     const std::string& filename,
                                     const bool append)
{
    const int block_count = static_cast<int>(std::count(output_blocks.begin(), output_blocks.end(), true));
    FileFragments fragments;
    MPI_Offset offset = 0;
    const int file_header[3] = {step, nlocal, block_count};
    if (rank == 0)
    {
        fragments.append(offset, file_header, sizeof(file_header));
    }
    offset += sizeof(file_header);

    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        if (!output_blocks[ir])
        {
            continue;
        }
        const int block_header[4] = {r_vectors[ir].x, r_vectors[ir].y, r_vectors[ir].z, global_nnz[ir]};
        if (rank == 0)
        {
            fragments.append(offset, block_header, sizeof(block_header));
        }
        offset += sizeof(block_header);

        int rank_nnz_offset = 0;
        for (int ip = 0; ip < rank; ++ip)
        {
            rank_nnz_offset += nnz_by_rank[ip * r_vectors.size() + ir];
        }
        fragments.append(offset + static_cast<MPI_Offset>(rank_nnz_offset) * sizeof(T),
                         blocks[ir].values.data(),
                         blocks[ir].values.size() * sizeof(T));
        offset += static_cast<MPI_Offset>(global_nnz[ir]) * sizeof(T);
        fragments.append(offset + static_cast<MPI_Offset>(rank_nnz_offset) * sizeof(int),
                         blocks[ir].columns.data(),
                         blocks[ir].columns.size() * sizeof(int));
        offset += static_cast<MPI_Offset>(global_nnz[ir]) * sizeof(int);

        if (global_nnz[ir] > 0)
        {
            std::vector<int> row_ptr;
            row_ptr.reserve(blocks[ir].row_counts.size() + 1);
            int row_offset = rank_nnz_offset;
            for (const int count : blocks[ir].row_counts)
            {
                row_ptr.push_back(row_offset);
                row_offset += count;
            }
            if (row_range.end == nlocal && row_range.begin < row_range.end)
            {
                row_ptr.push_back(global_nnz[ir]);
            }
            fragments.append(offset + static_cast<MPI_Offset>(row_range.begin) * sizeof(int),
                             row_ptr.data(),
                             row_ptr.size() * sizeof(int));
            offset += static_cast<MPI_Offset>(nlocal + 1) * sizeof(int);
        }
    }
    write_MPI_fragments(filename, offset, fragments, append);
}

template <typename T>
void write_distributed_text_matrix(const std::vector<ModuleBase::Vector3<int>>& r_vectors,
                                   const std::vector<DistributedRBlock<T>>& blocks,
                                   const std::vector<int>& global_nnz,
                                   const std::vector<int>& nnz_by_rank,
                                   const std::vector<bool>& output_blocks,
                                   const RowRange row_range,
                                   const int nlocal,
                                   const int rank,
                                   const int nproc,
                                   const int step,
                                   const std::string& matrix_name,
                                   const std::string& filename,
                                   const bool append)
{
    std::vector<std::string> values_text(r_vectors.size());
    std::vector<std::string> columns_text(r_vectors.size());
    std::vector<std::string> row_ptr_text(r_vectors.size());
    std::vector<unsigned long long> local_lengths(3 * r_vectors.size(), 0);
    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        if (!output_blocks[ir])
        {
            continue;
        }
        if (global_nnz[ir] == 0)
        {
            continue;
        }
        std::ostringstream values_stream;
        for (const auto& value : blocks[ir].values)
        {
            write_sparse_value(values_stream, value);
        }
        values_text[ir] = values_stream.str();

        std::ostringstream columns_stream;
        for (const int column : blocks[ir].columns)
        {
            columns_stream << " " << column;
        }
        columns_text[ir] = columns_stream.str();

        int row_offset = 0;
        for (int ip = 0; ip < rank; ++ip)
        {
            row_offset += nnz_by_rank[ip * r_vectors.size() + ir];
        }
        std::ostringstream row_ptr_stream;
        for (const int count : blocks[ir].row_counts)
        {
            row_ptr_stream << " " << row_offset;
            row_offset += count;
        }
        if (row_range.end == nlocal && row_range.begin < row_range.end)
        {
            row_ptr_stream << " " << global_nnz[ir];
        }
        row_ptr_text[ir] = row_ptr_stream.str();
        local_lengths[3 * ir] = values_text[ir].size();
        local_lengths[3 * ir + 1] = columns_text[ir].size();
        local_lengths[3 * ir + 2] = row_ptr_text[ir].size();
    }

    std::vector<unsigned long long> lengths_by_rank(nproc * local_lengths.size());
    MPI_Allgather(local_lengths.data(),
                  static_cast<int>(local_lengths.size()),
                  MPI_UNSIGNED_LONG_LONG,
                  lengths_by_rank.data(),
                  static_cast<int>(local_lengths.size()),
                  MPI_UNSIGNED_LONG_LONG,
                  MPI_COMM_WORLD);

    const int block_count = static_cast<int>(std::count(output_blocks.begin(), output_blocks.end(), true));
    std::ostringstream header_stream;
    header_stream << "STEP: " << step << "\n";
    header_stream << "Matrix Dimension of " << matrix_name << "(R): " << nlocal << "\n";
    header_stream << "Matrix number of " << matrix_name << "(R): " << block_count << "\n";
    const std::string header = header_stream.str();

    FileFragments fragments;
    MPI_Offset offset = 0;
    if (rank == 0)
    {
        fragments.append(offset, header);
    }
    offset += header.size();

    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        if (!output_blocks[ir])
        {
            continue;
        }
        std::ostringstream block_header_stream;
        block_header_stream << r_vectors[ir].x << " " << r_vectors[ir].y << " " << r_vectors[ir].z << " "
                            << global_nnz[ir] << "\n";
        const std::string block_header = block_header_stream.str();
        if (rank == 0)
        {
            fragments.append(offset, block_header);
        }
        offset += block_header.size();
        if (global_nnz[ir] == 0)
        {
            continue;
        }

        for (int section = 0; section < 3; ++section)
        {
            unsigned long long rank_offset = 0;
            unsigned long long total_length = 0;
            for (int ip = 0; ip < nproc; ++ip)
            {
                const auto length = lengths_by_rank[ip * local_lengths.size() + 3 * ir + section];
                if (ip < rank)
                {
                    rank_offset += length;
                }
                total_length += length;
            }
            const std::string* text = section == 0 ? &values_text[ir]
                                    : section == 1 ? &columns_text[ir]
                                                   : &row_ptr_text[ir];
            fragments.append(offset + rank_offset, *text);
            offset += total_length;
            if (rank == 0)
            {
                fragments.append(offset, "\n", 1);
            }
            ++offset;
        }
    }
    write_MPI_fragments(filename, offset, fragments, append);
}

template <typename T>
struct DistributedSparseMatrix
{
    std::vector<DistributedRBlock<T>> blocks;
    std::vector<int> global_nnz;
    std::vector<int> nnz_by_rank;
};

void calculate_dftu_R(const int spin, double* sr, double* hr)
{
    GlobalC::dftu.cal_eff_pot_mat_R_double(spin, sr, hr);
}

void calculate_dftu_R(const int spin, std::complex<double>* sr, std::complex<double>* hr)
{
    GlobalC::dftu.cal_eff_pot_mat_R_complex_double(spin, sr, hr);
}

DistributedSparseMatrix<std::complex<double>> prepare_td_distributed_matrix(
    const hamilt::HContainer<double>& hR,
    const hamilt::HContainer<std::complex<double>>& td_hR,
    const Parallel_Orbitals& pv,
    const std::vector<ModuleBase::Vector3<int>>& r_vectors,
    const double sparse_thr)
{
    int rank = 0;
    int nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    const int nlocal = PARAM.globalv.nlocal;
    const RowRange row_range = get_row_range(nlocal, rank, nproc);
    const auto base_blocks = redistribute_sparse_rows(hR, pv, r_vectors, nlocal, sparse_thr, row_range);
    // TD velocity-gauge matrices were historically cached at this threshold.
    const auto td_blocks = redistribute_sparse_rows(td_hR, pv, r_vectors, nlocal, 1.0e-10, row_range);

    DistributedSparseMatrix<std::complex<double>> result;
    result.blocks.resize(r_vectors.size());
    std::vector<int> local_nnz(r_vectors.size());
    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        auto& output = result.blocks[ir];
        output.row_counts.resize(row_range.end - row_range.begin, 0);
        size_t base_pos = 0;
        size_t td_pos = 0;
        for (size_t local_row = 0; local_row < output.row_counts.size(); ++local_row)
        {
            const size_t base_end = base_pos + base_blocks[ir].row_counts[local_row];
            const size_t td_end = td_pos + td_blocks[ir].row_counts[local_row];
            while (base_pos < base_end || td_pos < td_end)
            {
                int column = 0;
                std::complex<double> value = 0.0;
                if (td_pos == td_end
                    || (base_pos < base_end && base_blocks[ir].columns[base_pos] < td_blocks[ir].columns[td_pos]))
                {
                    column = base_blocks[ir].columns[base_pos];
                    value = base_blocks[ir].values[base_pos++];
                }
                else if (base_pos == base_end || td_blocks[ir].columns[td_pos] < base_blocks[ir].columns[base_pos])
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
                    ++output.row_counts[local_row];
                }
            }
        }
        local_nnz[ir] = static_cast<int>(output.values.size());
    }

    result.global_nnz.resize(r_vectors.size());
    MPI_Allreduce(local_nnz.data(),
                  result.global_nnz.data(),
                  static_cast<int>(r_vectors.size()),
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    result.nnz_by_rank.resize(nproc * r_vectors.size());
    MPI_Allgather(local_nnz.data(),
                  static_cast<int>(r_vectors.size()),
                  MPI_INT,
                  result.nnz_by_rank.data(),
                  static_cast<int>(r_vectors.size()),
                  MPI_INT,
                  MPI_COMM_WORLD);
    return result;
}

template <typename T>
DistributedSparseMatrix<T> prepare_dftu_distributed_matrix(
    const hamilt::HContainer<T>& hR,
    const hamilt::HContainer<T>& sR,
    const Parallel_Orbitals& pv,
    const std::vector<ModuleBase::Vector3<int>>& r_vectors,
    const int spin,
    const double sparse_thr)
{
    int rank = 0;
    int nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    const int nlocal = PARAM.globalv.nlocal;
    const RowRange row_range = get_row_range(nlocal, rank, nproc);
    const auto base_blocks = redistribute_sparse_rows(hR, pv, r_vectors, nlocal, sparse_thr, row_range);

    // Reproduce the legacy DFT+U output formula using the native block-cyclic
    // matrix layout required by ScaLAPACK. S(R) is filtered before the
    // multiplication, exactly as in sparse_format::cal_HR_dftu[_soc].
    std::vector<std::vector<std::pair<int, T>>> local_s_entries(r_vectors.size());
    std::vector<int> local_s_nnz(r_vectors.size(), 0);
    for (int iap = 0; iap < sR.size_atom_pairs(); ++iap)
    {
        const auto& atom_pair = sR.get_atom_pair(iap);
        for (int local_ir = 0; local_ir < atom_pair.get_R_size(); ++local_ir)
        {
            const auto r = atom_pair.get_R_index(local_ir);
            const auto r_iter = std::lower_bound(r_vectors.begin(), r_vectors.end(), r, less_R);
            if (r_iter == r_vectors.end() || !equal_R(*r_iter, r))
            {
                continue;
            }
            const size_t ir = std::distance(r_vectors.begin(), r_iter);
            const auto matrix_info = atom_pair.get_matrix_values(local_ir);
            const auto& indexes = std::get<0>(matrix_info);
            const T* data = std::get<1>(matrix_info);
            for (int local_row = 0; local_row < indexes[1]; ++local_row)
            {
                for (int local_col = 0; local_col < indexes[3]; ++local_col)
                {
                    const T value = data[local_row * indexes[3] + local_col];
                    if (std::abs(value) <= sparse_thr)
                    {
                        continue;
                    }
                    const int row = indexes[0] + local_row;
                    const int col = indexes[2] + local_col;
                    const int index = ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER(PARAM.inp.ks_solver)
                                          ? row + col * pv.nrow
                                          : row * pv.ncol + col;
                    local_s_entries[ir].emplace_back(index, value);
                    ++local_s_nnz[ir];
                }
            }
        }
    }
    std::vector<int> global_s_nnz(r_vectors.size(), 0);
    MPI_Allreduce(local_s_nnz.data(),
                  global_s_nnz.data(),
                  static_cast<int>(r_vectors.size()),
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    std::vector<std::vector<DistributedEntry<T>>> outgoing(nproc);
    std::vector<T> sr_tmp(pv.nloc);
    std::vector<T> hu_tmp(pv.nloc);
    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        if (global_s_nnz[ir] == 0)
        {
            continue;
        }
        std::fill(sr_tmp.begin(), sr_tmp.end(), T{});
        std::fill(hu_tmp.begin(), hu_tmp.end(), T{});
        for (const auto& entry : local_s_entries[ir])
        {
            sr_tmp[entry.first] = entry.second;
        }
        calculate_dftu_R(spin, sr_tmp.data(), hu_tmp.data());

        for (int row = 0; row < nlocal; ++row)
        {
            const int local_row = pv.global2local_row(row);
            if (local_row < 0)
            {
                continue;
            }
            const int owner = get_row_owner(row, nlocal, nproc);
            for (int col = 0; col < nlocal; ++col)
            {
                const int local_col = pv.global2local_col(col);
                if (local_col < 0)
                {
                    continue;
                }
                const int index = ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER(PARAM.inp.ks_solver)
                                      ? local_row + local_col * pv.nrow
                                      : local_row * pv.ncol + local_col;
                if (std::abs(hu_tmp[index]) > sparse_thr)
                {
                    outgoing[owner].push_back(
                        {static_cast<int>(ir), row, col, hu_tmp[index]});
                }
            }
        }
    }
    const auto u_blocks = redistribute_entries(outgoing, r_vectors, row_range);

    DistributedSparseMatrix<T> result;
    result.blocks.resize(r_vectors.size());
    std::vector<int> local_nnz(r_vectors.size(), 0);
    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        auto& output = result.blocks[ir];
        output.row_counts.resize(row_range.end - row_range.begin, 0);
        size_t base_pos = 0;
        size_t u_pos = 0;
        for (size_t local_row = 0; local_row < output.row_counts.size(); ++local_row)
        {
            const size_t base_end = base_pos + base_blocks[ir].row_counts[local_row];
            const size_t u_end = u_pos + u_blocks[ir].row_counts[local_row];
            while (base_pos < base_end || u_pos < u_end)
            {
                int column = 0;
                T value{};
                if (u_pos == u_end
                    || (base_pos < base_end && base_blocks[ir].columns[base_pos] < u_blocks[ir].columns[u_pos]))
                {
                    column = base_blocks[ir].columns[base_pos];
                    value = base_blocks[ir].values[base_pos++];
                }
                else if (base_pos == base_end || u_blocks[ir].columns[u_pos] < base_blocks[ir].columns[base_pos])
                {
                    column = u_blocks[ir].columns[u_pos];
                    value = u_blocks[ir].values[u_pos++];
                }
                else
                {
                    column = base_blocks[ir].columns[base_pos];
                    value = base_blocks[ir].values[base_pos++] + u_blocks[ir].values[u_pos++];
                }
                if (std::abs(value) > sparse_thr)
                {
                    output.columns.push_back(column);
                    output.values.push_back(value);
                    ++output.row_counts[local_row];
                }
            }
        }
        local_nnz[ir] = static_cast<int>(output.values.size());
    }

    result.global_nnz.resize(r_vectors.size());
    MPI_Allreduce(local_nnz.data(),
                  result.global_nnz.data(),
                  static_cast<int>(r_vectors.size()),
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    result.nnz_by_rank.resize(nproc * r_vectors.size());
    MPI_Allgather(local_nnz.data(),
                  static_cast<int>(r_vectors.size()),
                  MPI_INT,
                  result.nnz_by_rank.data(),
                  static_cast<int>(r_vectors.size()),
                  MPI_INT,
                  MPI_COMM_WORLD);
    return result;
}

DistributedSparseMatrix<std::complex<double>> add_td_correction(
    const DistributedSparseMatrix<double>& base_matrix,
    const hamilt::HContainer<std::complex<double>>& td_hR,
    const Parallel_Orbitals& pv,
    const std::vector<ModuleBase::Vector3<int>>& r_vectors,
    const double sparse_thr)
{
    int rank = 0;
    int nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    const int nlocal = PARAM.globalv.nlocal;
    const RowRange row_range = get_row_range(nlocal, rank, nproc);
    // TD velocity-gauge matrices were historically cached at this threshold.
    const auto td_blocks = redistribute_sparse_rows(td_hR, pv, r_vectors, nlocal, 1.0e-10, row_range);

    DistributedSparseMatrix<std::complex<double>> result;
    result.blocks.resize(r_vectors.size());
    std::vector<int> local_nnz(r_vectors.size(), 0);
    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        auto& output = result.blocks[ir];
        output.row_counts.resize(row_range.end - row_range.begin, 0);
        size_t base_pos = 0;
        size_t td_pos = 0;
        for (size_t local_row = 0; local_row < output.row_counts.size(); ++local_row)
        {
            const size_t base_end = base_pos + base_matrix.blocks[ir].row_counts[local_row];
            const size_t td_end = td_pos + td_blocks[ir].row_counts[local_row];
            while (base_pos < base_end || td_pos < td_end)
            {
                int column = 0;
                std::complex<double> value = 0.0;
                if (td_pos == td_end
                    || (base_pos < base_end
                        && base_matrix.blocks[ir].columns[base_pos] < td_blocks[ir].columns[td_pos]))
                {
                    column = base_matrix.blocks[ir].columns[base_pos];
                    value = base_matrix.blocks[ir].values[base_pos++];
                }
                else if (base_pos == base_end
                         || td_blocks[ir].columns[td_pos] < base_matrix.blocks[ir].columns[base_pos])
                {
                    column = td_blocks[ir].columns[td_pos];
                    value = td_blocks[ir].values[td_pos++];
                }
                else
                {
                    column = base_matrix.blocks[ir].columns[base_pos];
                    value = base_matrix.blocks[ir].values[base_pos++] + td_blocks[ir].values[td_pos++];
                }
                if (std::abs(value) > sparse_thr)
                {
                    output.columns.push_back(column);
                    output.values.push_back(value);
                    ++output.row_counts[local_row];
                }
            }
        }
        local_nnz[ir] = static_cast<int>(output.values.size());
    }

    result.global_nnz.resize(r_vectors.size());
    MPI_Allreduce(local_nnz.data(),
                  result.global_nnz.data(),
                  static_cast<int>(r_vectors.size()),
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    result.nnz_by_rank.resize(nproc * r_vectors.size());
    MPI_Allgather(local_nnz.data(),
                  static_cast<int>(r_vectors.size()),
                  MPI_INT,
                  result.nnz_by_rank.data(),
                  static_cast<int>(r_vectors.size()),
                  MPI_INT,
                  MPI_COMM_WORLD);
    return result;
}

template <typename T>
DistributedSparseMatrix<T> prepare_distributed_matrix(
    const hamilt::HContainer<T>& hR,
    const Parallel_Orbitals& pv,
    const std::vector<ModuleBase::Vector3<int>>& r_vectors,
    const double sparse_thr)
{
    int rank = 0;
    int nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    const int nlocal = PARAM.globalv.nlocal;
    const RowRange row_range = get_row_range(nlocal, rank, nproc);
    DistributedSparseMatrix<T> result;
    result.blocks = redistribute_sparse_rows(hR, pv, r_vectors, nlocal, sparse_thr, row_range);

    std::vector<int> local_nnz(r_vectors.size());
    result.global_nnz.resize(r_vectors.size());
    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        local_nnz[ir] = static_cast<int>(result.blocks[ir].values.size());
    }
    MPI_Allreduce(local_nnz.data(),
                  result.global_nnz.data(),
                  static_cast<int>(r_vectors.size()),
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    result.nnz_by_rank.resize(nproc * r_vectors.size());
    MPI_Allgather(local_nnz.data(),
                  static_cast<int>(r_vectors.size()),
                  MPI_INT,
                  result.nnz_by_rank.data(),
                  static_cast<int>(r_vectors.size()),
                  MPI_INT,
                  MPI_COMM_WORLD);
    return result;
}

template <typename T>
void write_distributed_matrix(const std::vector<ModuleBase::Vector3<int>>& r_vectors,
                              const DistributedSparseMatrix<T>& matrix,
                              const std::vector<bool>& output_blocks,
                              const std::string& filename,
                              const std::string& matrix_name,
                              const int step,
                              const bool binary,
                              const bool append)
{
    int rank = 0;
    int nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    const int nlocal = PARAM.globalv.nlocal;
    const RowRange row_range = get_row_range(nlocal, rank, nproc);
    if (binary)
    {
        write_distributed_binary_matrix(r_vectors,
                                        matrix.blocks,
                                        matrix.global_nnz,
                                        matrix.nnz_by_rank,
                                        output_blocks,
                                        row_range,
                                        nlocal,
                                        rank,
                                        nproc,
                                        step,
                                        filename,
                                        append);
    }
    else
    {
        write_distributed_text_matrix(r_vectors,
                                      matrix.blocks,
                                      matrix.global_nnz,
                                      matrix.nnz_by_rank,
                                      output_blocks,
                                      row_range,
                                      nlocal,
                                      rank,
                                      nproc,
                                      step,
                                      matrix_name,
                                      filename,
                                      append);
    }
}
#endif
} // namespace

// if 'binary=true', output binary file.
// The 'sparse_thr' is the accuracy of the sparse matrix.
// If the absolute value of the matrix element is less than or equal to the
// 'sparse_thr', it will be ignored.
void ModuleIO::output_HSR(const UnitCell& ucell,
                          const int& istep,
                          const ModuleBase::matrix& v_eff,
                          const Parallel_Orbitals& pv,
                          LCAO_HS_Arrays& HS_Arrays,
                          const Grid_Driver& grid, // mohan add 2024-04-06
                          const K_Vectors& kv,
                          hamilt::Hamilt<std::complex<double>>* p_ham,
#ifdef __EXX
                          const std::vector<std::map<int, std::map<TAC, RI::Tensor<double>>>>* Hexxd,
                          const std::vector<std::map<int, std::map<TAC, RI::Tensor<std::complex<double>>>>>* Hexxc,
#endif
                          const std::string& SR_filename,
                          const std::string& HR_filename_up,
                          const std::string HR_filename_down,
                          const bool& binary,
                          const double& sparse_thr) {
    ModuleBase::TITLE("ModuleIO", "output_HSR");
    ModuleBase::timer::tick("ModuleIO", "output_HSR");

    const int nspin = PARAM.inp.nspin;

    const auto* td_hR = TD_Velocity::tddft_velocity && TD_Velocity::td_vel_op != nullptr
                            ? TD_Velocity::td_vel_op->get_HR_pointer()
                            : nullptr;
    if (TD_Velocity::tddft_velocity && (nspin == 1 || nspin == 2) && td_hR == nullptr)
    {
        ModuleBase::WARNING_QUIT(
            "ModuleIO::output_HSR",
            "velocity-gauge TDDFT H(R) is unavailable; cannot write a complete Hamiltonian");
    }
    bool use_direct_hcontainer = !TD_Velocity::tddft_velocity || nspin == 1 || nspin == 2;
#ifndef __MPI
    // The DFT+U R-space multiplication is implemented with ScaLAPACK.
    if (PARAM.inp.dft_plus_u == 2)
    {
        use_direct_hcontainer = false;
    }
#endif
    if (use_direct_hcontainer)
    {
        const bool append = PARAM.inp.calculation == "md" && PARAM.inp.out_app_flag && istep;
        const auto make_filename = [istep](const std::string& filename) {
            std::ostringstream path;
            if (PARAM.inp.calculation == "md" && !PARAM.inp.out_app_flag)
            {
                path << PARAM.globalv.global_matrix_dir << istep << "_" << filename;
            }
            else
            {
                path << PARAM.globalv.global_out_dir << filename;
            }
            return path.str();
        };
        const std::string sr_path = make_filename(SR_filename);
        const std::string hr_up_path = make_filename(HR_filename_up);
        const std::string hr_down_path = make_filename(HR_filename_down);
        HS_Arrays.output_R_coor.clear();

        if (nspin == 1 || nspin == 2)
        {
            auto* p_ham_lcao = dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, double>*>(p_ham);
            auto& hR = *(p_ham_lcao->getHR());
            auto& sR = *(p_ham_lcao->getSR());
            if (TD_Velocity::tddft_velocity)
            {
                // The velocity-gauge correction is spin independent. For nspin=2,
                // combine the same td_hR with the spin-down and spin-up base hR.
#ifdef __MPI
                auto r_vectors = merge_R_vectors(collect_global_R(hR), collect_global_R(sR));
                r_vectors = merge_R_vectors(r_vectors, collect_global_R(*td_hR));
                const auto make_td_matrix = [&](const int spin) {
                    if (PARAM.inp.dft_plus_u == 2)
                    {
                        const auto h_dftu
                            = prepare_dftu_distributed_matrix(hR, sR, pv, r_vectors, spin, sparse_thr);
                        return add_td_correction(h_dftu, *td_hR, pv, r_vectors, sparse_thr);
                    }
                    return prepare_td_distributed_matrix(hR, *td_hR, pv, r_vectors, sparse_thr);
                };
                const auto h_down = make_td_matrix(nspin == 2 ? 1 : 0);
                DistributedSparseMatrix<std::complex<double>> h_up;
                if (nspin == 2)
                {
                    if (PARAM.inp.vl_in_h)
                    {
                        p_ham_lcao->updateHR(0);
                        h_up = make_td_matrix(0);
                    }
                    else
                    {
                        h_up.blocks.resize(r_vectors.size());
                        h_up.global_nnz.assign(r_vectors.size(), 0);
                        h_up.nnz_by_rank.assign(h_down.nnz_by_rank.size(), 0);
                    }
                }
                const auto s_matrix = prepare_distributed_matrix(sR, pv, r_vectors, sparse_thr);
                std::vector<bool> output_blocks(r_vectors.size(), false);
                for (size_t ir = 0; ir < r_vectors.size(); ++ir)
                {
                    output_blocks[ir] = h_down.global_nnz[ir] > 0 || s_matrix.global_nnz[ir] > 0
                                        || (nspin == 2 && h_up.global_nnz[ir] > 0);
                    if (output_blocks[ir])
                    {
                        HS_Arrays.output_R_coor.insert(
                            Abfs::Vector3_Order<int>(r_vectors[ir].x, r_vectors[ir].y, r_vectors[ir].z));
                    }
                }
                if (nspin == 2)
                {
                    write_distributed_matrix(r_vectors, h_up, output_blocks, hr_up_path, "H", istep, binary, append);
                    write_distributed_matrix(
                        r_vectors, h_down, output_blocks, hr_down_path, "H", istep, binary, append);
                }
                else
                {
                    write_distributed_matrix(
                        r_vectors, h_down, output_blocks, hr_up_path, "H", istep, binary, append);
                }
                write_distributed_matrix(r_vectors, s_matrix, output_blocks, sr_path, "S", istep, binary, append);
#else
                auto r_vectors = merge_R_vectors(collect_local_R(hR), collect_local_R(sR));
                r_vectors = merge_R_vectors(r_vectors, collect_local_R(*td_hR));
                const auto h_down
                    = make_td_sparse_blocks(hR, *td_hR, pv, r_vectors, PARAM.globalv.nlocal, sparse_thr);
                std::vector<SparseRBlock<std::complex<double>>> h_up;
                if (nspin == 2)
                {
                    if (PARAM.inp.vl_in_h)
                    {
                        p_ham_lcao->updateHR(0);
                        h_up = make_td_sparse_blocks(hR, *td_hR, pv, r_vectors, PARAM.globalv.nlocal, sparse_thr);
                    }
                    else
                    {
                        h_up.resize(r_vectors.size());
                        for (size_t ir = 0; ir < r_vectors.size(); ++ir)
                        {
                            h_up[ir].r = r_vectors[ir];
                        }
                    }
                }
                const auto s_matrix = make_sparse_blocks(sR, pv, r_vectors, PARAM.globalv.nlocal, sparse_thr);
                std::vector<bool> output_blocks(r_vectors.size(), false);
                for (size_t ir = 0; ir < r_vectors.size(); ++ir)
                {
                    output_blocks[ir] = !h_down[ir].values.empty() || !s_matrix[ir].values.empty()
                                        || (nspin == 2 && !h_up[ir].values.empty());
                    if (output_blocks[ir])
                    {
                        HS_Arrays.output_R_coor.insert(
                            Abfs::Vector3_Order<int>(r_vectors[ir].x, r_vectors[ir].y, r_vectors[ir].z));
                    }
                }
                if (nspin == 2)
                {
                    write_serial_matrix(h_up, output_blocks, hr_up_path, "H", istep, binary, append);
                    write_serial_matrix(h_down, output_blocks, hr_down_path, "H", istep, binary, append);
                }
                else
                {
                    write_serial_matrix(h_down, output_blocks, hr_up_path, "H", istep, binary, append);
                }
                write_serial_matrix(s_matrix, output_blocks, sr_path, "S", istep, binary, append);
#endif
                ModuleBase::timer::tick("ModuleIO", "output_HSR");
                return;
            }
#ifdef __MPI
            auto r_vectors = PARAM.inp.dft_plus_u == 2 ? collect_global_R(hR)
                                                       : merge_R_vectors(collect_global_R(hR), collect_global_R(sR));
#else
            auto r_vectors = merge_R_vectors(collect_local_R(hR), collect_local_R(sR));
#endif

#ifdef __MPI
            const int down_spin = nspin == 2 ? 1 : 0;
            const auto h_down = PARAM.inp.dft_plus_u == 2
                                    ? prepare_dftu_distributed_matrix(hR, sR, pv, r_vectors, down_spin, sparse_thr)
                                    : prepare_distributed_matrix(hR, pv, r_vectors, sparse_thr);
            DistributedSparseMatrix<double> h_up;
            if (nspin == 2)
            {
                if (PARAM.inp.vl_in_h)
                {
                    p_ham_lcao->updateHR(0);
                    h_up = PARAM.inp.dft_plus_u == 2
                               ? prepare_dftu_distributed_matrix(hR, sR, pv, r_vectors, 0, sparse_thr)
                               : prepare_distributed_matrix(hR, pv, r_vectors, sparse_thr);
                }
                else
                {
                    h_up.blocks.resize(r_vectors.size());
                    h_up.global_nnz.assign(r_vectors.size(), 0);
                    h_up.nnz_by_rank.assign(h_down.nnz_by_rank.size(), 0);
                }
            }
            const auto s_matrix = prepare_distributed_matrix(sR, pv, r_vectors, sparse_thr);
            std::vector<bool> output_blocks(r_vectors.size(), false);
            for (size_t ir = 0; ir < r_vectors.size(); ++ir)
            {
                output_blocks[ir] = h_down.global_nnz[ir] > 0 || s_matrix.global_nnz[ir] > 0
                                    || (nspin == 2 && h_up.global_nnz[ir] > 0);
                if (output_blocks[ir])
                {
                    HS_Arrays.output_R_coor.insert(
                        Abfs::Vector3_Order<int>(r_vectors[ir].x, r_vectors[ir].y, r_vectors[ir].z));
                }
            }
            if (nspin == 2)
            {
                write_distributed_matrix(r_vectors, h_up, output_blocks, hr_up_path, "H", istep, binary, append);
                write_distributed_matrix(r_vectors, h_down, output_blocks, hr_down_path, "H", istep, binary, append);
            }
            else
            {
                write_distributed_matrix(r_vectors, h_down, output_blocks, hr_up_path, "H", istep, binary, append);
            }
            write_distributed_matrix(r_vectors, s_matrix, output_blocks, sr_path, "S", istep, binary, append);
#else
            const auto h_down = make_sparse_blocks(hR, pv, r_vectors, PARAM.globalv.nlocal, sparse_thr);
            std::vector<SparseRBlock<double>> h_up;
            if (nspin == 2)
            {
                if (PARAM.inp.vl_in_h)
                {
                    p_ham_lcao->updateHR(0);
                    h_up = make_sparse_blocks(hR, pv, r_vectors, PARAM.globalv.nlocal, sparse_thr);
                }
                else
                {
                    h_up.resize(r_vectors.size());
                    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
                    {
                        h_up[ir].r = r_vectors[ir];
                    }
                }
            }
            const auto s_matrix = make_sparse_blocks(sR, pv, r_vectors, PARAM.globalv.nlocal, sparse_thr);
            std::vector<bool> output_blocks(r_vectors.size(), false);
            for (size_t ir = 0; ir < r_vectors.size(); ++ir)
            {
                output_blocks[ir] = !h_down[ir].values.empty() || !s_matrix[ir].values.empty()
                                    || (nspin == 2 && !h_up[ir].values.empty());
                if (output_blocks[ir])
                {
                    HS_Arrays.output_R_coor.insert(
                        Abfs::Vector3_Order<int>(r_vectors[ir].x, r_vectors[ir].y, r_vectors[ir].z));
                }
            }
            if (nspin == 2)
            {
                write_serial_matrix(h_up, output_blocks, hr_up_path, "H", istep, binary, append);
                write_serial_matrix(h_down, output_blocks, hr_down_path, "H", istep, binary, append);
            }
            else
            {
                write_serial_matrix(h_down, output_blocks, hr_up_path, "H", istep, binary, append);
            }
            write_serial_matrix(s_matrix, output_blocks, sr_path, "S", istep, binary, append);
#endif
            ModuleBase::timer::tick("ModuleIO", "output_HSR");
            return;
        }
        else if (nspin == 4)
        {
            auto* p_ham_lcao
                = dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, std::complex<double>>*>(p_ham);
            auto& hR = *(p_ham_lcao->getHR());
            auto& sR = *(p_ham_lcao->getSR());
#ifdef __MPI
            auto r_vectors = PARAM.inp.dft_plus_u == 2 ? collect_global_R(hR)
                                                       : merge_R_vectors(collect_global_R(hR), collect_global_R(sR));
#else
            auto r_vectors = merge_R_vectors(collect_local_R(hR), collect_local_R(sR));
#endif
#ifdef __MPI
            const auto h_matrix = PARAM.inp.dft_plus_u == 2
                                      ? prepare_dftu_distributed_matrix(hR, sR, pv, r_vectors, 0, sparse_thr)
                                      : prepare_distributed_matrix(hR, pv, r_vectors, sparse_thr);
            const auto s_matrix = prepare_distributed_matrix(sR, pv, r_vectors, sparse_thr);
            std::vector<bool> output_blocks(r_vectors.size(), false);
            for (size_t ir = 0; ir < r_vectors.size(); ++ir)
            {
                output_blocks[ir] = h_matrix.global_nnz[ir] > 0 || s_matrix.global_nnz[ir] > 0;
                if (output_blocks[ir])
                {
                    HS_Arrays.output_R_coor.insert(
                        Abfs::Vector3_Order<int>(r_vectors[ir].x, r_vectors[ir].y, r_vectors[ir].z));
                }
            }
            write_distributed_matrix(r_vectors, h_matrix, output_blocks, hr_up_path, "H", istep, binary, append);
            write_distributed_matrix(r_vectors, s_matrix, output_blocks, sr_path, "S", istep, binary, append);
#else
            const auto h_matrix = make_sparse_blocks(hR, pv, r_vectors, PARAM.globalv.nlocal, sparse_thr);
            const auto s_matrix = make_sparse_blocks(sR, pv, r_vectors, PARAM.globalv.nlocal, sparse_thr);
            std::vector<bool> output_blocks(r_vectors.size(), false);
            for (size_t ir = 0; ir < r_vectors.size(); ++ir)
            {
                output_blocks[ir] = !h_matrix[ir].values.empty() || !s_matrix[ir].values.empty();
                if (output_blocks[ir])
                {
                    HS_Arrays.output_R_coor.insert(
                        Abfs::Vector3_Order<int>(r_vectors[ir].x, r_vectors[ir].y, r_vectors[ir].z));
                }
            }
            write_serial_matrix(h_matrix, output_blocks, hr_up_path, "H", istep, binary, append);
            write_serial_matrix(s_matrix, output_blocks, sr_path, "S", istep, binary, append);
#endif
            ModuleBase::timer::tick("ModuleIO", "output_HSR");
            return;
        }
    }

    if (nspin == 1 || nspin == 4) {
        const int spin_now = 0;
        // jingan add 2021-6-4, modify 2021-12-2
        sparse_format::cal_HSR(ucell,pv, HS_Arrays, grid, spin_now, sparse_thr, kv.nmp, p_ham
#ifdef __EXX
            , Hexxd, Hexxc
#endif
        );
    }
    else if (nspin == 2) {
        int spin_now = 1;

        // save HR of spin down first (the current spin always be down)
        sparse_format::cal_HSR(ucell,pv, HS_Arrays, grid, spin_now, sparse_thr, kv.nmp, p_ham
#ifdef __EXX
            , Hexxd, Hexxc
#endif
        );

        // cal HR of the spin up
        if (PARAM.inp.vl_in_h) {
            const int ik = 0;
            p_ham->refresh();
            p_ham->updateHk(ik);
            spin_now = 0;
        }

        sparse_format::cal_HSR(ucell,pv, HS_Arrays, grid, spin_now, sparse_thr, kv.nmp, p_ham
#ifdef __EXX
            , Hexxd, Hexxc
#endif
        );
    }

    ModuleIO::save_HSR_sparse(istep,
                              pv,
                              HS_Arrays,
                              sparse_thr,
                              binary,
                              SR_filename,
                              HR_filename_up,
                              HR_filename_down);

    sparse_format::destroy_HS_R_sparse(HS_Arrays);

    ModuleBase::timer::tick("ModuleIO", "output_HSR");
    return;
}

void ModuleIO::output_dHR(const int& istep,
                          const ModuleBase::matrix& v_eff,
                          Gint_k& gint_k, // mohan add 2024-04-01
                          const UnitCell& ucell,
                          const Parallel_Orbitals& pv,
                          LCAO_HS_Arrays& HS_Arrays,
                          const Grid_Driver& grid, // mohan add 2024-04-06
                          const TwoCenterBundle& two_center_bundle,
                          const LCAO_Orbitals& orb,
                          const K_Vectors& kv,
                          const bool& binary,
                          const double& sparse_thr)
{
    ModuleBase::TITLE("ModuleIO", "output_dHR");
    ModuleBase::timer::tick("ModuleIO", "output_dHR");

    gint_k.allocate_pvdpR();

    const int nspin = PARAM.inp.nspin;

    if (nspin == 1 || nspin == 4) {
        // mohan add 2024-04-01
        const int cspin = 0;

        sparse_format::cal_dH(ucell,
                              pv,
                              HS_Arrays,
                              grid,
                              two_center_bundle,
                              orb,
                              cspin,
                              sparse_thr,
                              gint_k);
    } else if (nspin == 2) {
        for (int cspin = 0; cspin < 2; cspin++) {
            // note: some MPI process will not have grids when MPI cores are too
            // many, v_eff in these processes are empty
            const double* vr_eff1
                = v_eff.nc * v_eff.nr > 0 ? &(v_eff(cspin, 0)) : nullptr;

            if (!PARAM.globalv.gamma_only_local) {
                if (PARAM.inp.vl_in_h) {
                    Gint_inout inout(vr_eff1,
                                     cspin,
                                     Gint_Tools::job_type::dvlocal);
                    gint_k.cal_gint(&inout);
                }
            }

            sparse_format::cal_dH(ucell,
                                  pv,
                                  HS_Arrays,
                                  grid,
                                  two_center_bundle,
                                  orb,
                                  cspin,
                                  sparse_thr,
                                  gint_k);
        }
    }
    // mohan update 2024-04-01
    ModuleIO::save_dH_sparse(istep, pv, HS_Arrays, sparse_thr, binary);

    sparse_format::destroy_dH_R_sparse(HS_Arrays);

    gint_k.destroy_pvdpR();

    ModuleBase::timer::tick("ModuleIO", "output_dHR");
    return;
}

void ModuleIO::output_SR(Parallel_Orbitals& pv,
                         const Grid_Driver& /*grid*/,
                         hamilt::Hamilt<std::complex<double>>* p_ham,
                         const std::string& SR_filename,
                         const bool& binary,
                         const double& sparse_thr)
{
    ModuleBase::TITLE("ModuleIO", "output_SR");
    ModuleBase::timer::tick("ModuleIO", "output_SR");

    if (PARAM.inp.nspin != 4)
    {
        auto* p_ham_lcao = dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, double>*>(p_ham);
#ifdef __MPI
        const auto r_vectors = collect_global_R(*(p_ham_lcao->getSR()));
        const auto matrix = prepare_distributed_matrix(*(p_ham_lcao->getSR()), pv, r_vectors, sparse_thr);
        std::vector<bool> output_blocks(r_vectors.size(), false);
        for (size_t ir = 0; ir < r_vectors.size(); ++ir)
        {
            output_blocks[ir] = matrix.global_nnz[ir] > 0;
        }
        write_distributed_matrix(r_vectors, matrix, output_blocks, SR_filename, "S", 0, binary, false);
#else
        const auto r_vectors = collect_local_R(*(p_ham_lcao->getSR()));
        const auto blocks
            = make_sparse_blocks(*(p_ham_lcao->getSR()), pv, r_vectors, PARAM.globalv.nlocal, sparse_thr);
        std::vector<bool> output_blocks(r_vectors.size(), false);
        for (size_t ir = 0; ir < r_vectors.size(); ++ir)
        {
            output_blocks[ir] = !blocks[ir].values.empty();
        }
        write_serial_matrix(blocks, output_blocks, SR_filename, "S", 0, binary, false);
#endif
    }
    else
    {
        auto* p_ham_lcao
            = dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, std::complex<double>>*>(p_ham);
#ifdef __MPI
        const auto r_vectors = collect_global_R(*(p_ham_lcao->getSR()));
        const auto matrix = prepare_distributed_matrix(*(p_ham_lcao->getSR()), pv, r_vectors, sparse_thr);
        std::vector<bool> output_blocks(r_vectors.size(), false);
        for (size_t ir = 0; ir < r_vectors.size(); ++ir)
        {
            output_blocks[ir] = matrix.global_nnz[ir] > 0;
        }
        write_distributed_matrix(r_vectors, matrix, output_blocks, SR_filename, "S", 0, binary, false);
#else
        const auto r_vectors = collect_local_R(*(p_ham_lcao->getSR()));
        const auto blocks
            = make_sparse_blocks(*(p_ham_lcao->getSR()), pv, r_vectors, PARAM.globalv.nlocal, sparse_thr);
        std::vector<bool> output_blocks(r_vectors.size(), false);
        for (size_t ir = 0; ir < r_vectors.size(); ++ir)
        {
            output_blocks[ir] = !blocks[ir].values.empty();
        }
        write_serial_matrix(blocks, output_blocks, SR_filename, "S", 0, binary, false);
#endif
    }

    ModuleBase::timer::tick("ModuleIO", "output_SR");
    return;
}

void ModuleIO::output_TR(const int istep,
                         const UnitCell& ucell,
                         const Parallel_Orbitals& pv,
                         LCAO_HS_Arrays& HS_Arrays,
                         const Grid_Driver& grid,
                         const TwoCenterBundle& two_center_bundle,
                         const LCAO_Orbitals& orb,
                         const std::string& TR_filename,
                         const bool& binary,
                         const double& sparse_thr)
{
    ModuleBase::TITLE("ModuleIO", "output_TR");
    ModuleBase::timer::tick("ModuleIO", "output_TR");

    std::stringstream sst;
    if (PARAM.inp.calculation == "md" && !PARAM.inp.out_app_flag) {
        sst << PARAM.globalv.global_matrix_dir << istep << "_" << TR_filename;
    } else {
        sst << PARAM.globalv.global_out_dir << TR_filename;
    }

    sparse_format::cal_TR(ucell,
                          pv,
                          HS_Arrays,
                          grid,
                          two_center_bundle,
                          orb,
                          sparse_thr);

    ModuleIO::save_sparse(HS_Arrays.TR_sparse,
                          HS_Arrays.all_R_coor,
                          sparse_thr,
                          binary,
                          sst.str().c_str(),
                          pv,
                          "T",
                          istep);

    sparse_format::destroy_T_R_sparse(HS_Arrays);

    ModuleBase::timer::tick("ModuleIO", "output_TR");
    return;
}
