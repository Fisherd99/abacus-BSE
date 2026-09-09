#include "write_HS_R.h"

#include "module_parameter/parameter.h"
#include "module_base/timer.h"
#include "module_hamilt_lcao/hamilt_lcaodft/LCAO_HS_arrays.hpp"
#include "module_hamilt_lcao/hamilt_lcaodft/spar_dh.h"
#include "module_hamilt_lcao/hamilt_lcaodft/spar_hsr.h"
#include "module_hamilt_lcao/hamilt_lcaodft/spar_st.h"
#include "module_hamilt_lcao/module_hcontainer/transfer.h"
#include "write_HS_sparse.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <iomanip>
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

template <typename T>
std::vector<SparseRBlock<T>> make_sparse_blocks(const hamilt::HContainer<T>& hR,
                                                 const int nlocal,
                                                 const double sparse_thr)
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
    std::sort(r_vectors.begin(), r_vectors.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.x != rhs.x)
        {
            return lhs.x < rhs.x;
        }
        if (lhs.y != rhs.y)
        {
            return lhs.y < rhs.y;
        }
        return lhs.z < rhs.z;
    });

    std::vector<SparseRBlock<T>> blocks;
    blocks.reserve(r_vectors.size());
    for (const auto& r : r_vectors)
    {
        hR.fix_R(r.x, r.y, r.z);
        std::vector<std::vector<std::pair<int, T>>> rows(nlocal);
        for (int iap = 0; iap < hR.size_atom_pairs(); ++iap)
        {
            const auto matrix_info = hR.get_atom_pair(iap).get_matrix_values();
            const auto& indexes = std::get<0>(matrix_info);
            const T* data = std::get<1>(matrix_info);
            for (int irow = 0; irow < indexes[1]; ++irow)
            {
                auto& row = rows[indexes[0] + irow];
                for (int icol = 0; icol < indexes[3]; ++icol)
                {
                    const T value = data[irow * indexes[3] + icol];
                    if (std::abs(value) > sparse_thr)
                    {
                        row.emplace_back(indexes[2] + icol, value);
                    }
                }
            }
        }
        hR.unfix_R();

        SparseRBlock<T> block;
        block.r = r;
        block.row_ptr.reserve(nlocal + 1);
        block.row_ptr.push_back(0);
        for (auto& row : rows)
        {
            std::sort(row.begin(), row.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.first < rhs.first;
            });
            for (const auto& entry : row)
            {
                block.columns.push_back(entry.first);
                block.values.push_back(entry.second);
            }
            block.row_ptr.push_back(static_cast<int>(block.values.size()));
        }
        if (!block.values.empty())
        {
            blocks.push_back(std::move(block));
        }
    }
    return blocks;
}

template <typename T>
void write_serial_SR(const hamilt::HContainer<T>& hR,
                     const std::string& filename,
                     const bool binary,
                     const double sparse_thr)
{
    const int nlocal = PARAM.globalv.nlocal;
    const auto blocks = make_sparse_blocks(hR, nlocal, sparse_thr);
    std::ofstream ofs(filename, binary ? std::ios::binary : std::ios::out);
    const int step = 0;
    const int block_count = static_cast<int>(blocks.size());
    if (binary)
    {
        ofs.write(reinterpret_cast<const char*>(&step), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&nlocal), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&block_count), sizeof(int));
    }
    else
    {
        ofs << "STEP: 0\n";
        ofs << "Matrix Dimension of S(R): " << nlocal << "\n";
        ofs << "Matrix number of S(R): " << block_count << "\n";
    }

    for (const auto& block : blocks)
    {
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
            ofs.write(reinterpret_cast<const char*>(block.values.data()), nnz * sizeof(T));
            ofs.write(reinterpret_cast<const char*>(block.columns.data()), nnz * sizeof(int));
            ofs.write(reinterpret_cast<const char*>(block.row_ptr.data()),
                      (nlocal + 1) * sizeof(int));
        }
        else
        {
            ofs << rx << " " << ry << " " << rz << " " << nnz << "\n";
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
    std::sort(received.begin(), received.end(), [](const auto& lhs, const auto& rhs) {
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
                         const FileFragments& fragments)
{
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD,
                  const_cast<char*>(filename.c_str()),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL,
                  &file);
    MPI_File_set_size(file, file_size);

    MPI_Datatype filetype = MPI_DATATYPE_NULL;
    if (fragments.lengths.empty())
    {
        MPI_File_set_view(file, 0, MPI_BYTE, MPI_BYTE, const_cast<char*>("native"), MPI_INFO_NULL);
    }
    else
    {
        MPI_Type_create_hindexed(static_cast<int>(fragments.lengths.size()),
                                 fragments.lengths.data(),
                                 fragments.offsets.data(),
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
void write_distributed_binary_SR(const std::vector<ModuleBase::Vector3<int>>& r_vectors,
                                 const std::vector<DistributedRBlock<T>>& blocks,
                                 const std::vector<int>& global_nnz,
                                 const std::vector<int>& nnz_by_rank,
                                 const RowRange row_range,
                                 const int nlocal,
                                 const int rank,
                                 const int nproc,
                                 const std::string& filename)
{
    const int block_count = static_cast<int>(std::count_if(global_nnz.begin(), global_nnz.end(), [](const int n) {
        return n > 0;
    }));
    FileFragments fragments;
    MPI_Offset offset = 0;
    const int file_header[3] = {0, nlocal, block_count};
    if (rank == 0)
    {
        fragments.append(offset, file_header, sizeof(file_header));
    }
    offset += sizeof(file_header);

    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        if (global_nnz[ir] == 0)
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
    write_MPI_fragments(filename, offset, fragments);
}

template <typename T>
void write_distributed_text_SR(const std::vector<ModuleBase::Vector3<int>>& r_vectors,
                               const std::vector<DistributedRBlock<T>>& blocks,
                               const std::vector<int>& global_nnz,
                               const std::vector<int>& nnz_by_rank,
                               const RowRange row_range,
                               const int nlocal,
                               const int rank,
                               const int nproc,
                               const std::string& filename)
{
    std::vector<std::string> values_text(r_vectors.size());
    std::vector<std::string> columns_text(r_vectors.size());
    std::vector<std::string> row_ptr_text(r_vectors.size());
    std::vector<unsigned long long> local_lengths(3 * r_vectors.size(), 0);
    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
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

    const int block_count = static_cast<int>(std::count_if(global_nnz.begin(), global_nnz.end(), [](const int n) {
        return n > 0;
    }));
    std::ostringstream header_stream;
    header_stream << "STEP: 0\n";
    header_stream << "Matrix Dimension of S(R): " << nlocal << "\n";
    header_stream << "Matrix number of S(R): " << block_count << "\n";
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
        if (global_nnz[ir] == 0)
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
    write_MPI_fragments(filename, offset, fragments);
}

template <typename T>
void write_distributed_SR(const hamilt::HContainer<T>& hR,
                          const Parallel_Orbitals& pv,
                          const std::string& filename,
                          const bool binary,
                          const double sparse_thr)
{
    int rank = 0;
    int nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    if (nproc == 1)
    {
        write_serial_SR(hR, filename, binary, sparse_thr);
        return;
    }
    const int nlocal = PARAM.globalv.nlocal;
    const RowRange row_range = get_row_range(nlocal, rank, nproc);
    const auto r_vectors = collect_global_R(hR);
    const auto blocks = redistribute_sparse_rows(hR, pv, r_vectors, nlocal, sparse_thr, row_range);

    std::vector<int> local_nnz(r_vectors.size()), global_nnz(r_vectors.size());
    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        local_nnz[ir] = static_cast<int>(blocks[ir].values.size());
    }
    MPI_Allreduce(local_nnz.data(),
                  global_nnz.data(),
                  static_cast<int>(r_vectors.size()),
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    std::vector<int> nnz_by_rank(nproc * r_vectors.size());
    MPI_Allgather(local_nnz.data(),
                  static_cast<int>(r_vectors.size()),
                  MPI_INT,
                  nnz_by_rank.data(),
                  static_cast<int>(r_vectors.size()),
                  MPI_INT,
                  MPI_COMM_WORLD);

    if (binary)
    {
        write_distributed_binary_SR(r_vectors,
                                    blocks,
                                    global_nnz,
                                    nnz_by_rank,
                                    row_range,
                                    nlocal,
                                    rank,
                                    nproc,
                                    filename);
    }
    else
    {
        write_distributed_text_SR(r_vectors,
                                  blocks,
                                  global_nnz,
                                  nnz_by_rank,
                                  row_range,
                                  nlocal,
                                  rank,
                                  nproc,
                                  filename);
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
        write_distributed_SR(*(p_ham_lcao->getSR()), pv, SR_filename, binary, sparse_thr);
#else
        write_serial_SR(*(p_ham_lcao->getSR()), SR_filename, binary, sparse_thr);
#endif
    }
    else
    {
        auto* p_ham_lcao
            = dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, std::complex<double>>*>(p_ham);
#ifdef __MPI
        write_distributed_SR(*(p_ham_lcao->getSR()), pv, SR_filename, binary, sparse_thr);
#else
        write_serial_SR(*(p_ham_lcao->getSR()), SR_filename, binary, sparse_thr);
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
