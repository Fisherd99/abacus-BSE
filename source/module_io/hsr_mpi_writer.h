#ifndef MODULE_IO_HSR_MPI_WRITER_H
#define MODULE_IO_HSR_MPI_WRITER_H

#ifdef __MPI

#include "hsr_mpi_distribution.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstring>
#include <deque>
#include <mpi.h>
#include <sstream>
#include <string>
#include <vector>

namespace ModuleIO
{
namespace detail
{
struct FileFragment
{
    const char* data;
    size_t buffer_offset;
    size_t byte_count;
    MPI_Aint file_offset;
};

struct FileFragments
{
    std::vector<FileFragment> fragments;
    std::deque<std::vector<char>> owned_buffers;
    size_t byte_count = 0;

    FileFragments() = default;
    FileFragments(const FileFragments&) = delete;
    FileFragments& operator=(const FileFragments&) = delete;

    // Keep small or temporary data alive until the collective write finishes.
    void append(const MPI_Offset offset, const void* source, const size_t size)
    {
        if (size == 0)
        {
            return;
        }
        owned_buffers.emplace_back(size);
        std::memcpy(owned_buffers.back().data(), source, size);
        append_reference(offset, owned_buffers.back().data(), size);
    }

    void append(const MPI_Offset offset, const std::string& source)
    {
        append(offset, source.data(), source.size());
    }

    // The referenced storage must remain valid until write_MPI_fragments returns.
    void append_reference(const MPI_Offset offset, const void* source, const size_t size)
    {
        if (size == 0)
        {
            return;
        }
        fragments.push_back({static_cast<const char*>(source), byte_count, size, static_cast<MPI_Aint>(offset)});
        byte_count += size;
    }
};

struct ChunkFileView
{
    std::vector<int> block_lengths;
    std::vector<MPI_Aint> file_offsets;
    std::vector<MPI_Aint> memory_addresses;
};

inline ChunkFileView make_chunk_file_view(const std::vector<FileFragment>& fragments,
                                          const size_t chunk_begin,
                                          const size_t chunk_size,
                                          const MPI_Offset base_offset)
{
    ChunkFileView view;
    const size_t chunk_end = chunk_begin + chunk_size;
    for (const auto& fragment: fragments)
    {
        const size_t fragment_end = fragment.buffer_offset + fragment.byte_count;
        const size_t overlap_begin = std::max(fragment.buffer_offset, chunk_begin);
        const size_t overlap_end = std::min(fragment_end, chunk_end);
        if (overlap_begin >= overlap_end)
        {
            continue;
        }
        view.block_lengths.push_back(static_cast<int>(overlap_end - overlap_begin));
        view.file_offsets.push_back(fragment.file_offset + base_offset
                                    + static_cast<MPI_Aint>(overlap_begin - fragment.buffer_offset));
        MPI_Aint address = 0;
        MPI_Get_address(const_cast<char*>(fragment.data) + overlap_begin - fragment.buffer_offset, &address);
        view.memory_addresses.push_back(address);
    }
    return view;
}

inline void write_MPI_fragments(const std::string& filename,
                                const MPI_Offset file_size,
                                const FileFragments& fragments,
                                const bool append,
                                MPI_Comm communicator,
                                const size_t max_bytes_per_round = 0)
{
    MPI_File file;
    MPI_File_open(communicator,
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

    const auto local_size = static_cast<unsigned long long>(fragments.byte_count);
    unsigned long long max_size = 0;
    MPI_Allreduce(&local_size, &max_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, communicator);
    const size_t chunk_limit = max_bytes_per_round == 0 ? static_cast<size_t>(INT_MAX)
                                                        : std::min(max_bytes_per_round, static_cast<size_t>(INT_MAX));
    const unsigned long long round_count = max_size == 0 ? 0 : (max_size - 1) / chunk_limit + 1;
    for (unsigned long long round = 0; round < round_count; ++round)
    {
        const size_t chunk_begin = static_cast<size_t>(round) * chunk_limit;
        const size_t remaining = chunk_begin < fragments.byte_count ? fragments.byte_count - chunk_begin : 0;
        const size_t chunk_size = std::min(chunk_limit, remaining);
        const auto view = make_chunk_file_view(fragments.fragments, chunk_begin, chunk_size, base_offset);
        MPI_Datatype filetype = MPI_DATATYPE_NULL;
        MPI_Datatype memorytype = MPI_DATATYPE_NULL;
        if (view.block_lengths.empty())
        {
            MPI_File_set_view(file, 0, MPI_BYTE, MPI_BYTE, const_cast<char*>("native"), MPI_INFO_NULL);
        }
        else
        {
            MPI_Type_create_hindexed(static_cast<int>(view.block_lengths.size()),
                                     view.block_lengths.data(),
                                     view.file_offsets.data(),
                                     MPI_BYTE,
                                     &filetype);
            MPI_Type_commit(&filetype);
            MPI_Type_create_hindexed(static_cast<int>(view.block_lengths.size()),
                                     view.block_lengths.data(),
                                     view.memory_addresses.data(),
                                     MPI_BYTE,
                                     &memorytype);
            MPI_Type_commit(&memorytype);
            MPI_File_set_view(file, 0, MPI_BYTE, filetype, const_cast<char*>("native"), MPI_INFO_NULL);
        }
        if (memorytype == MPI_DATATYPE_NULL)
        {
            MPI_File_write_all(file, nullptr, 0, MPI_BYTE, MPI_STATUS_IGNORE);
        }
        else
        {
            MPI_File_write_all(file, MPI_BOTTOM, 1, memorytype, MPI_STATUS_IGNORE);
            MPI_Type_free(&memorytype);
        }
        if (filetype != MPI_DATATYPE_NULL)
        {
            MPI_Type_free(&filetype);
        }
    }
    MPI_File_close(&file);
}

template <typename T>
void write_distributed_matrix(const std::vector<ModuleBase::Vector3<int>>& r_vectors,
                              const DistributedSparseMatrix<T>& matrix,
                              const std::vector<bool>& output_blocks,
                              const std::string& filename,
                              const std::string& matrix_name,
                              const int step,
                              const bool binary,
                              const bool append,
                              const DistributionContext& context)
{
    const int block_count = static_cast<int>(std::count(output_blocks.begin(), output_blocks.end(), true));
    FileFragments fragments;
    MPI_Offset offset = 0;
    if (binary)
    {
        const int file_header[3] = {step, context.nlocal, block_count};
        if (context.rank == 0)
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
            const int block_header[4] = {r_vectors[ir].x, r_vectors[ir].y, r_vectors[ir].z, matrix.global_nnz[ir]};
            if (context.rank == 0)
            {
                fragments.append(offset, block_header, sizeof(block_header));
            }
            offset += sizeof(block_header);

            int rank_nnz_offset = 0;
            for (int ip = 0; ip < context.rank; ++ip)
            {
                rank_nnz_offset += matrix.nnz_by_rank[ip * r_vectors.size() + ir];
            }
            const auto& block = matrix.blocks[ir];
            fragments.append_reference(offset + static_cast<MPI_Offset>(rank_nnz_offset) * sizeof(T),
                                       block.values.data(),
                                       block.values.size() * sizeof(T));
            offset += static_cast<MPI_Offset>(matrix.global_nnz[ir]) * sizeof(T);
            fragments.append_reference(offset + static_cast<MPI_Offset>(rank_nnz_offset) * sizeof(int),
                                       block.columns.data(),
                                       block.columns.size() * sizeof(int));
            offset += static_cast<MPI_Offset>(matrix.global_nnz[ir]) * sizeof(int);

            if (matrix.global_nnz[ir] > 0)
            {
                std::vector<int> row_ptr;
                row_ptr.reserve(block.row_counts.size() + 1);
                int row_offset = rank_nnz_offset;
                for (const int count: block.row_counts)
                {
                    row_ptr.push_back(row_offset);
                    row_offset += count;
                }
                if (context.row_range.end == context.nlocal && context.row_range.begin < context.row_range.end)
                {
                    row_ptr.push_back(matrix.global_nnz[ir]);
                }
                fragments.append(offset + static_cast<MPI_Offset>(context.row_range.begin) * sizeof(int),
                                 row_ptr.data(),
                                 row_ptr.size() * sizeof(int));
                offset += static_cast<MPI_Offset>(context.nlocal + 1) * sizeof(int);
            }
        }
        write_MPI_fragments(filename, offset, fragments, append, context.communicator);
    }
    else
    {
        std::vector<std::string> values_text(r_vectors.size());
        std::vector<std::string> columns_text(r_vectors.size());
        std::vector<std::string> row_ptr_text(r_vectors.size());
        std::vector<unsigned long long> local_lengths(3 * r_vectors.size(), 0);
        for (size_t ir = 0; ir < r_vectors.size(); ++ir)
        {
            if (!output_blocks[ir] || matrix.global_nnz[ir] == 0)
            {
                continue;
            }
            std::ostringstream values_stream;
            for (const auto& value: matrix.blocks[ir].values)
            {
                write_sparse_value(values_stream, value);
            }
            values_text[ir] = values_stream.str();
            std::ostringstream columns_stream;
            for (const int column: matrix.blocks[ir].columns)
            {
                columns_stream << " " << column;
            }
            columns_text[ir] = columns_stream.str();

            int row_offset = 0;
            for (int ip = 0; ip < context.rank; ++ip)
            {
                row_offset += matrix.nnz_by_rank[ip * r_vectors.size() + ir];
            }
            std::ostringstream row_ptr_stream;
            for (const int count: matrix.blocks[ir].row_counts)
            {
                row_ptr_stream << " " << row_offset;
                row_offset += count;
            }
            if (context.row_range.end == context.nlocal && context.row_range.begin < context.row_range.end)
            {
                row_ptr_stream << " " << matrix.global_nnz[ir];
            }
            row_ptr_text[ir] = row_ptr_stream.str();
            local_lengths[3 * ir] = values_text[ir].size();
            local_lengths[3 * ir + 1] = columns_text[ir].size();
            local_lengths[3 * ir + 2] = row_ptr_text[ir].size();
        }

        std::vector<unsigned long long> lengths_by_rank(context.nproc * local_lengths.size());
        MPI_Allgather(local_lengths.data(),
                      static_cast<int>(local_lengths.size()),
                      MPI_UNSIGNED_LONG_LONG,
                      lengths_by_rank.data(),
                      static_cast<int>(local_lengths.size()),
                      MPI_UNSIGNED_LONG_LONG,
                      context.communicator);

        std::ostringstream header_stream;
        header_stream << "STEP: " << step << "\n";
        header_stream << "Matrix Dimension of " << matrix_name << "(R): " << context.nlocal << "\n";
        header_stream << "Matrix number of " << matrix_name << "(R): " << block_count << "\n";
        const std::string header = header_stream.str();
        if (context.rank == 0)
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
                                << matrix.global_nnz[ir] << "\n";
            const std::string block_header = block_header_stream.str();
            if (context.rank == 0)
            {
                fragments.append(offset, block_header);
            }
            offset += block_header.size();
            if (matrix.global_nnz[ir] == 0)
            {
                continue;
            }
            for (int section = 0; section < 3; ++section)
            {
                unsigned long long rank_offset = 0;
                unsigned long long total_length = 0;
                for (int ip = 0; ip < context.nproc; ++ip)
                {
                    const auto length = lengths_by_rank[ip * local_lengths.size() + 3 * ir + section];
                    if (ip < context.rank)
                    {
                        rank_offset += length;
                    }
                    total_length += length;
                }
                const std::string* text = section == 0   ? &values_text[ir]
                                          : section == 1 ? &columns_text[ir]
                                                         : &row_ptr_text[ir];
                fragments.append_reference(offset + rank_offset, text->data(), text->size());
                offset += total_length;
                if (context.rank == 0)
                {
                    fragments.append(offset, "\n", 1);
                }
                ++offset;
            }
        }
        write_MPI_fragments(filename, offset, fragments, append, context.communicator);
    }
}
} // namespace detail
} // namespace ModuleIO

#endif // __MPI
#endif // MODULE_IO_HSR_MPI_WRITER_H
