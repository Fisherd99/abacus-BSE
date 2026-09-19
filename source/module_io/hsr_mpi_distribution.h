#ifndef MODULE_IO_HSR_MPI_DISTRIBUTION_H
#define MODULE_IO_HSR_MPI_DISTRIBUTION_H

#ifdef __MPI

#include "hsr_sparse_utils.h"
#include "module_hamilt_lcao/module_hcontainer/transfer.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

namespace ModuleIO
{
namespace detail
{
template <typename T>
struct DistributedEntry
{
    int ir;
    int row;
    int column;
    T value;
};

// Exchange entries in synchronized rounds while keeping every MPI_Alltoallv
// count and displacement representable as int. The explicit limit makes the
// multi-round path testable without allocating INT_MAX elements.
template <typename T>
std::vector<DistributedEntry<T>> exchange_entries_chunked(const std::vector<std::vector<DistributedEntry<T>>>& outgoing,
                                                          MPI_Comm communicator,
                                                          size_t max_entries_per_peer = 0)
{
    int nproc = 0;
    MPI_Comm_size(communicator, &nproc);
    constexpr size_t metadata_width = 3;
    const size_t safe_entries_per_peer = static_cast<size_t>(INT_MAX) / (metadata_width * static_cast<size_t>(nproc));
    const size_t entries_per_peer
        = max_entries_per_peer == 0 ? safe_entries_per_peer : std::min(max_entries_per_peer, safe_entries_per_peer);

    unsigned long long local_round_count = 0;
    for (const auto& entries: outgoing)
    {
        if (!entries.empty())
        {
            const auto rounds = static_cast<unsigned long long>((entries.size() - 1) / entries_per_peer + 1);
            local_round_count = std::max(local_round_count, rounds);
        }
    }
    unsigned long long round_count = 0;
    MPI_Allreduce(&local_round_count, &round_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, communicator);

    std::vector<DistributedEntry<T>> received;
    for (unsigned long long round = 0; round < round_count; ++round)
    {
        std::vector<int> send_counts(nproc), recv_counts(nproc);
        std::vector<int> send_displs(nproc, 0), recv_displs(nproc, 0);
        const size_t begin = static_cast<size_t>(round) * entries_per_peer;
        for (int ip = 0; ip < nproc; ++ip)
        {
            const size_t remaining = begin < outgoing[ip].size() ? outgoing[ip].size() - begin : 0;
            send_counts[ip] = static_cast<int>(std::min(entries_per_peer, remaining));
        }
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, communicator);
        for (int ip = 1; ip < nproc; ++ip)
        {
            send_displs[ip] = send_displs[ip - 1] + send_counts[ip - 1];
            recv_displs[ip] = recv_displs[ip - 1] + recv_counts[ip - 1];
        }
        const int send_size = send_displs.back() + send_counts.back();
        const int recv_size = recv_displs.back() + recv_counts.back();
        std::vector<int> send_meta(metadata_width * send_size);
        std::vector<int> recv_meta(metadata_width * recv_size);
        std::vector<T> send_values(send_size), recv_values(recv_size);
        for (int ip = 0; ip < nproc; ++ip)
        {
            int index = send_displs[ip];
            const size_t end = begin + static_cast<size_t>(send_counts[ip]);
            for (size_t i = begin; i < end; ++i)
            {
                const auto& entry = outgoing[ip][i];
                send_meta[metadata_width * index] = entry.ir;
                send_meta[metadata_width * index + 1] = entry.row;
                send_meta[metadata_width * index + 2] = entry.column;
                send_values[index] = entry.value;
                ++index;
            }
        }

        std::vector<int> send_counts_meta(nproc), recv_counts_meta(nproc);
        std::vector<int> send_displs_meta(nproc), recv_displs_meta(nproc);
        for (int ip = 0; ip < nproc; ++ip)
        {
            send_counts_meta[ip] = metadata_width * send_counts[ip];
            recv_counts_meta[ip] = metadata_width * recv_counts[ip];
            send_displs_meta[ip] = metadata_width * send_displs[ip];
            recv_displs_meta[ip] = metadata_width * recv_displs[ip];
        }
        MPI_Alltoallv(send_meta.data(),
                      send_counts_meta.data(),
                      send_displs_meta.data(),
                      MPI_INT,
                      recv_meta.data(),
                      recv_counts_meta.data(),
                      recv_displs_meta.data(),
                      MPI_INT,
                      communicator);
        MPI_Alltoallv(send_values.data(),
                      send_counts.data(),
                      send_displs.data(),
                      MPITraits<T>::datatype(),
                      recv_values.data(),
                      recv_counts.data(),
                      recv_displs.data(),
                      MPITraits<T>::datatype(),
                      communicator);

        const size_t old_size = received.size();
        received.resize(old_size + static_cast<size_t>(recv_size));
        for (int i = 0; i < recv_size; ++i)
        {
            received[old_size + i] = {recv_meta[metadata_width * i],
                                      recv_meta[metadata_width * i + 1],
                                      recv_meta[metadata_width * i + 2],
                                      recv_values[i]};
        }
    }
    return received;
}

struct RowRange
{
    int begin;
    int end;
};

struct DistributionContext
{
    MPI_Comm communicator;
    int rank;
    int nproc;
    int nlocal;
    RowRange row_range;
};

inline RowRange get_row_range(const int nlocal, const int rank, const int nproc)
{
    const int quotient = nlocal / nproc;
    const int remainder = nlocal % nproc;
    const int begin = rank * quotient + std::min(rank, remainder);
    return {begin, begin + quotient + (rank < remainder ? 1 : 0)};
}

inline DistributionContext make_distribution_context(const int nlocal, MPI_Comm communicator = MPI_COMM_WORLD)
{
    DistributionContext context{communicator, 0, 1, nlocal, {0, nlocal}};
    MPI_Comm_rank(communicator, &context.rank);
    MPI_Comm_size(communicator, &context.nproc);
    context.row_range = get_row_range(nlocal, context.rank, context.nproc);
    return context;
}

inline int get_row_owner(const int row, const int nlocal, const int nproc)
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
std::vector<ModuleBase::Vector3<int>> collect_global_R(const hamilt::HContainer<T>& hR,
                                                       const DistributionContext& context)
{
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
    std::vector<int> counts(context.nproc);
    MPI_Allgather(&local_size, 1, MPI_INT, counts.data(), 1, MPI_INT, context.communicator);
    std::vector<int> displs(context.nproc, 0);
    for (int ip = 1; ip < context.nproc; ++ip)
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
                   context.communicator);

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
struct DistributedRBlock
{
    std::vector<T> values;
    std::vector<int> columns;
    std::vector<int> row_counts;
};

template <typename T>
std::vector<DistributedRBlock<T>> redistribute_entries(const std::vector<std::vector<DistributedEntry<T>>>& outgoing,
                                                       const size_t block_count,
                                                       const DistributionContext& context)
{
    auto received = exchange_entries_chunked(outgoing, context.communicator);
    std::stable_sort(received.begin(),
                     received.end(),
                     [](const DistributedEntry<T>& lhs, const DistributedEntry<T>& rhs) {
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

    std::vector<DistributedRBlock<T>> blocks(block_count);
    for (auto& block: blocks)
    {
        block.row_counts.assign(context.row_range.end - context.row_range.begin, 0);
    }
    for (const auto& entry: received)
    {
        auto& block = blocks[entry.ir];
        block.values.push_back(entry.value);
        block.columns.push_back(entry.column);
        ++block.row_counts[entry.row - context.row_range.begin];
    }
    return blocks;
}

template <typename T>
std::vector<DistributedRBlock<T>> redistribute_sparse_rows(const hamilt::HContainer<T>& hR,
                                                           const Parallel_Orbitals& pv,
                                                           const std::vector<ModuleBase::Vector3<int>>& r_vectors,
                                                           const double sparse_thr,
                                                           const DistributionContext& context)
{
    std::vector<std::vector<DistributedEntry<T>>> outgoing(context.nproc);
    for_each_sparse_entry(hR,
                          pv,
                          r_vectors,
                          sparse_thr,
                          [&](const int ir, const int row, const int column, const T& value) {
                              const int owner = get_row_owner(row, context.nlocal, context.nproc);
                              outgoing[owner].push_back({ir, row, column, value});
                          });
    return redistribute_entries(outgoing, r_vectors.size(), context);
}

template <typename T>
struct DistributedSparseMatrix
{
    std::vector<DistributedRBlock<T>> blocks;
    std::vector<int> global_nnz;
    std::vector<int> nnz_by_rank;
};

template <typename T>
void collect_nnz_metadata(DistributedSparseMatrix<T>& matrix, const DistributionContext& context)
{
    std::vector<int> local_nnz(matrix.blocks.size());
    for (size_t ir = 0; ir < matrix.blocks.size(); ++ir)
    {
        local_nnz[ir] = static_cast<int>(matrix.blocks[ir].values.size());
    }
    matrix.global_nnz.resize(matrix.blocks.size());
    MPI_Allreduce(local_nnz.data(),
                  matrix.global_nnz.data(),
                  static_cast<int>(local_nnz.size()),
                  MPI_INT,
                  MPI_SUM,
                  context.communicator);
    matrix.nnz_by_rank.resize(context.nproc * local_nnz.size());
    MPI_Allgather(local_nnz.data(),
                  static_cast<int>(local_nnz.size()),
                  MPI_INT,
                  matrix.nnz_by_rank.data(),
                  static_cast<int>(local_nnz.size()),
                  MPI_INT,
                  context.communicator);
}

template <typename Result, typename Left, typename Right>
std::vector<DistributedRBlock<Result>> merge_distributed_blocks(const std::vector<DistributedRBlock<Left>>& lhs,
                                                                const std::vector<DistributedRBlock<Right>>& rhs,
                                                                const double sparse_thr)
{
    std::vector<DistributedRBlock<Result>> result(lhs.size());
    for (size_t ir = 0; ir < lhs.size(); ++ir)
    {
        auto& output = result[ir];
        output.row_counts.resize(lhs[ir].row_counts.size(), 0);
        size_t lhs_pos = 0;
        size_t rhs_pos = 0;
        for (size_t row = 0; row < output.row_counts.size(); ++row)
        {
            const size_t lhs_end = lhs_pos + lhs[ir].row_counts[row];
            const size_t rhs_end = rhs_pos + rhs[ir].row_counts[row];
            while (lhs_pos < lhs_end || rhs_pos < rhs_end)
            {
                int column = 0;
                Result value{};
                if (rhs_pos == rhs_end || (lhs_pos < lhs_end && lhs[ir].columns[lhs_pos] < rhs[ir].columns[rhs_pos]))
                {
                    column = lhs[ir].columns[lhs_pos];
                    value = lhs[ir].values[lhs_pos++];
                }
                else if (lhs_pos == lhs_end || rhs[ir].columns[rhs_pos] < lhs[ir].columns[lhs_pos])
                {
                    column = rhs[ir].columns[rhs_pos];
                    value = rhs[ir].values[rhs_pos++];
                }
                else
                {
                    column = lhs[ir].columns[lhs_pos];
                    value = lhs[ir].values[lhs_pos++] + rhs[ir].values[rhs_pos++];
                }
                if (std::abs(value) > sparse_thr)
                {
                    output.columns.push_back(column);
                    output.values.push_back(value);
                    ++output.row_counts[row];
                }
            }
        }
    }
    return result;
}

template <typename T>
DistributedSparseMatrix<T> prepare_distributed_matrix(const hamilt::HContainer<T>& hR,
                                                      const Parallel_Orbitals& pv,
                                                      const std::vector<ModuleBase::Vector3<int>>& r_vectors,
                                                      const double sparse_thr,
                                                      const DistributionContext& context)
{
    DistributedSparseMatrix<T> result;
    result.blocks = redistribute_sparse_rows(hR, pv, r_vectors, sparse_thr, context);
    collect_nnz_metadata(result, context);
    return result;
}
} // namespace detail
} // namespace ModuleIO

#endif // __MPI
#endif // MODULE_IO_HSR_MPI_DISTRIBUTION_H
