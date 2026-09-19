#include "module_io/hsr_mpi_distribution.h"

#include "gtest/gtest.h"
#include <algorithm>
#include <complex>
#include <vector>

namespace
{
template <typename T>
void test_multi_round_exchange(const T& value_offset)
{
    int rank = 0;
    int nproc = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    using Entry = ModuleIO::detail::DistributedEntry<T>;
    std::vector<std::vector<Entry>> outgoing(nproc);
    for (int destination = 0; destination < nproc; ++destination)
    {
        // Different peer sizes exercise both multiple rounds and zero-count
        // participation after a peer has exhausted its data.
        for (int i = 0; i < destination + 3; ++i)
        {
            outgoing[destination].push_back({rank, i, destination, value_offset + static_cast<double>(100 * rank + i)});
        }
    }

    auto received = ModuleIO::detail::exchange_entries_chunked(outgoing, MPI_COMM_WORLD, 2);
    std::sort(received.begin(), received.end(), [](const Entry& lhs, const Entry& rhs) {
        return lhs.ir < rhs.ir || (lhs.ir == rhs.ir && lhs.row < rhs.row);
    });

    ASSERT_EQ(received.size(), static_cast<size_t>(nproc * (rank + 3)));
    for (int source = 0; source < nproc; ++source)
    {
        for (int i = 0; i < rank + 3; ++i)
        {
            const auto& entry = received[source * (rank + 3) + i];
            EXPECT_EQ(entry.ir, source);
            EXPECT_EQ(entry.row, i);
            EXPECT_EQ(entry.column, rank);
            EXPECT_EQ(entry.value, value_offset + static_cast<double>(100 * source + i));
        }
    }
}

TEST(HsrMpiDistribution, ExchangesDoubleEntriesInMultipleRounds)
{
    test_multi_round_exchange(0.5);
}

TEST(HsrMpiDistribution, ExchangesComplexEntriesInMultipleRounds)
{
    test_multi_round_exchange(std::complex<double>(0.5, -0.25));
}

TEST(HsrMpiDistribution, CollectsMetadataWithinContextCommunicator)
{
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int color = world_rank % 2;
    MPI_Comm communicator = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &communicator);
    const auto context = ModuleIO::detail::make_distribution_context(4, communicator);

    ModuleIO::detail::DistributedSparseMatrix<double> matrix;
    matrix.blocks.resize(1);
    matrix.blocks[0].values.assign(color + 1, 1.0);
    ModuleIO::detail::collect_nnz_metadata(matrix, context);

    EXPECT_EQ(matrix.global_nnz, std::vector<int>(1, context.nproc * (color + 1)));
    EXPECT_EQ(matrix.nnz_by_rank, std::vector<int>(context.nproc, color + 1));
    MPI_Comm_free(&communicator);
}
} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
