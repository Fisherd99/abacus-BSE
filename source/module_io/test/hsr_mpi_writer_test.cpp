#include "module_io/hsr_mpi_writer.h"

#include "module_parameter/parameter.h"

#include "gtest/gtest.h"
#include <complex>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace
{
std::vector<char> read_file(const std::string& filename)
{
    std::ifstream input(filename, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

template <typename T>
T make_value(const int ir, const int row, const int entry);

template <>
double make_value(const int ir, const int row, const int entry)
{
    return 100.0 * ir + 10.0 * row + entry + 0.25;
}

template <>
std::complex<double> make_value(const int ir, const int row, const int entry)
{
    return {100.0 * ir + 10.0 * row + entry + 0.25, -row - 0.5 * entry};
}

template <typename T>
std::vector<ModuleIO::detail::SparseRBlock<T>> make_serial_blocks(
    const std::vector<ModuleBase::Vector3<int>>& r_vectors,
    const int nlocal)
{
    std::vector<ModuleIO::detail::SparseRBlock<T>> blocks(r_vectors.size());
    for (size_t ir = 0; ir < blocks.size(); ++ir)
    {
        blocks[ir].r = r_vectors[ir];
        blocks[ir].row_ptr.push_back(0);
        for (int row = 0; row < nlocal; ++row)
        {
            const int entries = ir == 1 ? 0 : (row + static_cast<int>(ir)) % 3;
            for (int entry = 0; entry < entries; ++entry)
            {
                blocks[ir].columns.push_back((2 * row + entry + ir) % nlocal);
                blocks[ir].values.push_back(make_value<T>(ir, row, entry));
            }
            blocks[ir].row_ptr.push_back(static_cast<int>(blocks[ir].values.size()));
        }
    }
    return blocks;
}

template <typename T>
ModuleIO::detail::DistributedSparseMatrix<T> make_distributed_matrix(
    const std::vector<ModuleIO::detail::SparseRBlock<T>>& serial,
    const ModuleIO::detail::DistributionContext& context)
{
    ModuleIO::detail::DistributedSparseMatrix<T> matrix;
    matrix.blocks.resize(serial.size());
    for (size_t ir = 0; ir < serial.size(); ++ir)
    {
        auto& local = matrix.blocks[ir];
        for (int row = context.row_range.begin; row < context.row_range.end; ++row)
        {
            const int begin = serial[ir].row_ptr[row];
            const int end = serial[ir].row_ptr[row + 1];
            local.row_counts.push_back(end - begin);
            local.values.insert(local.values.end(), serial[ir].values.begin() + begin, serial[ir].values.begin() + end);
            local.columns.insert(local.columns.end(),
                                 serial[ir].columns.begin() + begin,
                                 serial[ir].columns.begin() + end);
        }
    }
    ModuleIO::detail::collect_nnz_metadata(matrix, context);
    return matrix;
}

template <typename T>
void expect_serial_and_mpi_outputs_equal(const bool binary, const std::string& type_name)
{
    constexpr int nlocal = 7;
    constexpr int step = 13;
    PARAM.set_sys_nlocal(nlocal);
    const auto context = ModuleIO::detail::make_distribution_context(nlocal);
    const std::vector<ModuleBase::Vector3<int>> r_vectors = {{-1, 0, 2}, {0, 0, 0}, {1, 2, -1}, {3, 0, 0}};
    const std::vector<bool> output_blocks = {true, true, true, false};
    const auto serial = make_serial_blocks<T>(r_vectors, nlocal);
    const auto distributed = make_distributed_matrix(serial, context);
    const std::string suffix = binary ? "binary" : "text";
    const std::string serial_path = "/tmp/hsr_serial_" + type_name + "_" + suffix;
    const std::string mpi_path = "/tmp/hsr_mpi_" + type_name + "_" + suffix;

    if (context.rank == 0)
    {
        std::remove(serial_path.c_str());
        std::remove(mpi_path.c_str());
        ModuleIO::detail::write_serial_matrix(serial, output_blocks, serial_path, "H", step, binary, false);
    }
    MPI_Barrier(context.communicator);
    ModuleIO::detail::write_distributed_matrix(r_vectors,
                                               distributed,
                                               output_blocks,
                                               mpi_path,
                                               "H",
                                               step,
                                               binary,
                                               false,
                                               context);
    MPI_Barrier(context.communicator);

    if (context.rank == 0)
    {
        EXPECT_EQ(read_file(mpi_path), read_file(serial_path));
        std::remove(serial_path.c_str());
        std::remove(mpi_path.c_str());
    }
}

TEST(HsrMpiWriter, WritesReferencedBuffersWithSubcommunicator)
{
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int color = world_rank % 2;
    MPI_Comm communicator = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &communicator);

    int rank = 0;
    int nproc = 0;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &nproc);

    const std::string filename = "/tmp/hsr_mpi_writer_subcommunicator_" + std::to_string(color) + ".bin";
    if (rank == 0)
    {
        std::remove(filename.c_str());
    }
    MPI_Barrier(communicator);

    constexpr int header = 777;
    const int payload[2] = {world_rank, world_rank + 100};
    ModuleIO::detail::FileFragments fragments;
    if (rank == 0)
    {
        fragments.append(0, &header, sizeof(header));
    }
    fragments.append_reference(sizeof(header) + rank * sizeof(payload), payload, sizeof(payload));

    EXPECT_EQ(fragments.owned_buffers.size(), rank == 0 ? 1U : 0U);
    ModuleIO::detail::write_MPI_fragments(filename,
                                          sizeof(header) + nproc * sizeof(payload),
                                          fragments,
                                          false,
                                          communicator,
                                          5);
    MPI_Barrier(communicator);

    if (rank == 0)
    {
        std::ifstream input(filename, std::ios::binary);
        std::vector<int> contents(1 + 2 * nproc);
        input.read(reinterpret_cast<char*>(contents.data()),
                   static_cast<std::streamsize>(contents.size() * sizeof(int)));
        ASSERT_TRUE(input.good());
        EXPECT_EQ(contents[0], header);
        for (int ip = 0; ip < nproc; ++ip)
        {
            const int source_world_rank = color + 2 * ip;
            EXPECT_EQ(contents[1 + 2 * ip], source_world_rank);
            EXPECT_EQ(contents[2 + 2 * ip], source_world_rank + 100);
        }
    }

    MPI_Barrier(communicator);
    if (rank == 0)
    {
        std::remove(filename.c_str());
    }
    MPI_Comm_free(&communicator);
}

TEST(HsrMpiWriter, SerialAndMpiDoubleTextOutputsMatch)
{
    expect_serial_and_mpi_outputs_equal<double>(false, "double");
}

TEST(HsrMpiWriter, SerialAndMpiDoubleBinaryOutputsMatch)
{
    expect_serial_and_mpi_outputs_equal<double>(true, "double");
}

TEST(HsrMpiWriter, SerialAndMpiComplexTextOutputsMatch)
{
    expect_serial_and_mpi_outputs_equal<std::complex<double>>(false, "complex");
}

TEST(HsrMpiWriter, SerialAndMpiComplexBinaryOutputsMatch)
{
    expect_serial_and_mpi_outputs_equal<std::complex<double>>(true, "complex");
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
