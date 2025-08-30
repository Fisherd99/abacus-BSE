#pragma once
#include "source_io/module_parameter/parameter.h"
#include "source_io/csr_reader.h"
#include "source_io/sparse_matrix.h"
#include "source_lcao/module_hcontainer/hcontainer.h"
#include "source_io/single_R_io.h"
#include "source_base/parallel_reduce.h"
#include "source_lcao/module_ri/RI_Util.h"

namespace LR_Util
{
/// @brief rRFileReader reads the r(R) + S(R) sum_{R1} R_1 / Nk from a file and converts them to HContainer format.
class rRFileReader 
{
    public:
        rRFileReader(const std::string& file_r, const std::string& file_S,
            const Parallel_Orbitals& paraV, const UnitCell& ucell, const K_Vectors& kv);

        void parseFile();

        /// @brief Convert the sparses_matrices[idirection] to HContainer[idirection], modified from 
        /// `module_hamilt_lcao/module_hcontainer/test/test_hcontainer_readCSR.cpp`, but consider parallel 2D
        void convert_rR_HContainer();

        /// @brief Output rR HContainer to a file in sparse format for test
        void output_rR_HContainer(std::string output_filename) const;

    public:
        ModuleIO::csrFileReader<double> S_csr_reader;
        std::ifstream ifs;
        const Parallel_Orbitals& paraV;
        const UnitCell& ucell;
        int step;
        int matrixDimension;
        int numberOfR;
        // store R coordinates
        std::vector<std::array<int,3>> RCoordinates;
        // store R1 coordinates for Born-von Karmen cells
        std::vector<std::array<int,3>> R1list;
        // store rR matrix. outer array: x, y, z; inner vector: R coordinates
        std::array<std::vector<ModuleIO::SparseMatrix<double>>, 3> sparse_matrices;
        // HContainer for x, y, z directions
        std::array<hamilt::HContainer<double>, 3> rR;
};
}