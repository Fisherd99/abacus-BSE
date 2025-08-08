#pragma once
#include "source_io/module_parameter/parameter.h"
#include "source_io/csr_reader.h"
#include "source_io/sparse_matrix.h"
#include "source_lcao/module_hcontainer/hcontainer.h"
#include "source_io/single_R_io.h"
#include "source_base/parallel_reduce.h"

namespace LR_Util
{
/// @brief rRFileReader reads the r(R) matrices from a file and converts them to HContainer format.
class rRFileReader 
{
    public:
        rRFileReader(const std::string& filename, const Parallel_Orbitals& paraV, const int nat):
        ifs(filename), paraV(paraV), nat(nat), 
        rR{hamilt::HContainer<double>(&paraV), hamilt::HContainer<double>(&paraV), hamilt::HContainer<double>(&paraV)}
        {
            if (!ifs.is_open()) {
                ModuleBase::WARNING_QUIT("rRFileReader::rRFileReader", "Failed to open file: " + filename);
            }
            parseFile();
        }

        void parseFile()
        {
            std::string line;
            std::getline(ifs, line);
            std::istringstream(line.substr(line.find(":") + 1)) >> step;
            std::getline(ifs, line);
            std::istringstream(line.substr(line.find(":") + 1)) >> matrixDimension;
            assert(matrixDimension == PARAM.globalv.nlocal);
            std::getline(ifs, line);
            std::istringstream(line.substr(line.find(":") + 1)) >> numberOfR;

            // Read the matrices
            for (int i = 0; i < numberOfR; i++)
            {
                std::array<int,3> RCoord;
                ifs >> RCoord[0] >> RCoord[1] >> RCoord[2];
                RCoordinates.push_back(RCoord);

                ModuleIO::SparseMatrix<double> matrix(matrixDimension, matrixDimension);
                auto get_data = [&]() { // [&] captures all outside variables by reference
                    int nonZero;
                    ifs >> nonZero;
                    if (nonZero == 0) {
                        return; // skip if no non-zero elements
                    }
                    std::vector<double> csr_values(nonZero);
                    std::vector<int> csr_col_ind(nonZero);
                    std::vector<int> csr_row_ptr(matrixDimension + 1);
                    for (int i = 0; i < nonZero; ++i) ifs >> csr_values[i];
                    for (int i = 0; i < nonZero; ++i) ifs >> csr_col_ind[i];
                    for (int i = 0; i < matrixDimension + 1; ++i) ifs >> csr_row_ptr[i];
                    matrix.readCSR(csr_values, csr_col_ind, csr_row_ptr);
                };
                for (int idirection = 0; idirection < 3; ++idirection) {
                    get_data();                
                    sparse_matrices[idirection].push_back(matrix);
                }

            }
        }

        /// @brief Convert the sparses_matrices[idirection] to HContainer[idirection], modified from 
        /// `module_hamilt_lcao/module_hcontainer/test/test_hcontainer_readCSR.cpp`, but consider parallel 2D
        void convert_rR_HContainer() {
            for (int iR = 0; iR < numberOfR; iR++)// R coordinate
            {
                std::array<int,3> RCoord = RCoordinates[iR];
                for (int iat = 0; iat < nat; iat++) //atom I
                {
                    int begin_row = paraV.atom_begin_row[iat];
                    int end_row = paraV.atom_begin_row[iat + 1];
                    int numberofRow = end_row - begin_row;
                    for (int jat = 0; jat < nat; jat++) //atom J
                    {
                        int begin_col = paraV.atom_begin_col[jat];
                        int end_col = paraV.atom_begin_col[jat + 1];
                        int numberofCol = end_col - begin_col;

                        GlobalV::ofs_running<<"Converting HContainer. RCoord: " << RCoord[0] << " " << RCoord[1] << " " << RCoord[2] << ", iat: " << iat << ", jat: " << jat << std::endl;

                        hamilt::BaseMatrix<double> tmp_matrix(numberofRow, numberofCol);
                        tmp_matrix.allocate(nullptr, true);
                        for (int idirection = 0; idirection < 3; ++idirection){
                            tmp_matrix.set_zero();
                            for (const auto& element: this->sparse_matrices[idirection][iR].getElements())
                            {
                                int global_row = element.first.first;
                                int global_col = element.first.second;
                                int row = paraV.global2local_row(global_row);
                                int col = paraV.global2local_col(global_col);
                                if (row < begin_row || row >= end_row || col < begin_col || col >= end_col)
                                {
                                    continue;
                                }
                                tmp_matrix.add_element(row-begin_row, col-begin_col, element.second);
/*#ifdef __DEBUG
                                GlobalV::ofs_running<<"RANK:"<<GlobalV::MY_RANK << " adding element: " << row - begin_row << " " << col - begin_col << " value: " <<std::setprecision(10) <<element.second << std::endl;
#endif*/
                            }
                            // add BaseMatrix to AtomPair
                            auto tmp_ap = hamilt::AtomPair<double>(iat, jat, RCoord[0], RCoord[1], RCoord[2], &paraV);
                            tmp_ap.allocate(nullptr, true);
                            tmp_ap.set_zero();
                            tmp_ap.convert_add(tmp_matrix, RCoord[0], RCoord[1], RCoord[2]);
/*#ifdef __DEBUG
                            GlobalV::ofs_running<<"RANK:"<<GlobalV::MY_RANK <<" Now check tmp_ap: " << iat << " " << jat << std::endl;
                            for(int iw=0;iw<numberofRow;iw++){
                                for(int jw=0;jw<numberofCol;jw++){
                                    GlobalV::ofs_running<<std::setprecision(10) << tmp_ap.get_value(iw,jw) << " ";
                                }
                                GlobalV::ofs_running<<std::endl;
                            }
#endif */
                            // add AtomPair to HContainer
                            rR[idirection].insert_pair(tmp_ap);
/*
                            GlobalV::ofs_running<<"RANK:"<<GlobalV::MY_RANK<< " Now check HC data in atom pair: " << iat << " " << jat << " R: " << RCoord[0] << " " << RCoord[1] << " " << RCoord[2] << std::endl;
                            auto ap = rR[idirection].get_atom_pair(iat, jat);
                            double* data = ap.get_HR_values(RCoord[0], RCoord[1], RCoord[2]).get_pointer();
                            for(int iw=0;iw<numberofRow;iw++){
                                for(int jw=0;jw<numberofCol;jw++){
                                    GlobalV::ofs_running<<std::setprecision(10) << *data << " ";
                                    data++;
                                }
                                GlobalV::ofs_running<<std::endl;
                            }
                            GlobalV::ofs_running.flush();
                            // seems only rank 0 can output to ofs_running
*/
                        }
                    }
                }
            }
        }

        /// @brief Output rR HContainer to a file in sparse format for test
        void output_rR_HContainer(std::string output_filename) const {
            double sparse_threshold = 1e-10;
            int precision = 8; // precision for output
            bool binary = false; // output in binary format

            std::ofstream ofs_out;
            if (GlobalV::MY_RANK == 0) {
                ofs_out.open(output_filename, std::ios::out);
                ofs_out << "STEP: " << 0 << std::endl;
                ofs_out << "Matrix Dimension of r(R): " << matrixDimension << std::endl;
                ofs_out << "Matrix number of r(R): " << rR[0].size_R_loop() << std::endl;
            }
            for(auto& Rcoord : RCoordinates) 
            {
                if(GlobalV::MY_RANK == 0) {
                    ofs_out << Rcoord[0] << " " << Rcoord[1] << " " << Rcoord[2] << std::endl;
                }
                for (int idirection = 0; idirection < 3; ++idirection) 
                {
                    rR[idirection].fix_R(Rcoord[0], Rcoord[1], Rcoord[2]);
                    std::map<size_t, std::map<size_t, double>> temp_sparse;

                    for (int iap = 0; iap < rR[idirection].size_atom_pairs(); ++iap) 
                    {
                        auto atom_pair = rR[idirection].get_atom_pair(iap);
                        auto tmp_matrix_info = atom_pair.get_matrix_values();
                        int* tmp_index = std::get<0>(tmp_matrix_info).data();//vector<int>{irow_start, irow_size, icol_start, icol_size}
                        double* tmp_data = std::get<1>(tmp_matrix_info);//matrix data
                        for (int irow = tmp_index[0]; irow < tmp_index[0] + tmp_index[1]; ++irow) {
                            for (int icol = tmp_index[2]; icol < tmp_index[2] + tmp_index[3]; ++icol) {
                                if (std::abs(*tmp_data) > sparse_threshold) {
                                    temp_sparse[paraV.local2global_row(irow)][paraV.local2global_col(icol)] = *tmp_data;
                                }
                                tmp_data++;
                            }
                        }
                    }

                    int nnz = 0;
                    for (auto& row_loop: temp_sparse)
                    {
                        nnz += row_loop.second.size();
                    }            
                    Parallel_Reduce::reduce_all(&nnz, 1);
                    if (GlobalV::MY_RANK == 0) {
                        ofs_out << nnz << std::endl;
                    }                    
                    if (nnz != 0) {
                        ModuleIO::output_single_R(ofs_out, temp_sparse, sparse_threshold, binary, this->paraV);
                    }

                    rR[idirection].unfix_R();
                }
            }
            if (GlobalV::MY_RANK == 0) {
                ofs_out.close();
            }

        } 

        std::ifstream ifs;
        const Parallel_Orbitals& paraV;
        const int nat;
        int step;
        int matrixDimension;
        int numberOfR;
        // store R coordinates
        std::vector<std::array<int,3>> RCoordinates;
        // store rR matrix. outer array: x, y, z; inner vector: R coordinates
        std::array<std::vector<ModuleIO::SparseMatrix<double>>, 3> sparse_matrices;
        // HContainer for x, y, z directions
        std::array<hamilt::HContainer<double>, 3> rR;
};
}