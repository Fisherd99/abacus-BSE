#include <gtest/gtest.h>
#include "../hamilt_bse_solver.h"
#include <mpi.h>
#include "source_base/module_container/base/third_party/blas.h"

#define rand01 (static_cast<double>(rand()) / static_cast<double>(RAND_MAX) - 0.5 ) // [-0.5, 0.5]

std::vector<std::complex<double>> generate_conjugate_matrix(int n) {
    std::vector<std::complex<double>> matrix(n*n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            if(i==j) matrix[i * n + j] = std::complex<double>(5+rand01, 0); // gaurantee {{A,B},{A*,B*}} is positive definite
            else{
                matrix[i * n + j] = std::complex<double>(rand01, rand01);
                matrix[j * n + i] = std::conj(matrix[i * n + j]);
            }
        }
    }
    return matrix;
}
std::vector<std::complex<double>> generate_symmetry_matrix(int n) {
    std::vector<std::complex<double>> matrix(n*n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            if(i==j) matrix[i * n + j] = std::complex<double>(rand01, rand01);
            else{
                matrix[i * n + j] = std::complex<double>(rand01, rand01);
                matrix[j * n + i] = matrix[i * n + j];
            }
        }
    }
    return matrix;
}
void check_eq(std::complex<double>* data1, std::complex<double>* data2, int size, double eps)
    {
        for (int i = 0;i < size;++i)
        {
            EXPECT_NEAR(data1[i].real(), data2[i].real(), eps);
            EXPECT_NEAR(data1[i].imag(), data2[i].imag(), eps);
        }
    };

TEST(BSETest, skewSolver) {
    int my_rank, num_procs;
    Cblacs_pinfo(&my_rank, &num_procs); 

    int nA = 2;
    std::vector<double> ev(2*nA);
    std::vector<std::complex<double>> global_v(4*nA*nA, 0.0);
    std::vector<std::complex<double>> A_part = {
    {3.0, 0.0}, {0.5, -1.0}, {0.5, 1.0}, {6.0, 0.0}
    };
    std::vector<std::complex<double>> B_part = {
    {1.2, 0.6}, {0.4, 0.5}, {0.4, 0.5}, {1.4, 0.3}
    };
    BSE::solve_full(my_rank, A_part, B_part, nA, ev, global_v);
    EXPECT_NEAR(ev[0], -6.127295611, 1e-8); 
    EXPECT_NEAR(ev[1], -2.299184312, 1e-8);
}

TEST(BSETest, skewSolver2) {
    int my_rank, num_procs;
    Cblacs_pinfo(&my_rank, &num_procs); 

    int nA = 3;
    std::vector<double> ev(2*nA);
    std::vector<std::complex<double>> global_v(4*nA*nA, 0.0);
    std::vector<std::complex<double>> A_part = generate_conjugate_matrix(nA);
    std::vector<std::complex<double>> B_part = generate_symmetry_matrix(nA);
    BSE::solve_full(my_rank, A_part, B_part, nA, ev, global_v);

    std::vector<std::complex<double>> full_H (4*nA*nA, 0.0);
    std::vector<std::complex<double>> Hv(4*nA*nA, 0.0);
    std::vector<std::complex<double>> Ωv(4*nA*nA, 0.0);
    for (int i = 0; i < 2*nA; ++i) {
        for (int j = 0; j < 2*nA; ++j) {
            Ωv[i * 2*nA + j] = ev[i] * global_v[i * 2*nA + j];
        }
    }
    BSE::arrayFlatten1(nA, A_part, B_part, full_H);

    container::BlasConnector::gemm('N', 'N', 2*nA, 2*nA, 2*nA, 1.0,
        full_H.data(), 2*nA,
        global_v.data(), 2*nA,
        0.0,
        Hv.data(), 2*nA);
    check_eq(Hv.data(), Ωv.data(), 4*nA*nA, 1e-8);

    std::vector<std::complex<double>> identity(4*nA*nA, 0.0);
    for (int i = 0; i < 2*nA; ++i) {
        identity[i * 2*nA + i] = 1.0;
    }
    std::vector<std::complex<double>> left_v(4*nA*nA, 0.0);
    for (int i = 0; i < nA; ++i) {
        for (int j = 0; j < nA; ++j) {
            left_v[i * 2*nA + j] = -global_v[i * 2*nA + j];
            left_v[(nA+i) * 2*nA + j] = global_v[(nA+i) * 2*nA + j];
            left_v[i * 2*nA + nA+j] = global_v[i * 2*nA + nA+j];
            left_v[(nA+i) * 2*nA + nA+j] = -global_v[(nA+i) * 2*nA + nA+j];
        }
    }

    container::BlasConnector::gemm('C', 'N', 2*nA, 2*nA, 2*nA, 1.0,
        left_v.data(), 2*nA,
        global_v.data(), 2*nA,
        0.0,
        Ωv.data(), 2*nA);// overwriten Ωv by left_v.v
    check_eq(Ωv.data(), identity.data(), 4*nA*nA, 1e-8);
}


int main(int argc, char **argv) {
    srand(time(nullptr));
    int thread_level = -1;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);
    if (thread_level != MPI_THREAD_MULTIPLE)
    {
        std::cerr << "MPI_Init_thread request " << MPI_THREAD_MULTIPLE << " but provide " << thread_level << std::endl;
    }
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}