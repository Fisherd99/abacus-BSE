#include "hamilt_bse_solver.h"

namespace BSE
{
template <typename T>
void printM(const std::vector<T>& A, int m, int n, std::string file, std::string name)
{
    std::ofstream ofs(file);
    ofs << name << std::endl;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            ofs << std::setw(25) << A[j * m + i];
        }
        ofs << std::endl;
    }
    ofs.close();
}

void arrayFlatten1(int nA,
                   const std::vector<std::complex<double>>& A,
                   const std::vector<std::complex<double>>& B,
                   std::vector<std::complex<double>>& result)
{ // result={{A,B},{-A*,-B*}}
    assert(result.size() == std::pow(2 * nA, 2));

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int j = 0; j < nA; j++)
    {
        for (int i = 0; i < nA; i++)
        {
            result[i + j * 2 * nA] = A[i + j * nA];
            result[nA + i + j * 2 * nA] = -std::conj(B[i + j * nA]);
            result[i + (nA + j) * 2 * nA] = B[i + j * nA];
            result[nA + i + (nA + j) * 2 * nA] = -std::conj(A[i + j * nA]);
        }
    }
}

void arrayFlatten2(int nA /*A_part dim*/,
                   const std::vector<std::complex<double>>& A,
                   const std::vector<std::complex<double>>& B,
                   std::vector<double>& result)
{ // result={{Re(A+B), Im(A-B)},{-Im(A+B), Re(A-B)}}
    assert(result.size() == std::pow(2 * nA, 2));

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int j = 0; j < nA; j++)
    {
        for (int i = 0; i < nA; i++)
        {
            result[i + j * 2 * nA] = A[i + j * nA].real() + B[i + j * nA].real();
            result[nA + i + j * 2 * nA] = -A[i + j * nA].imag() - B[i + j * nA].imag();
            result[i + (nA + j) * 2 * nA] = A[i + j * nA].imag() - B[i + j * nA].imag();
            result[nA + i + (nA + j) * 2 * nA] = A[i + j * nA].real() - B[i + j * nA].real();
        }
    }
}

void solve_full(const int& my_rank,
                const std::vector<std::complex<double>>& A_part,
                const std::vector<std::complex<double>>& B_part,
                const int& nA /*part_dim*/,
                std::vector<double>& ev,
                std::vector<std::complex<double>>& global_v)
{
    ModuleBase::TITLE("HamiltBSE", "elpa_solve_full");
    ModuleBase::timer::tick("HamiltBSE", "elpa_solve_full");

    int n = 2 * nA; // full_dim
    int nb = 1;     // block_dim
    Parallel_2D pM;
    pM.init(n, n, nb, MPI_COMM_WORLD, false);

    elpa_t elpaInstance;
    int status;

    if (elpa_init(20210430) != ELPA_OK)
    {
        fprintf(stderr, "Error: ELPA API version not supported");
        exit(1);
    }

    elpaInstance = elpa_allocate(&status);
    if (status != ELPA_OK)
    {
        std::cout << "Could not allocate elpa instance" << std::endl;
        exit(1);
    }
    elpa_set(elpaInstance, "na", n, &status);
    elpa_set(elpaInstance, "nev", n, &status);
    elpa_set(elpaInstance, "local_nrows", pM.get_row_size(), &status);
    elpa_set(elpaInstance, "local_ncols", pM.get_col_size(), &status);
    elpa_set(elpaInstance, "nblk", nb, &status);
    elpa_set(elpaInstance, "mpi_comm_parent", MPI_Comm_c2f(MPI_COMM_WORLD), &status);
    elpa_set(elpaInstance, "process_row", pM.coord[0], &status);
    elpa_set(elpaInstance, "process_col", pM.coord[1], &status);
    elpa_set(elpaInstance, "solver", ELPA_SOLVER_2STAGE, &status);
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif
    elpa_set(elpaInstance, "omp_threads", num_threads, &status);
    status = elpa_setup(elpaInstance);
    if (status != ELPA_OK)
    {
        fprintf(stderr, "Could not set up the ELPA object");
    }

    // step1: construct M
    // M = {{Re(A_part+B_part), Im(A_part-B_part)}, {-Im(A_part+B_part), Re(A_part-B_part)}}

    std::vector<double> global_M(n * n, 0.0);
    std::vector<double> M(pM.get_local_size());
    arrayFlatten2(nA, A_part, B_part, global_M);

    // stp2: construct J = {{0, I}, {-I, 0}}
    std::vector<double> global_J(n * n, 0);
    std::vector<double> J(pM.get_local_size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < nA; i++)
    {
        global_J[nA + i + i * n] = -1;
        global_J[i + (i + nA) * n] = 1;
    }

    // step3: Cholesky factorization M = U^T U
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int j = 0; j < pM.get_col_size(); ++j)
    {
        for (int i = 0; i < pM.get_row_size(); ++i)
        {
            M[j * pM.get_row_size() + i] = global_M[pM.local2global_col(j) * n + pM.local2global_row(i)];
            J[j * pM.get_row_size() + i] = global_J[pM.local2global_col(j) * n + pM.local2global_row(i)];
        }
    }

    elpa_cholesky(elpaInstance, M.data(), &status); // output M is upper triangular matrix U, M=U^T U

    // std::vector<double> global_U(n * n, 0.0);
    // LR_Util::gather_2d_to_full(pM, M.data(), global_U.data(), false, n, n);
    // if (my_rank == 0) {std::cout<<"U:"<<std::endl; printM( global_U, n, n );}

    // step4: compute anti-symmetric matrix UJL = U J U^T
    std::vector<double> UJ(pM.get_local_size(), 0.0), UJL(pM.get_local_size(), 0.0);

    ScalapackConnector::gemm('N', 'N', n, n, n, 1.0,
        M.data(), 1, 1, pM.desc, 
        J.data(), 1, 1, pM.desc,
        0.0,
        UJ.data(), 1, 1, pM.desc);

    // std::vector<double> global_UJ (n*n);
    // LR_Util::gather_2d_to_full(pM, UJ.data(), global_UJ.data(), false, n, n);   
    // if (my_rank == 0) {std::cout<<"UJ:"<<std::endl; printM( global_UJ, n, n );}

    ScalapackConnector::gemm('N', 'T', n, n, n, 1.0,
        UJ.data(), 1, 1, pM.desc,
        M.data(), 1, 1, pM.desc,
        0.0,
        UJL.data(), 1, 1, pM.desc);

    // std::vector<double> global_UJL(n*n);
    // LR_Util::gather_2d_to_full(pM, UJL.data(), global_UJL.data(), false, n, n);
    // if (my_rank == 0) {std::cout<<"UJL:"<<std::endl; printM( global_UJL, n, n );}

    // step5: compute eigenvalues ev and eigenvectors z of UJL
    std::vector<double> z(2 * pM.get_local_size()); // 2 for elpa_skew stores complex as 2 double

    elpa_skew_eigenvectors(elpaInstance, UJL.data(), ev.data(), z.data(), &status);
    assert(status == ELPA_OK);
/*
    std::vector<std::complex<double>> global_z(n * n, 0.0);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int j = 0; j < pM.get_col_size(); ++j)
    {
        for (int i = 0; i < pM.get_row_size(); ++i)
        {
            global_z[pM.local2global_col(j) * n + pM.local2global_row(i)]
                = std::complex<double>(z[j * pM.get_row_size() + i],
                                       z[(j * pM.get_row_size() + i) + pM.get_local_size()]);
        }
    }
    Parallel_Reduce::reduce_all(global_z.data(), global_z.size());
    if (my_rank == 0) {std::cout<<"z:"<<std::endl; printM( global_z, n, n );}
*/
    // step6: compute normalized eigenvectors v = SQLzΩ^(-1/2), where Ω = diag(ev)
    std::vector<std::complex<double>> v(pM.get_local_size(), 0.0);
    std::vector<std::complex<double>> global_SQ(n * n, 0.0); // global_SQ = {{I, -i I}, {-I, -i I}} / sqrt(2)
    std::vector<std::complex<double>> SQ(pM.get_local_size(), 0.0);
    std::vector<std::complex<double>> Lz(n * n, 0.0);
    std::vector<double> Lz_real(pM.get_local_size(), 0.0);
    std::vector<double> Lz_imag(pM.get_local_size(), 0.0);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < nA; i++)
    {
        global_SQ[i + i * n] = 1 / std::sqrt(2);
        global_SQ[nA + i + i * n] = -1 / std::sqrt(2);
        global_SQ[i + (i + nA) * n] = {0, -1 / std::sqrt(2)};
        global_SQ[nA + i + (i + nA) * n] = {0, -1 / std::sqrt(2)};
    }
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int j = 0; j < pM.get_col_size(); ++j)
    {
        for (int i = 0; i < pM.get_row_size(); ++i)
        {
            SQ[j * pM.get_row_size() + i] = global_SQ[pM.local2global_col(j) * n + pM.local2global_row(i)];
        }
    }

// 6.1: zΩ^(-1/2). NOTE: both positive and negative eigenvalues are handled here
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int j = 0; j < pM.get_col_size(); ++j)
    {
        double ev_sqrt = std::sqrt(std::abs(ev[pM.local2global_col(j)]));
        for (int i = 0; i < pM.get_row_size(); ++i)
        {
            z[j * pM.get_row_size() + i] /= ev_sqrt;                       // real part
            z[j * pM.get_row_size() + i + pM.get_local_size()] /= ev_sqrt; // imaginary part
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // 6.2: LzΩ^(-1/2), combine real and imaginary part
    ScalapackConnector::gemm('T', 'N', n, n, n, 1.0,
        M.data(), 1, 1, pM.desc,
        z.data(), 1, 1, pM.desc,
        0.0,
        Lz_real.data(), 1, 1, pM.desc);
    ScalapackConnector::gemm('T', 'N', n, n, n, 1.0,
        M.data(), 1, 1, pM.desc,
        z.data() + pM.get_local_size(), 1, 1, pM.desc,
        0.0,
        Lz_imag.data(), 1, 1, pM.desc);

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int j = 0; j < pM.get_col_size(); ++j)
    {
        for (int i = 0; i < pM.get_row_size(); ++i)
        {
            Lz[j * pM.get_row_size() + i]
                = std::complex<double>(Lz_real[j * pM.get_row_size() + i], Lz_imag[j * pM.get_row_size() + i]);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // 6.3: v=SQLzΩ^{-1/2}
    ScalapackConnector::gemm('N', 'N', n, n, n, 1.0,
        SQ.data(), 1, 1, pM.desc,
        Lz.data(), 1, 1, pM.desc,
        0.0,
        v.data(), 1, 1, pM.desc);

    LR_Util::gather_2d_to_full(pM, v.data(), global_v.data(), false, n, n);

    elpa_deallocate(elpaInstance, &status);
    elpa_uninit(&status);

    ModuleBase::timer::tick("HamiltBSE", "elpa_solve_full");
}
} // namespace BSE