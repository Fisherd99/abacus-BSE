#include "bse.h"

#include <elpa/elpa.h>

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

#pragma omp parallel for collapse(2)
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

#pragma omp parallel for collapse(2)
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

void solve_tda(int my_rank,
                      std::vector<std::complex<double>> A_part,
                      int nA /*part_dim*/,
                      std::vector<double>& ev,
                      std::vector<std::complex<double>>& global_v)
{
    ModuleBase::TITLE("BSE", "solver_tda");
    ModuleBase::timer::tick("BSE", "solver_tda");

    int nb = 1; // block_dim
    Parallel_2D pA, pz;
    pA.init(nA, nA, nb, MPI_COMM_WORLD, false);
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
    elpa_set(elpaInstance, "na", nA, &status);
    elpa_set(elpaInstance, "nev", nA, &status);
    elpa_set(elpaInstance, "local_nrows", pA.get_row_size(), &status);
    elpa_set(elpaInstance, "local_ncols", pA.get_col_size(), &status);
    elpa_set(elpaInstance, "nblk", nb, &status);
    elpa_set(elpaInstance, "mpi_comm_parent", MPI_Comm_c2f(MPI_COMM_WORLD), &status);
    elpa_set(elpaInstance, "process_row", pA.coord[0], &status);
    elpa_set(elpaInstance, "process_col", pA.coord[1], &status);

    status = elpa_setup(elpaInstance);
    if (status != ELPA_OK)
    {
        fprintf(stderr, "Could not set up the ELPA object");
    }

    std::vector<std::complex<double>> v(pA.get_local_size(), 0.0);

    elpa_eigenvectors(elpaInstance, A_part.data(), ev.data(), v.data(), &status);

#pragma omp parallel for collapse(2)
    for (int j = 0; j < pA.get_col_size(); ++j)
    {
        for (int i = 0; i < pA.get_row_size(); ++i)
        {
            global_v[pA.local2global_col(j) * nA + pA.local2global_row(i)] = v[j * pA.get_row_size() + i];
        }
    }
    Parallel_Reduce::reduce_all(global_v.data(), global_v.size());

    elpa_deallocate(elpaInstance, &status);
    elpa_uninit(&status);

    ModuleBase::timer::tick("BSE", "solver_TDA");
}

void solve_full(int my_rank,
                       std::vector<std::complex<double>> A_part,
                       std::vector<std::complex<double>> B_part,
                       int nA /*part_dim*/,
                       std::vector<double>& ev,
                       std::vector<std::complex<double>>& global_v)
{
    ModuleBase::TITLE("BSE", "solver_full");
    ModuleBase::timer::tick("BSE", "solver_full");

    int n = 2 * nA; // full_dim
    int nb = 1;     // block_dim
    Parallel_2D pM, pz;
    pM.init(n, n, nb, MPI_COMM_WORLD, false);
    pz.set(n, n, nb, pM.blacs_ctxt);

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

#pragma omp parallel for
    for (int i = 0; i < nA; i++)
    {
        global_J[nA + i + i * n] = -1;
        global_J[i + (i + nA) * n] = 1;
    }

    // step3: Cholesky factorization M = U^T U
    for (int j = 0; j < pM.get_col_size(); ++j)
    {
        for (int i = 0; i < pM.get_row_size(); ++i)
        {
            M[j * pM.get_row_size() + i] = global_M[pM.local2global_col(j) * n + pM.local2global_row(i)];
            J[j * pM.get_row_size() + i] = global_J[pM.local2global_col(j) * n + pM.local2global_row(i)];
        }
    }

    elpa_set(elpaInstance, "solver", ELPA_SOLVER_2STAGE, &status);
    // elpa_set(elpaInstance, "omp_threads", 2, &status);
    elpa_cholesky(elpaInstance, M.data(), &status); // output M is upper triangular matrix U, M=U^T U

    std::vector<double> global_U(n * n, 0.0);
    for (int j = 0; j < pM.get_col_size(); ++j)
    {
        for (int i = 0; i < pM.get_row_size(); ++i)
        {
            global_U[pM.local2global_col(j) * n + pM.local2global_row(i)] = M[j * pM.get_row_size() + i];
        }
    }
    Parallel_Reduce::reduce_all(global_U.data(), global_U.size());

    // step4: compute anti-symmetric matrix UJL = U J U^T
    std::vector<double> UJ(pM.get_local_size(), 0.0), UJL(pM.get_local_size(), 0.0);

    ScalapackConnector::gemm('N', 'N', n, n, n, 1.0,
        M.data(), 1, 1, pM.desc, 
        J.data(), 1, 1, pM.desc,
        0.0,
        UJ.data(), 1, 1, pM.desc);

    std::vector<double> global_UJ(n * n);
#pragma omp parallel for collapse(2)
    for (int j = 0; j < pM.get_col_size(); ++j)
    {
        for (int i = 0; i < pM.get_row_size(); ++i)
        {
            global_UJ[pM.local2global_col(j) * n + pM.local2global_row(i)] = UJ[j * pM.get_row_size() + i];
        }
    }
    Parallel_Reduce::reduce_all(global_UJ.data(), global_UJ.size());

    ScalapackConnector::gemm('N', 'T', n, n, n, 1.0,
        UJ.data(), 1, 1, pM.desc,
        M.data(), 1, 1, pM.desc,
        0.0,
        UJL.data(), 1, 1, pM.desc);

    std::vector<double> global_UJL(n * n);
#pragma omp parallel for collapse(2)
    for (int j = 0; j < pM.get_col_size(); ++j)
    {
        for (int i = 0; i < pM.get_row_size(); ++i)
        {
            global_UJL[pM.local2global_col(j) * n + pM.local2global_row(i)] = UJL[j * pM.get_row_size() + i];
        }
    }
    Parallel_Reduce::reduce_all(global_UJL.data(), global_UJL.size());

    // step5: compute eigenvalues ev and eigenvectors z of UJL
    std::vector<double> z(2 * pz.get_local_size()); // 2 for elpa_skew stores complex as 2 double
    std::vector<std::complex<double>> global_z(n * n, 0.0);

    elpa_skew_eigenvectors(elpaInstance, UJL.data(), ev.data(), z.data(), &status);
    assert(status == ELPA_OK);

#pragma omp parallel for collapse(2)
    for (int j = 0; j < pz.get_col_size(); ++j)
    {
        for (int i = 0; i < pz.get_row_size(); ++i)
        {
            global_z[pz.local2global_col(j) * n + pz.local2global_row(i)]
                = std::complex<double>(z[j * pz.get_row_size() + i],
                                       z[(j * pz.get_row_size() + i) + pz.get_local_size()]);
        }
    }
    Parallel_Reduce::reduce_all(global_z.data(), global_z.size());

    // step6: compute normalized eigenvectors v = SQLzΩ^(-1/2), where Ω = diag(ev)
    std::vector<std::complex<double>> v(pz.get_local_size(), 0.0);
    std::vector<std::complex<double>> global_SQ(n * n, 0.0); // global_SQ = {{I, -i I}, {-I, -i I}} / sqrt(2)
    std::vector<std::complex<double>> SQ(pz.get_local_size(), 0.0);
    std::vector<std::complex<double>> Lz(n * n, 0.0);
    std::vector<double> Lz_real(pz.get_local_size(), 0.0);
    std::vector<double> Lz_imag(pz.get_local_size(), 0.0);
#pragma omp parallel for
    for (int i = 0; i < nA; i++)
    {
        global_SQ[i + i * n] = 1 / std::sqrt(2);
        global_SQ[nA + i + i * n] = -1 / std::sqrt(2);
        global_SQ[i + (i + nA) * n] = {0, -1 / std::sqrt(2)};
        global_SQ[nA + i + (i + nA) * n] = {0, -1 / std::sqrt(2)};
    }
#pragma omp parallel for collapse(2)
    for (int j = 0; j < pz.get_col_size(); ++j)
    {
        for (int i = 0; i < pz.get_row_size(); ++i)
        {
            SQ[j * pz.get_row_size() + i] = global_SQ[pz.local2global_col(j) * n + pz.local2global_row(i)];
        }
    }

// 6.1: zΩ^(-1/2). FISH_TODO: only positive eigenvalues
#pragma omp parallel for collapse(2)
    for (int j = 0; j < pz.get_col_size(); ++j)
    {
        double ev_sqrt = std::sqrt(std::abs(ev[pz.local2global_col(j)]));
        for (int i = 0; i < pz.get_row_size(); ++i)
        {
            z[j * pz.get_row_size() + i] /= ev_sqrt;                       // real part
            z[j * pz.get_row_size() + i + pz.get_local_size()] /= ev_sqrt; // imaginary part
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // 6.2: LzΩ^(-1/2), combine real and imaginary part
    ScalapackConnector::gemm('T', 'N', n, n, n, 1.0,
        M.data(), 1, 1, pM.desc,
        z.data(), 1, 1, pz.desc,
        0.0,
        Lz_real.data(), 1, 1, pz.desc);
    ScalapackConnector::gemm('T', 'N', n, n, n, 1.0,
        M.data(), 1, 1, pM.desc,
        z.data() + pz.get_local_size(), 1, 1, pz.desc,
        0.0,
        Lz_imag.data(), 1, 1, pz.desc);

#pragma omp parallel for collapse(2)
    for (int j = 0; j < pz.get_col_size(); ++j)
    {
        for (int i = 0; i < pz.get_row_size(); ++i)
        {
            Lz[j * pz.get_row_size() + i]
                = std::complex<double>(Lz_real[j * pz.get_row_size() + i], Lz_imag[j * pz.get_row_size() + i]);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // 6.3: v=SQLzΩ^(-1/2)
    ScalapackConnector::gemm('N', 'N', n, n, n, 1.0,
        SQ.data(), 1, 1, pM.desc,
        Lz.data(), 1, 1, pM.desc,
        0.0,
        v.data(), 1, 1, pM.desc);
#pragma omp parallel for collapse(2)
    for (int j = 0; j < pM.get_col_size(); ++j)
    {
        for (int i = 0; i < pM.get_row_size(); ++i)
        {
            global_v[pM.local2global_col(j) * n + pM.local2global_row(i)] = v[j * pM.get_row_size() + i];
        }
    }
    Parallel_Reduce::reduce_all(global_v.data(), global_v.size());

    elpa_deallocate(elpaInstance, &status);
    elpa_uninit(&status);

    ModuleBase::timer::tick("BSE", "solver_full");
}
} // namespace BSE