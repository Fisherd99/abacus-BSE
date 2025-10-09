#pragma once

#include "hamilt_bse_solver.h"
namespace BSE
{
template <typename T>
void solve_tda(const int& my_rank,
                const std::vector<T>& A_part,
                const Parallel_2D& pA,
                const int& nA /*part_dim*/,
                std::vector<double>& ev,
                std::vector<T>& global_v)
{
    ModuleBase::TITLE("BSE", "solver_tda");
    ModuleBase::timer::tick("BSE", "solver_tda");

    assert(pA.get_row_size() == nA);
    assert(pA.get_col_size() == nA);
    assert(A_part.size() == nA * nA);

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
    elpa_set(elpaInstance, "nblk", pA.get_block_size(), &status);
    elpa_set(elpaInstance, "mpi_comm_parent", MPI_Comm_c2f(MPI_COMM_WORLD), &status);
    elpa_set(elpaInstance, "process_row", pA.coord[0], &status);
    elpa_set(elpaInstance, "process_col", pA.coord[1], &status);
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
    std::vector<T> A(pA.get_local_size(), 0.0);
    std::vector<T> v(pA.get_local_size(), 0.0);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int j = 0; j < pA.get_col_size(); ++j)
    {
        for (int i = 0; i < pA.get_row_size(); ++i)
        {
            A[j * pA.get_row_size() + i] = A_part[pA.local2global_col(j) * nA + pA.local2global_row(i)];
        }
    }
    elpa_eigenvectors(elpaInstance, A.data(), ev.data(), v.data(), &status);
    LR_Util::gather_2d_to_full(pA, v.data(), global_v.data(), false, nA, nA);

    elpa_deallocate(elpaInstance, &status);
    elpa_uninit(&status);

    ModuleBase::timer::tick("BSE", "solver_TDA");
}
}