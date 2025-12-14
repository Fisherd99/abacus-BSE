#pragma once
#include <cstddef>
#include "lr_util.h"
#include <algorithm>
#include "source_cell/unitcell.h"
#include "source_base/constants.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_base/module_external/lapack_connector.h"
namespace LR_Util
{
    /// =================PHYSICS====================

    template <typename TCell>
    int cal_nelec(const TCell& ucell) {
        int nelec = 0;
        for (int it = 0; it < ucell.ntype; ++it) {
            nelec += ucell.atoms[it].ncpp.zv * ucell.atoms[it].na;
}
        return nelec;
    }

    /// =================ALGORITHM====================
    ///for lack of make_unique in c++11 
    template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args &&... args)
    {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }

    //====== newers and deleters========
    /// @brief  new 2d pointer
    /// @tparam T
    /// @param size1
    /// @param size2
    template <typename T>
    void _allocate_2order_nested_ptr(T**& p2, size_t size1, size_t size2)
    {
        p2 = new T * [size1];
        for (size_t i = 0; i < size1; ++i)
        {
            p2[i] = new T[size2];
        }
    };

    /// @brief  delete 2d pointer 
    /// @tparam T 
    /// @param p2 
    /// @param size 
    template <typename T>
    void _deallocate_2order_nested_ptr(T** p2, size_t size)
    {
        if (p2 != nullptr)
        {
            for (size_t i = 0; i < size; ++i)
            {
                if (p2[i] != nullptr) { delete[] p2[i]; }
            }
            delete[] p2;
        }
    };

    inline double get_conj(const double& x)
    {
        return x;
    }
    inline std::complex<double> get_conj(const std::complex<double>& x)
    {
        return std::conj(x);
    }
    template <typename T>
    void matsym(const T* in, const int n, T* out)
    {
        for (int i = 0; i < n; ++i) {
            out[i * n + i] = 0.5 * in[i * n + i] + 0.5 * get_conj(in[i * n + i]);
        }
        for (int i = 0;i < n;++i) {
            for (int j = i + 1;j < n;++j)
            {
                out[i * n + j] = 0.5 * (in[i * n + j] + get_conj(in[j * n + i]));
                out[j * n + i] = get_conj(out[i * n + j]);
            }
        }
    }
    template <typename T>
    void matsym(T* inout, const int n)
    {
        for (int i = 0; i < n; ++i) {
            inout[i * n + i] = 0.5 * (inout[i * n + i] + get_conj(inout[i * n + i]));
        }
        for (int i = 0;i < n;++i) {
            for (int j = i + 1;j < n;++j)
            {
                inout[i * n + j] = 0.5 * (inout[i * n + j] + get_conj(inout[j * n + i]));
                inout[j * n + i] = get_conj(inout[i * n + j]);
            }
        }
    }
    template<typename T>
    bool is_hermitian(const T* mat, const int n, const double threshold){
        bool is_herm = true;
        std::vector<T> minus_mat(n*n);
        std::vector<T> sum_mat(n*n);
        for (int i = 0;i < n;++i) {
            for (int j = i;j < n;++j) {
                minus_mat[i * n + j] = mat[i * n + j] - get_conj(mat[j * n + i]);
                minus_mat[j * n + i] = -get_conj(minus_mat[i * n + j]);
                if (std::abs(minus_mat[i * n + j]) > threshold) { is_herm = false; }
                sum_mat[i * n + j] = mat[i * n + j] + get_conj(mat[j * n + i]);
                sum_mat[j * n + i] = get_conj(sum_mat[i * n + j]);
            }
        }
        const char norm_type = 'F';
        double norm1 = LapackConnector::lange(norm_type, n, n, minus_mat.data(), n, nullptr);
        double norm2 = LapackConnector::lange(norm_type, n, n, sum_mat.data(), n, nullptr);
        std::cout << "|  Hermitian check: ||A - A^H||_F = " << norm1 << ", ||A + A^H||_F = " << norm2 << std::endl;
        std::cout << "|   ||A - A^H||_F / ||A + A^H||_F = " << norm1 / norm2 << std::endl;
        return is_herm;
    }

    template<typename T>
    bool is_symmetric(const T* mat, const int n, const double threshold){
        bool is_sym = true;
        std::vector<T> minus_mat(n*n);
        std::vector<T> sum_mat(n*n);
        for (int i = 0;i < n;++i) {
            for (int j = i;j < n;++j) {
                minus_mat[i * n + j] = mat[i * n + j] - mat[j * n + i];
                minus_mat[j * n + i] = -minus_mat[i * n + j];
                if (std::abs(minus_mat[i * n + j]) > threshold) {is_sym = false; }
                sum_mat[i * n + j] = mat[i * n + j] + mat[j * n + i];
                sum_mat[j * n + i] = sum_mat[i * n + j];
            }
        }
        const char norm_type = 'F';
        double norm1 = LapackConnector::lange(norm_type, n, n, minus_mat.data(), n, nullptr);
        double norm2 = LapackConnector::lange(norm_type, n, n, sum_mat.data(), n, nullptr);
        std::cout << "|  Symmetric check: ||B - B^T||_F = " << norm1 << ", ||B + B^T||_F = " << norm2 << std::endl;
        std::cout << "|   ||B - B^T||_F / ||B + B^T||_F = " << norm1 / norm2 << std::endl;
        return is_sym;
    }

    /// get the Psi wrapper of the selected spin from the Psi object
    template<typename T>
    psi::Psi<T> get_psi_spin(const psi::Psi<T>& psi_in, const int& is, const int& nk)
    {
        return psi::Psi<T>(&psi_in(is * nk, 0, 0), 
                           nk, 
                           psi_in.get_nbands(),
                           psi_in.get_nbasis(),
                           true);
    }

    /// psi(nk=1, nbands=nb, nk * nbasis) -> psi(nb, nk, nbasis) without memory copy
    template<typename T, typename Device>
    psi::Psi<T, Device> k1_to_bfirst_wrapper(const psi::Psi<T, Device>& psi_kfirst, int nk_in, int nbasis_in)
    {
        assert(psi_kfirst.get_nk() == 1);
        assert(nk_in * nbasis_in == psi_kfirst.get_nbasis());

        int ib_now = psi_kfirst.get_current_b();
        psi_kfirst.fix_b(0);    // for get_pointer() to get the head pointer
        psi::Psi<T, Device> psi_bfirst(psi_kfirst.get_pointer(), 
                                       nk_in, 
                                       psi_kfirst.get_nbands(), 
                                       nbasis_in, 
                                       nbasis_in, 
                                       false);
        psi_kfirst.fix_b(ib_now);
        return psi_bfirst;
    }

    ///  psi(nb, nk, nbasis) -> psi(nk=1, nbands=nb, nk * nbasis)  without memory copy
    template<typename T, typename Device>
    psi::Psi<T, Device> bfirst_to_k1_wrapper(const psi::Psi<T, Device>& psi_bfirst)
    {
        int ib_now = psi_bfirst.get_current_b();
        int ik_now = psi_bfirst.get_current_k();

        psi_bfirst.fix_kb(0, 0);    // for get_pointer() to get the head pointer
        psi::Psi<T, Device> psi_kfirst(psi_bfirst.get_pointer(), 
                                       1, 
                                       psi_bfirst.get_nbands(), 
                                       psi_bfirst.get_nk() * psi_bfirst.get_nbasis(), 
                                       psi_bfirst.get_nk() * psi_bfirst.get_nbasis(),
                                       true);
        psi_bfirst.fix_kb(ik_now, ib_now);
        return psi_kfirst;
    }
//=================2D-block Parallel===============

#ifdef __MPI
    /// @brief assign global X to 2d-matrix, its col is band(excition state), and row is { spin, k-point, occ, virt }
    /// @attention pX is 2d-blocked as {occ, virt}, this assignment is used to calculate transition density matrix c_b X_{bj} c_j
    /// @todo this function is a merge version of HamiltULR::global2local and HamiltLR::global2local, they should be replaced
    template <typename T>
    void global2local_X(T* local_X, T* global_X, const int& nband, const int& nk, 
        const std::vector<int>& nocc, const std::vector<int>& nvirt, const std::vector<Parallel_2D>& pX,
        const bool openshell)
    {
        const int nspin_X = openshell ? 2 : 1;
        const std::vector<int> npairs = { nocc[0] * nvirt[0], nocc[1] * nvirt[1] };
        const int gdim = openshell ? nk * (npairs[0] + npairs[1] ) : nk * npairs[0];
        const int ldim = openshell ? nk * (pX[0].get_local_size() + pX[1].get_local_size()) : nk * pX[0].get_local_size();
        
        for (int ib = 0;ib < nband;++ib)
        {
            const int loffset_b = ib * ldim;
            const int goffset_b = ib * gdim;
            for (int is = 0;is < nspin_X;++is)
            {
                const int loffset_bs = loffset_b + is * nk * pX[0].get_local_size();
                const int goffset_bs = goffset_b + is * nk * npairs[0];                    
                for (int ik = 0;ik < nk;++ik)
                {
                    const int loffset = loffset_bs + ik * pX[is].get_local_size();
                    const int goffset = goffset_bs + ik * npairs[is];
                    for (int lo = 0;lo < pX[is].get_col_size();++lo)
                    {
                        const int go = pX[is].local2global_col(lo);
                        for (int lv = 0;lv < pX[is].get_row_size();++lv)
                        {
                            const int gv = pX[is].local2global_row(lv);
                            local_X[loffset + lo * pX[is].get_row_size() + lv] = global_X[goffset + go * nvirt[is] + gv];
                        }
                    }
                }
            }
        }
    }

    template <typename T>
    void gather_2d_to_full(const Parallel_2D& pv, const T* submat, T* fullmat, bool row_major, int global_nrow, int global_ncol)
    {
        //ModuleBase::TITLE("LR_Util", "gather_2d_to_full");
        assert(pv.get_global_row_size() == global_nrow);
        assert(pv.get_global_col_size() == global_ncol);
        auto get_mpi_datatype = []() -> MPI_Datatype {
            if (std::is_same<T, int>::value) { return MPI_INT; }
            if (std::is_same<T, float>::value) { return MPI_FLOAT; }
            else if (std::is_same<T, double>::value) { return MPI_DOUBLE; }
            if (std::is_same<T, std::complex<float>>::value) { return MPI_COMPLEX; }
            else if (std::is_same<T, std::complex<double>>::value) { return MPI_DOUBLE_COMPLEX; }
            else { throw std::runtime_error("gather_2d_to_full: unsupported type"); }
            };

        // zeros
        for (int i = 0;i < global_nrow * global_ncol;++i) { fullmat[i] = 0.0; }
        // copy
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (int j = 0;j < pv.get_col_size();++j) {
            for (int i = 0;i < pv.get_row_size();++i) {
                if (row_major) {
                    fullmat[pv.local2global_row(i) * global_ncol + pv.local2global_col(j)] = submat[i * pv.get_col_size() + j];
                } else {
                    fullmat[pv.local2global_col(j) * global_nrow + pv.local2global_row(i)] = submat[j * pv.get_row_size() + i];
                }
            }
        }
        //reduce to root
        MPI_Allreduce(MPI_IN_PLACE, fullmat, global_nrow * global_ncol, get_mpi_datatype(), MPI_SUM, pv.comm());
    };
#endif

}
