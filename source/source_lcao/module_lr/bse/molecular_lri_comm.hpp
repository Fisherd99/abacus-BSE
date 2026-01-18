#include "molecular_lri.h"
#include <algorithm>
#include <cstddef> // offsetof
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __MPI
#include <mpi.h>
#endif
namespace BSE
{

struct BlockHead
{
    int gr0;   // global row start
    int gc0;   // global col start
    int nr;    // rows in block
    int nc;    // cols in block
};

#ifdef __MPI
inline MPI_Datatype mpi_type_blockhead()
{
    static MPI_Datatype dt = MPI_DATATYPE_NULL;
    static bool committed = false;
    if (!committed)
    {
        int blen[4] = {1, 1, 1, 1};
        MPI_Aint disp[4];
        MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};

        disp[0] = static_cast<MPI_Aint>(offsetof(BlockHead, gr0));
        disp[1] = static_cast<MPI_Aint>(offsetof(BlockHead, gc0));
        disp[2] = static_cast<MPI_Aint>(offsetof(BlockHead, nr));
        disp[3] = static_cast<MPI_Aint>(offsetof(BlockHead, nc));

        MPI_Type_create_struct(4, blen, disp, types, &dt);
        MPI_Type_commit(&dt);
        committed = true;
    }
    return dt;
}
#endif

/// @brief W[k_ai][k_bj] to 2d local matrix WA[aik1, bjk2]
template <typename T>
void MolecularLRI<T>::transform_k_2dlocal(std::vector<T>& m_2d,
    const std::map<Tk, std::map<Tk, RI::Tensor<T>>>& m_lri,
    const Parallel_2D& pm_2d)
{
    ModuleBase::TITLE("MolecularLRI", "transform_k_2dlocal");
    ModuleBase::timer::tick("MolecularLRI", "transform_k_2dlocal");
    const int npair = this->nocc * this->nvirt;
    const double fac = 2.0 / static_cast<double>(this->nk); // factor 2 for Ha → Ry
    const int nb = pm_2d.get_block_size();
#ifdef __MPI
    MPI_Datatype g_mpi_blockhead = mpi_type_blockhead();
    // 0. outer loop: communicate per 8 k1
    for (int k1_s = 0; k1_s < this->nk; k1_s+=8)
    {
        const int k1_m = std::min(k1_s+8, this->nk);
        std::vector<int> send_head_counts(GlobalV::NPROC, 0), recv_head_counts(GlobalV::NPROC, 0);
        std::vector<int> send_buffer_counts(GlobalV::NPROC, 0), recv_buffer_counts(GlobalV::NPROC, 0);
        // 1. calculate and coummunicate block counts, then calculate block displs
        for (int kai = k1_s; kai < k1_m; ++kai)
        {
            if (std::find(this->list_k1_index.begin(), this->list_k1_index.end(), kai)
                == this->list_k1_index.end() ) continue;
            const int row_base = kai * npair;
            for (const Tk k2 : this->k2_list)
            {
                const int kbj = this->kpoint_index_map.at(k2);
                const int col_base = kbj * npair;
                for (int j = 0; j < npair; )
                {
                    const int global_col = col_base + j;
                    const int j_next = std::min(global_col/nb*nb + nb - col_base, npair);
                    for (int i = 0; i < npair; )
                    {
                        const int global_row = row_base + i;
                        const int i_next = std::min(global_row/nb*nb + nb - row_base, npair);
                        if (!pm_2d.in_this_processor(global_row, global_col))
                        {
                            const int owner = pm_2d.owner_processor(global_row, global_col);
                            ++send_head_counts[owner];
                            send_buffer_counts[owner]+=(i_next-i) * (j_next-j);
                        }
                        i = i_next;
                    }
                    j = j_next;
                }
            }
        }
        assert(send_head_counts.at(GlobalV::MY_RANK) == 0);
        assert(send_buffer_counts.at(GlobalV::MY_RANK) == 0);
        MPI_Alltoall(send_head_counts.data(), 1, MPI_INT, recv_head_counts.data(), 1, MPI_INT, pm_2d.comm());
        MPI_Alltoall(send_buffer_counts.data(), 1, MPI_INT,
                     recv_buffer_counts.data(), 1, MPI_INT, pm_2d.comm());
        
        std::vector<int> shdispls(GlobalV::NPROC, 0), rhdispls(GlobalV::NPROC, 0); //displacements of block heads
        std::vector<int> sbdispls(GlobalV::NPROC, 0), rbdispls(GlobalV::NPROC, 0); //displacements of buffer
        int send_head_total = 0, recv_head_total = 0, send_buffer_total = 0, recv_buffer_total = 0;
        for (int p = 0; p < GlobalV::NPROC; ++p)
        {
            shdispls[p] = send_head_total; send_head_total += send_head_counts[p];
            rhdispls[p] = recv_head_total; recv_head_total += recv_head_counts[p];
            sbdispls[p] = send_buffer_total; send_buffer_total += send_buffer_counts[p];
            rbdispls[p] = recv_buffer_total; recv_buffer_total += recv_buffer_counts[p];
        }

        // 2. prepare block heads and buffers for send and recv
        std::vector<BlockHead> send_heads(send_head_total), recv_heads(recv_head_total);
        std::vector<T> send_buffers(send_buffer_total), recv_buffers(recv_buffer_total);
        std::vector<int> cursor_head = shdispls;
        std::vector<int> cursor_buffer = sbdispls;
        for (int kai = k1_s; kai < k1_m; ++kai)
        {
            if (std::find(this->list_k1_index.begin(), this->list_k1_index.end(), kai)
                == this->list_k1_index.end() ) continue;
            const int row_base = kai * npair;
            const Tk k1 = RI_Util::Vector3_to_array3(this->kv.kvec_d.at(kai));
            for (const Tk k2 : this->k2_list)
            {
                const int kbj = this->kpoint_index_map.at(k2);
                const int col_base = kbj * npair;
                const RI::Tensor<T>& m_kai_kbj = m_lri.at(k1).at(k2);
                for (int j = 0; j < npair; )
                {
                    const int global_col = col_base + j;
                    const int j_next = std::min(global_col/nb*nb + nb - col_base, npair);
                    for (int i = 0; i < npair; )
                    {
                        const int global_row = row_base + i;
                        const int i_next = std::min(global_row/nb*nb + nb - row_base, npair);
                        if (pm_2d.in_this_processor(global_row, global_col))
                        {
                            const int lr0 = pm_2d.global2local_row(global_row);
                            const int lc0 = pm_2d.global2local_col(global_col);
                            const int lld = pm_2d.get_row_size();
                            for(int jj = j; jj < j_next; ++jj)
                            {
                                for(int ii = i; ii < i_next; ++ii)
                                {
                                    const int lr = lr0 + (ii - i);
                                    const int lc = lc0 + (jj - j);
                                    const int idx_2d = lr + lc * lld;
                                    m_2d[idx_2d] = (*m_kai_kbj.data)[ii + jj * npair] * fac;
                                }
                            }
                        }
                        else
                        {
                            const int owner = pm_2d.owner_processor(global_row, global_col);
                            BlockHead& head = send_heads[cursor_head[owner]++];
                            head.gr0 = global_row;
                            head.gc0 = global_col;
                            head.nr = i_next - i;
                            head.nc = j_next - j;
                            for(int jj = j; jj < j_next; ++jj)
                            {
                                for (int ii = i; ii < i_next; ++ii)
                                {
                                    send_buffers[cursor_buffer[owner]++] = (*m_kai_kbj.data)[ii + jj * npair] * fac;
                                }
                            }
                        }
                        i = i_next;
                    }
                    j = j_next;
                }
            }
        }
        for (int p = 0; p < GlobalV::NPROC; ++p)
        {
            assert(cursor_head[p] == shdispls[p] + send_head_counts[p]);
            assert(cursor_buffer[p] == sbdispls[p] + send_buffer_counts[p]);
        }
        // 3. communicate
        MPI_Alltoallv(send_heads.data(), send_head_counts.data(), shdispls.data(), g_mpi_blockhead,
                    recv_heads.data(), recv_head_counts.data(), rhdispls.data(), g_mpi_blockhead,
                    pm_2d.comm());
        MPI_Alltoallv(send_buffers.data(), send_buffer_counts.data(), sbdispls.data(), BSE_Util::MPIType<T>::value,
                    recv_buffers.data(), recv_buffer_counts.data(), rbdispls.data(), BSE_Util::MPIType<T>::value,
                    pm_2d.comm());
        // 4. unpack recv head and buffer
        int buf_cursor = 0;
        for (int iblock = 0; iblock < recv_head_total; ++iblock)
        {
            const BlockHead& rh = recv_heads[iblock];
            const int nr = rh.nr;
            const int nc = rh.nc;
            const int lr = pm_2d.global2local_row(rh.gr0);
            const int lc = pm_2d.global2local_col(rh.gc0);
            const int lld = pm_2d.get_row_size();
            for (int j = 0; j < nc; ++j)
            {
                for (int i = 0; i < nr; ++i)
                {
                    const int idx_buffer = i + j * nr;
                    const int idx_2d = (lr + i) + (lc + j) * lld;
                    m_2d[idx_2d] = recv_buffers[idx_buffer + buf_cursor];
                }
            }
            buf_cursor += nr * nc;
        }
        assert(buf_cursor == recv_buffer_total);
    }
#else
    // gather all Wk
    auto gather_matrix = [&](std::vector<T>& target,
        const std::valarray<T>& value,
        int k1_step,
        int k2_step,
        double factor) -> void
    {
        for (int j = 0; j < npair; ++j)
        {
            for (int i = 0; i < npair; ++i)
            {
                const int idx_target = (k1_step + i) + (k2_step + j) * this->ndim;
                const int idx_value = i + j * npair;
                target[idx_target] = value[idx_value] * factor;
            }
        }
    };

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (const Tk k1 : this->k1_list)
    {
        const int kai = this->kpoint_index_map.at(k1);
        const int k1_step = kai * npair;
        for (const Tk k2 : this->k2_list)
        {
            const int kbj = this->kpoint_index_map.at(k2);
            const int k2_step = kbj * npair;
            const RI::Tensor<T>& m_kai_kbj = m_lri.at(k1).at(k2);
            gather_matrix(m_2d, *m_kai_kbj.data, k1_step, k2_step, fac);
        }
    }
#endif
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "transform_k_2dlocal");
    ModuleBase::timer::tick("MolecularLRI", "transform_k_2dlocal");
}
}