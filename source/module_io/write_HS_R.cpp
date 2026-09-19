#include "write_HS_R.h"

#include "hsr_mpi_distribution.h"
#include "hsr_mpi_writer.h"
#include "hsr_sparse_utils.h"
#include "module_base/timer.h"
#include "module_hamilt_lcao/hamilt_lcaodft/LCAO_HS_arrays.hpp"
#include "module_hamilt_lcao/hamilt_lcaodft/spar_dh.h"
#include "module_hamilt_lcao/hamilt_lcaodft/spar_hsr.h"
#include "module_hamilt_lcao/hamilt_lcaodft/spar_st.h"
#include "module_hamilt_lcao/module_dftu/dftu.h"
#include "module_hamilt_lcao/module_tddft/td_velocity.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_parameter/parameter.h"
#include "write_HS_sparse.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <tuple>

namespace
{
using ModuleIO::detail::collect_local_R;
using ModuleIO::detail::equal_R;
using ModuleIO::detail::less_R;
using ModuleIO::detail::make_sparse_blocks;
using ModuleIO::detail::make_td_sparse_blocks;
using ModuleIO::detail::merge_R_vectors;
using ModuleIO::detail::SparseRBlock;
using ModuleIO::detail::write_serial_matrix;

#ifdef __MPI
using ModuleIO::detail::collect_global_R;
using ModuleIO::detail::collect_nnz_metadata;
using ModuleIO::detail::DistributedEntry;
using ModuleIO::detail::DistributedRBlock;
using ModuleIO::detail::DistributedSparseMatrix;
using ModuleIO::detail::DistributionContext;
using ModuleIO::detail::get_row_owner;
using ModuleIO::detail::merge_distributed_blocks;
using ModuleIO::detail::prepare_distributed_matrix;
using ModuleIO::detail::redistribute_entries;
using ModuleIO::detail::redistribute_sparse_rows;

DistributionContext make_distribution_context()
{
    return ModuleIO::detail::make_distribution_context(PARAM.globalv.nlocal);
}

using ModuleIO::detail::FileFragments;
using ModuleIO::detail::write_distributed_matrix;
using ModuleIO::detail::write_MPI_fragments;

void calculate_dftu_R(const int spin, double* sr, double* hr)
{
    GlobalC::dftu.cal_eff_pot_mat_R_double(spin, sr, hr);
}

void calculate_dftu_R(const int spin, std::complex<double>* sr, std::complex<double>* hr)
{
    GlobalC::dftu.cal_eff_pot_mat_R_complex_double(spin, sr, hr);
}

DistributedSparseMatrix<std::complex<double>> prepare_td_distributed_matrix(
    const hamilt::HContainer<double>& hR,
    const hamilt::HContainer<std::complex<double>>& td_hR,
    const Parallel_Orbitals& pv,
    const std::vector<ModuleBase::Vector3<int>>& r_vectors,
    const double sparse_thr,
    const DistributionContext& context)
{
    const auto base_blocks = redistribute_sparse_rows(hR, pv, r_vectors, sparse_thr, context);
    // TD velocity-gauge matrices were historically cached at this threshold.
    const auto td_blocks = redistribute_sparse_rows(td_hR, pv, r_vectors, 1.0e-10, context);

    DistributedSparseMatrix<std::complex<double>> result;
    result.blocks = merge_distributed_blocks<std::complex<double>>(base_blocks, td_blocks, sparse_thr);
    collect_nnz_metadata(result, context);
    return result;
}

template <typename T>
DistributedSparseMatrix<T> prepare_dftu_distributed_matrix(const hamilt::HContainer<T>& hR,
                                                           const hamilt::HContainer<T>& sR,
                                                           const Parallel_Orbitals& pv,
                                                           const std::vector<ModuleBase::Vector3<int>>& r_vectors,
                                                           const int spin,
                                                           const double sparse_thr,
                                                           const DistributionContext& context)
{
    const auto base_blocks = redistribute_sparse_rows(hR, pv, r_vectors, sparse_thr, context);

    // Reproduce the legacy DFT+U output formula using the native block-cyclic
    // matrix layout required by ScaLAPACK. S(R) is filtered before the
    // multiplication, exactly as in sparse_format::cal_HR_dftu[_soc].
    std::vector<std::vector<std::pair<int, T>>> local_s_entries(r_vectors.size());
    std::vector<int> local_s_nnz(r_vectors.size(), 0);
    for (int iap = 0; iap < sR.size_atom_pairs(); ++iap)
    {
        const auto& atom_pair = sR.get_atom_pair(iap);
        for (int local_ir = 0; local_ir < atom_pair.get_R_size(); ++local_ir)
        {
            const auto r = atom_pair.get_R_index(local_ir);
            const auto r_iter = std::lower_bound(r_vectors.begin(), r_vectors.end(), r, less_R);
            if (r_iter == r_vectors.end() || !equal_R(*r_iter, r))
            {
                continue;
            }
            const size_t ir = std::distance(r_vectors.begin(), r_iter);
            const auto matrix_info = atom_pair.get_matrix_values(local_ir);
            const auto& indexes = std::get<0>(matrix_info);
            const T* data = std::get<1>(matrix_info);
            for (int local_row = 0; local_row < indexes[1]; ++local_row)
            {
                for (int local_col = 0; local_col < indexes[3]; ++local_col)
                {
                    const T value = data[local_row * indexes[3] + local_col];
                    if (std::abs(value) <= sparse_thr)
                    {
                        continue;
                    }
                    const int row = indexes[0] + local_row;
                    const int col = indexes[2] + local_col;
                    const int index = ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER(PARAM.inp.ks_solver)
                                          ? row + col * pv.nrow
                                          : row * pv.ncol + col;
                    local_s_entries[ir].emplace_back(index, value);
                    ++local_s_nnz[ir];
                }
            }
        }
    }
    std::vector<int> global_s_nnz(r_vectors.size(), 0);
    MPI_Allreduce(local_s_nnz.data(),
                  global_s_nnz.data(),
                  static_cast<int>(r_vectors.size()),
                  MPI_INT,
                  MPI_SUM,
                  context.communicator);

    std::vector<std::vector<DistributedEntry<T>>> outgoing(context.nproc);
    std::vector<T> sr_tmp(pv.nloc);
    std::vector<T> hu_tmp(pv.nloc);
    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        if (global_s_nnz[ir] == 0)
        {
            continue;
        }
        std::fill(sr_tmp.begin(), sr_tmp.end(), T{});
        std::fill(hu_tmp.begin(), hu_tmp.end(), T{});
        for (const auto& entry: local_s_entries[ir])
        {
            sr_tmp[entry.first] = entry.second;
        }
        calculate_dftu_R(spin, sr_tmp.data(), hu_tmp.data());

        for (int row = 0; row < context.nlocal; ++row)
        {
            const int local_row = pv.global2local_row(row);
            if (local_row < 0)
            {
                continue;
            }
            const int owner = get_row_owner(row, context.nlocal, context.nproc);
            for (int col = 0; col < context.nlocal; ++col)
            {
                const int local_col = pv.global2local_col(col);
                if (local_col < 0)
                {
                    continue;
                }
                const int index = ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER(PARAM.inp.ks_solver)
                                      ? local_row + local_col * pv.nrow
                                      : local_row * pv.ncol + local_col;
                if (std::abs(hu_tmp[index]) > sparse_thr)
                {
                    outgoing[owner].push_back({static_cast<int>(ir), row, col, hu_tmp[index]});
                }
            }
        }
    }
    const auto u_blocks = redistribute_entries(outgoing, r_vectors.size(), context);

    DistributedSparseMatrix<T> result;
    result.blocks = merge_distributed_blocks<T>(base_blocks, u_blocks, sparse_thr);
    collect_nnz_metadata(result, context);
    return result;
}

DistributedSparseMatrix<std::complex<double>> add_td_correction(const DistributedSparseMatrix<double>& base_matrix,
                                                                const hamilt::HContainer<std::complex<double>>& td_hR,
                                                                const Parallel_Orbitals& pv,
                                                                const std::vector<ModuleBase::Vector3<int>>& r_vectors,
                                                                const double sparse_thr,
                                                                const DistributionContext& context)
{
    // TD velocity-gauge matrices were historically cached at this threshold.
    const auto td_blocks = redistribute_sparse_rows(td_hR, pv, r_vectors, 1.0e-10, context);

    DistributedSparseMatrix<std::complex<double>> result;
    result.blocks = merge_distributed_blocks<std::complex<double>>(base_matrix.blocks, td_blocks, sparse_thr);
    collect_nnz_metadata(result, context);
    return result;
}

#endif

template <typename HasEntries>
std::vector<bool> select_output_blocks(const std::vector<ModuleBase::Vector3<int>>& r_vectors,
                                       LCAO_HS_Arrays& arrays,
                                       HasEntries has_entries)
{
    std::vector<bool> output_blocks(r_vectors.size(), false);
    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        output_blocks[ir] = has_entries(ir);
        if (output_blocks[ir])
        {
            const auto& r = r_vectors[ir];
            arrays.output_R_coor.insert(Abfs::Vector3_Order<int>(r.x, r.y, r.z));
        }
    }
    return output_blocks;
}

std::string make_matrix_output_path(const std::string& filename, const int step)
{
    std::ostringstream path;
    if (PARAM.inp.calculation == "md" && !PARAM.inp.out_app_flag)
    {
        path << PARAM.globalv.global_matrix_dir << step << "_" << filename;
    }
    else
    {
        path << PARAM.globalv.global_out_dir << filename;
    }
    return path.str();
}

template <typename T>
std::vector<SparseRBlock<T>> make_empty_sparse_blocks(const std::vector<ModuleBase::Vector3<int>>& r_vectors)
{
    std::vector<SparseRBlock<T>> blocks(r_vectors.size());
    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        blocks[ir].r = r_vectors[ir];
    }
    return blocks;
}

#ifdef __MPI
template <typename T>
DistributedSparseMatrix<T> make_empty_distributed_matrix(const size_t block_count, const DistributionContext& context)
{
    DistributedSparseMatrix<T> matrix;
    matrix.blocks.resize(block_count);
    for (auto& block: matrix.blocks)
    {
        block.row_counts.assign(context.row_range.end - context.row_range.begin, 0);
    }
    matrix.global_nnz.assign(block_count, 0);
    matrix.nnz_by_rank.assign(context.nproc * block_count, 0);
    return matrix;
}
#endif

template <typename H, typename S>
void write_serial_hsr_matrices(const std::vector<SparseRBlock<H>>& h_down,
                               const std::vector<SparseRBlock<H>>* h_up,
                               const std::vector<SparseRBlock<S>>& overlap,
                               const std::vector<bool>& output_blocks,
                               const std::string& hr_up_path,
                               const std::string& hr_down_path,
                               const std::string& sr_path,
                               const int step,
                               const bool binary,
                               const bool append)
{
    if (h_up != nullptr)
    {
        write_serial_matrix(*h_up, output_blocks, hr_up_path, "H", step, binary, append);
        write_serial_matrix(h_down, output_blocks, hr_down_path, "H", step, binary, append);
    }
    else
    {
        write_serial_matrix(h_down, output_blocks, hr_up_path, "H", step, binary, append);
    }
    write_serial_matrix(overlap, output_blocks, sr_path, "S", step, binary, append);
}

#ifdef __MPI
template <typename H, typename S>
void write_distributed_hsr_matrices(const std::vector<ModuleBase::Vector3<int>>& r_vectors,
                                    const DistributedSparseMatrix<H>& h_down,
                                    const DistributedSparseMatrix<H>* h_up,
                                    const DistributedSparseMatrix<S>& overlap,
                                    const std::vector<bool>& output_blocks,
                                    const std::string& hr_up_path,
                                    const std::string& hr_down_path,
                                    const std::string& sr_path,
                                    const int step,
                                    const bool binary,
                                    const bool append,
                                    const DistributionContext& context)
{
    if (h_up != nullptr)
    {
        write_distributed_matrix(r_vectors, *h_up, output_blocks, hr_up_path, "H", step, binary, append, context);
        write_distributed_matrix(r_vectors, h_down, output_blocks, hr_down_path, "H", step, binary, append, context);
    }
    else
    {
        write_distributed_matrix(r_vectors, h_down, output_blocks, hr_up_path, "H", step, binary, append, context);
    }
    write_distributed_matrix(r_vectors, overlap, output_blocks, sr_path, "S", step, binary, append, context);
}
#endif

struct HsrOutputOptions
{
    std::string hr_up_path;
    std::string hr_down_path;
    std::string sr_path;
    int step;
    int nspin;
    bool binary;
    bool append;
    double sparse_thr;
};

template <typename HMatrix, typename SMatrix>
struct PreparedHsrMatrices
{
    HMatrix h_down;
    HMatrix h_up;
    SMatrix overlap;
    bool has_h_up;
};

struct SerialHsrBackend
{
    const Parallel_Orbitals& pv;
    int nlocal;

    template <typename T>
    std::vector<SparseRBlock<T>> prepare(const hamilt::HContainer<T>& matrix,
                                         const std::vector<ModuleBase::Vector3<int>>& r_vectors,
                                         const double sparse_thr) const
    {
        return make_sparse_blocks(matrix, pv, r_vectors, nlocal, sparse_thr);
    }

    template <typename T>
    std::vector<SparseRBlock<T>> empty_like(const std::vector<SparseRBlock<T>>&,
                                            const std::vector<ModuleBase::Vector3<int>>& r_vectors) const
    {
        return make_empty_sparse_blocks<T>(r_vectors);
    }

    template <typename T>
    bool has_entries(const std::vector<SparseRBlock<T>>& matrix, const size_t ir) const
    {
        return !matrix[ir].values.empty();
    }

    template <typename H, typename S>
    void write(const std::vector<ModuleBase::Vector3<int>>&,
               const std::vector<SparseRBlock<H>>& h_down,
               const std::vector<SparseRBlock<H>>* h_up,
               const std::vector<SparseRBlock<S>>& overlap,
               const std::vector<bool>& output_blocks,
               const HsrOutputOptions& options) const
    {
        write_serial_hsr_matrices(h_down,
                                  h_up,
                                  overlap,
                                  output_blocks,
                                  options.hr_up_path,
                                  options.hr_down_path,
                                  options.sr_path,
                                  options.step,
                                  options.binary,
                                  options.append);
    }
};

#ifdef __MPI
struct DistributedHsrBackend
{
    const Parallel_Orbitals& pv;
    DistributionContext context;

    template <typename T>
    DistributedSparseMatrix<T> prepare(const hamilt::HContainer<T>& matrix,
                                       const std::vector<ModuleBase::Vector3<int>>& r_vectors,
                                       const double sparse_thr) const
    {
        return prepare_distributed_matrix(matrix, pv, r_vectors, sparse_thr, context);
    }

    template <typename T>
    DistributedSparseMatrix<T> empty_like(const DistributedSparseMatrix<T>&,
                                          const std::vector<ModuleBase::Vector3<int>>& r_vectors) const
    {
        return make_empty_distributed_matrix<T>(r_vectors.size(), context);
    }

    template <typename T>
    bool has_entries(const DistributedSparseMatrix<T>& matrix, const size_t ir) const
    {
        return matrix.global_nnz[ir] > 0;
    }

    template <typename H, typename S>
    void write(const std::vector<ModuleBase::Vector3<int>>& r_vectors,
               const DistributedSparseMatrix<H>& h_down,
               const DistributedSparseMatrix<H>* h_up,
               const DistributedSparseMatrix<S>& overlap,
               const std::vector<bool>& output_blocks,
               const HsrOutputOptions& options) const
    {
        write_distributed_hsr_matrices(r_vectors,
                                       h_down,
                                       h_up,
                                       overlap,
                                       output_blocks,
                                       options.hr_up_path,
                                       options.hr_down_path,
                                       options.sr_path,
                                       options.step,
                                       options.binary,
                                       options.append,
                                       context);
    }
};
#endif

template <typename T, typename Backend, typename MakeHamiltonian>
void prepare_and_write_hsr(hamilt::HamiltLCAO<std::complex<double>, T>& hamiltonian,
                           const hamilt::HContainer<T>& overlap_source,
                           const std::vector<ModuleBase::Vector3<int>>& r_vectors,
                           LCAO_HS_Arrays& arrays,
                           const HsrOutputOptions& options,
                           const Backend& backend,
                           MakeHamiltonian make_hamiltonian)
{
    auto h_down = make_hamiltonian(options.nspin == 2 ? 1 : 0);
    using HMatrix = decltype(h_down);
    using SMatrix = decltype(backend.prepare(overlap_source, r_vectors, options.sparse_thr));
    auto h_up = backend.empty_like(h_down, r_vectors);
    const bool has_h_up = options.nspin == 2;
    if (has_h_up && PARAM.inp.vl_in_h)
    {
        hamiltonian.updateHR(0);
        h_up = make_hamiltonian(0);
    }
    PreparedHsrMatrices<HMatrix, SMatrix> matrices{std::move(h_down),
                                                   std::move(h_up),
                                                   backend.prepare(overlap_source, r_vectors, options.sparse_thr),
                                                   has_h_up};

    const auto output_blocks = select_output_blocks(r_vectors, arrays, [&](const size_t ir) {
        return backend.has_entries(matrices.h_down, ir) || backend.has_entries(matrices.overlap, ir)
               || (matrices.has_h_up && backend.has_entries(matrices.h_up, ir));
    });
    backend.write(r_vectors,
                  matrices.h_down,
                  matrices.has_h_up ? &matrices.h_up : nullptr,
                  matrices.overlap,
                  output_blocks,
                  options);
}

HsrOutputOptions make_hsr_output_options(const std::string& hr_up_filename,
                                         const std::string& hr_down_filename,
                                         const std::string& sr_filename,
                                         const int step,
                                         const int nspin,
                                         const bool binary,
                                         const double sparse_thr)
{
    return {make_matrix_output_path(hr_up_filename, step),
            make_matrix_output_path(hr_down_filename, step),
            make_matrix_output_path(sr_filename, step),
            step,
            nspin,
            binary,
            PARAM.inp.calculation == "md" && PARAM.inp.out_app_flag && step != 0,
            sparse_thr};
}

template <typename T>
void output_direct_hsr(hamilt::HamiltLCAO<std::complex<double>, T>& hamiltonian,
                       const Parallel_Orbitals& pv,
                       LCAO_HS_Arrays& arrays,
                       const HsrOutputOptions& options)
{
    auto& hR = *hamiltonian.getHR();
    auto& sR = *hamiltonian.getSR();
#ifdef __MPI
    const auto context = make_distribution_context();
    const auto r_vectors = PARAM.inp.dft_plus_u == 2
                               ? collect_global_R(hR, context)
                               : merge_R_vectors(collect_global_R(hR, context), collect_global_R(sR, context));
    const DistributedHsrBackend backend{pv, context};
    const auto make_h_matrix = [&](const int spin) {
        return PARAM.inp.dft_plus_u == 2
                   ? prepare_dftu_distributed_matrix(hR, sR, pv, r_vectors, spin, options.sparse_thr, context)
                   : backend.prepare(hR, r_vectors, options.sparse_thr);
    };
    prepare_and_write_hsr(hamiltonian, sR, r_vectors, arrays, options, backend, make_h_matrix);
#else
    const auto r_vectors = merge_R_vectors(collect_local_R(hR), collect_local_R(sR));
    const SerialHsrBackend backend{pv, PARAM.globalv.nlocal};
    const auto make_h_matrix = [&](const int) { return backend.prepare(hR, r_vectors, options.sparse_thr); };
    prepare_and_write_hsr(hamiltonian, sR, r_vectors, arrays, options, backend, make_h_matrix);
#endif
}

void output_td_hsr(hamilt::HamiltLCAO<std::complex<double>, double>& hamiltonian,
                   const hamilt::HContainer<std::complex<double>>& td_hR,
                   const Parallel_Orbitals& pv,
                   LCAO_HS_Arrays& arrays,
                   const HsrOutputOptions& options)
{
    auto& hR = *hamiltonian.getHR();
    auto& sR = *hamiltonian.getSR();
    // The velocity-gauge correction is spin independent. For nspin=2,
    // combine the same correction with both spin channels.
#ifdef __MPI
    const auto context = make_distribution_context();
    auto r_vectors = merge_R_vectors(collect_global_R(hR, context), collect_global_R(sR, context));
    r_vectors = merge_R_vectors(r_vectors, collect_global_R(td_hR, context));
    const DistributedHsrBackend backend{pv, context};
    const auto make_h_matrix = [&](const int spin) {
        if (PARAM.inp.dft_plus_u == 2)
        {
            const auto h_dftu
                = prepare_dftu_distributed_matrix(hR, sR, pv, r_vectors, spin, options.sparse_thr, context);
            return add_td_correction(h_dftu, td_hR, pv, r_vectors, options.sparse_thr, context);
        }
        return prepare_td_distributed_matrix(hR, td_hR, pv, r_vectors, options.sparse_thr, context);
    };
    prepare_and_write_hsr(hamiltonian, sR, r_vectors, arrays, options, backend, make_h_matrix);
#else
    auto r_vectors = merge_R_vectors(collect_local_R(hR), collect_local_R(sR));
    r_vectors = merge_R_vectors(r_vectors, collect_local_R(td_hR));
    const SerialHsrBackend backend{pv, PARAM.globalv.nlocal};
    const auto make_h_matrix = [&](const int) {
        return make_td_sparse_blocks(hR, td_hR, pv, r_vectors, backend.nlocal, options.sparse_thr);
    };
    prepare_and_write_hsr(hamiltonian, sR, r_vectors, arrays, options, backend, make_h_matrix);
#endif
}

void output_legacy_hsr(
    const UnitCell& ucell,
    const int step,
    const Parallel_Orbitals& pv,
    LCAO_HS_Arrays& arrays,
    const Grid_Driver& grid,
    const K_Vectors& kv,
    hamilt::Hamilt<std::complex<double>>* hamiltonian,
#ifdef __EXX
    const std::vector<std::map<int, std::map<ModuleIO::TAC, RI::Tensor<double>>>>* hexx_real,
    const std::vector<std::map<int, std::map<ModuleIO::TAC, RI::Tensor<std::complex<double>>>>>* hexx_complex,
#endif
    const std::string& sr_filename,
    const std::string& hr_up_filename,
    const std::string& hr_down_filename,
    const bool binary,
    const double sparse_thr)
{
    const int nspin = PARAM.inp.nspin;
    if (nspin == 1 || nspin == 4)
    {
        sparse_format::cal_HSR(ucell,
                               pv,
                               arrays,
                               grid,
                               0,
                               sparse_thr,
                               kv.nmp,
                               hamiltonian
#ifdef __EXX
                               ,
                               hexx_real,
                               hexx_complex
#endif
        );
    }
    else if (nspin == 2)
    {
        int spin = 1;
        sparse_format::cal_HSR(ucell,
                               pv,
                               arrays,
                               grid,
                               spin,
                               sparse_thr,
                               kv.nmp,
                               hamiltonian
#ifdef __EXX
                               ,
                               hexx_real,
                               hexx_complex
#endif
        );
        if (PARAM.inp.vl_in_h)
        {
            hamiltonian->refresh();
            hamiltonian->updateHk(0);
            spin = 0;
        }
        sparse_format::cal_HSR(ucell,
                               pv,
                               arrays,
                               grid,
                               spin,
                               sparse_thr,
                               kv.nmp,
                               hamiltonian
#ifdef __EXX
                               ,
                               hexx_real,
                               hexx_complex
#endif
        );
    }

    ModuleIO::save_HSR_sparse(step, pv, arrays, sparse_thr, binary, sr_filename, hr_up_filename, hr_down_filename);
    sparse_format::destroy_HS_R_sparse(arrays);
}

template <typename T>
void output_overlap_matrix(const hamilt::HContainer<T>& sR,
                           const Parallel_Orbitals& pv,
                           const std::string& filename,
                           const bool binary,
                           const double sparse_thr)
{
#ifdef __MPI
    const auto context = make_distribution_context();
    const auto r_vectors = collect_global_R(sR, context);
    const auto matrix = prepare_distributed_matrix(sR, pv, r_vectors, sparse_thr, context);
    std::vector<bool> output_blocks(r_vectors.size(), false);
    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        output_blocks[ir] = matrix.global_nnz[ir] > 0;
    }
    write_distributed_matrix(r_vectors, matrix, output_blocks, filename, "S", 0, binary, false, context);
#else
    const auto r_vectors = collect_local_R(sR);
    const auto blocks = make_sparse_blocks(sR, pv, r_vectors, PARAM.globalv.nlocal, sparse_thr);
    std::vector<bool> output_blocks(r_vectors.size(), false);
    for (size_t ir = 0; ir < r_vectors.size(); ++ir)
    {
        output_blocks[ir] = !blocks[ir].values.empty();
    }
    write_serial_matrix(blocks, output_blocks, filename, "S", 0, binary, false);
#endif
}
} // namespace

// if 'binary=true', output binary file.
// The 'sparse_thr' is the accuracy of the sparse matrix.
// If the absolute value of the matrix element is less than or equal to the
// 'sparse_thr', it will be ignored.
void ModuleIO::output_HSR(const UnitCell& ucell,
                          const int& istep,
                          const ModuleBase::matrix& v_eff,
                          const Parallel_Orbitals& pv,
                          LCAO_HS_Arrays& HS_Arrays,
                          const Grid_Driver& grid, // mohan add 2024-04-06
                          const K_Vectors& kv,
                          hamilt::Hamilt<std::complex<double>>* p_ham,
#ifdef __EXX
                          const std::vector<std::map<int, std::map<TAC, RI::Tensor<double>>>>* Hexxd,
                          const std::vector<std::map<int, std::map<TAC, RI::Tensor<std::complex<double>>>>>* Hexxc,
#endif
                          const std::string& SR_filename,
                          const std::string& HR_filename_up,
                          const std::string HR_filename_down,
                          const bool& binary,
                          const double& sparse_thr)
{
    ModuleBase::TITLE("ModuleIO", "output_HSR");
    ModuleBase::timer::tick("ModuleIO", "output_HSR");

    const int nspin = PARAM.inp.nspin;

    const auto* td_hR = TD_Velocity::tddft_velocity && TD_Velocity::td_vel_op != nullptr
                            ? TD_Velocity::td_vel_op->get_HR_pointer()
                            : nullptr;
    if (TD_Velocity::tddft_velocity && (nspin == 1 || nspin == 2) && td_hR == nullptr)
    {
        ModuleBase::WARNING_QUIT("ModuleIO::output_HSR",
                                 "velocity-gauge TDDFT H(R) is unavailable; cannot write a complete Hamiltonian");
    }
    bool use_direct_hcontainer = !TD_Velocity::tddft_velocity || nspin == 1 || nspin == 2;
#ifndef __MPI
    // The DFT+U R-space multiplication is implemented with ScaLAPACK.
    if (PARAM.inp.dft_plus_u == 2)
    {
        use_direct_hcontainer = false;
    }
#endif
    if (use_direct_hcontainer)
    {
        const auto options
            = make_hsr_output_options(HR_filename_up, HR_filename_down, SR_filename, istep, nspin, binary, sparse_thr);
        HS_Arrays.output_R_coor.clear();

        if (nspin == 1 || nspin == 2)
        {
            auto* p_ham_lcao = dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, double>*>(p_ham);
            if (TD_Velocity::tddft_velocity)
            {
                output_td_hsr(*p_ham_lcao, *td_hR, pv, HS_Arrays, options);
            }
            else
            {
                output_direct_hsr(*p_ham_lcao, pv, HS_Arrays, options);
            }
        }
        else if (nspin == 4)
        {
            auto* p_ham_lcao = dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, std::complex<double>>*>(p_ham);
            output_direct_hsr(*p_ham_lcao, pv, HS_Arrays, options);
        }
        ModuleBase::timer::tick("ModuleIO", "output_HSR");
        return;
    }

    output_legacy_hsr(ucell,
                      istep,
                      pv,
                      HS_Arrays,
                      grid,
                      kv,
                      p_ham,
#ifdef __EXX
                      Hexxd,
                      Hexxc,
#endif
                      SR_filename,
                      HR_filename_up,
                      HR_filename_down,
                      binary,
                      sparse_thr);

    ModuleBase::timer::tick("ModuleIO", "output_HSR");
    return;
}

void ModuleIO::output_dHR(const int& istep,
                          const ModuleBase::matrix& v_eff,
                          Gint_k& gint_k, // mohan add 2024-04-01
                          const UnitCell& ucell,
                          const Parallel_Orbitals& pv,
                          LCAO_HS_Arrays& HS_Arrays,
                          const Grid_Driver& grid, // mohan add 2024-04-06
                          const TwoCenterBundle& two_center_bundle,
                          const LCAO_Orbitals& orb,
                          const K_Vectors& kv,
                          const bool& binary,
                          const double& sparse_thr)
{
    ModuleBase::TITLE("ModuleIO", "output_dHR");
    ModuleBase::timer::tick("ModuleIO", "output_dHR");

    gint_k.allocate_pvdpR();

    const int nspin = PARAM.inp.nspin;

    if (nspin == 1 || nspin == 4)
    {
        // mohan add 2024-04-01
        const int cspin = 0;

        sparse_format::cal_dH(ucell, pv, HS_Arrays, grid, two_center_bundle, orb, cspin, sparse_thr, gint_k);
    }
    else if (nspin == 2)
    {
        for (int cspin = 0; cspin < 2; cspin++)
        {
            // note: some MPI process will not have grids when MPI cores are too
            // many, v_eff in these processes are empty
            const double* vr_eff1 = v_eff.nc * v_eff.nr > 0 ? &(v_eff(cspin, 0)) : nullptr;

            if (!PARAM.globalv.gamma_only_local)
            {
                if (PARAM.inp.vl_in_h)
                {
                    Gint_inout inout(vr_eff1, cspin, Gint_Tools::job_type::dvlocal);
                    gint_k.cal_gint(&inout);
                }
            }

            sparse_format::cal_dH(ucell, pv, HS_Arrays, grid, two_center_bundle, orb, cspin, sparse_thr, gint_k);
        }
    }
    // mohan update 2024-04-01
    ModuleIO::save_dH_sparse(istep, pv, HS_Arrays, sparse_thr, binary);

    sparse_format::destroy_dH_R_sparse(HS_Arrays);

    gint_k.destroy_pvdpR();

    ModuleBase::timer::tick("ModuleIO", "output_dHR");
    return;
}

void ModuleIO::output_SR(Parallel_Orbitals& pv,
                         const Grid_Driver& /*grid*/,
                         hamilt::Hamilt<std::complex<double>>* p_ham,
                         const std::string& SR_filename,
                         const bool& binary,
                         const double& sparse_thr)
{
    ModuleBase::TITLE("ModuleIO", "output_SR");
    ModuleBase::timer::tick("ModuleIO", "output_SR");

    if (PARAM.inp.nspin != 4)
    {
        auto* p_ham_lcao = dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, double>*>(p_ham);
        output_overlap_matrix(*(p_ham_lcao->getSR()), pv, SR_filename, binary, sparse_thr);
    }
    else
    {
        auto* p_ham_lcao = dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, std::complex<double>>*>(p_ham);
        output_overlap_matrix(*(p_ham_lcao->getSR()), pv, SR_filename, binary, sparse_thr);
    }

    ModuleBase::timer::tick("ModuleIO", "output_SR");
    return;
}

void ModuleIO::output_TR(const int istep,
                         const UnitCell& ucell,
                         const Parallel_Orbitals& pv,
                         LCAO_HS_Arrays& HS_Arrays,
                         const Grid_Driver& grid,
                         const TwoCenterBundle& two_center_bundle,
                         const LCAO_Orbitals& orb,
                         const std::string& TR_filename,
                         const bool& binary,
                         const double& sparse_thr)
{
    ModuleBase::TITLE("ModuleIO", "output_TR");
    ModuleBase::timer::tick("ModuleIO", "output_TR");

    const std::string output_path = make_matrix_output_path(TR_filename, istep);

    sparse_format::cal_TR(ucell, pv, HS_Arrays, grid, two_center_bundle, orb, sparse_thr);

    ModuleIO::save_sparse(HS_Arrays.TR_sparse,
                          HS_Arrays.all_R_coor,
                          sparse_thr,
                          binary,
                          output_path.c_str(),
                          pv,
                          "T",
                          istep);

    sparse_format::destroy_T_R_sparse(HS_Arrays);

    ModuleBase::timer::tick("ModuleIO", "output_TR");
    return;
}
