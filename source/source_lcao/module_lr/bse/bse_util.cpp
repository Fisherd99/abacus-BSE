#include "bse_util.h"
#include "source_lcao/module_lr/utils/lr_util.hpp"

namespace BSE_Util
{
std::vector<std::complex<double>>
to_complex(const std::vector<double>& vin)
{
    std::vector<std::complex<double>> vout;
    vout.resize(vin.size());
    std::transform(vin.begin(), vin.end(), vout.begin(),
                   [](double x){ return std::complex<double>(x, 0.0); });
    return vout;
}

double inner_product(const double* vec1, const double* vec2, const int& size)
{
    double result = 0.0;
    for (int i = 0; i < size; ++i)
    {
        result += vec1[i] * vec2[i];
    }
    return result;
}

std::complex<double> inner_product(const std::complex<double>* vec1,
                                   const std::complex<double>* vec2,
                                   const int& size)
{
    std::complex<double> result = 0.0;
    for (int i = 0; i < size; ++i)
    {
        result += std::conj(vec1[i]) * vec2[i];
    }
    return result;
}


#ifdef __MPI
template <>
container::Tensor cal_dm_trans_onebase_pblas(
    const psi::Psi<double>& c,
    const Parallel_2D& pc,
    const int& ik,
    const int& naos,
    const int& imo1, // imo1 → imo2, for excitation imo1[0,nocc), imo2[nocc,nocc+nvirt)
    const int& imo2,
    const Parallel_Orbitals& pmat,
    const double& factor)
{
    ModuleBase::TITLE("BSE_Util", "cal_dm_trans_onebase_pblas(double)");
    assert(pc.comm() == pmat.comm());
    assert(pc.blacs_ctxt == pmat.blacs_ctxt);
    assert(pmat.get_local_size() > 0);
    assert(pmat.get_global_row_size() == naos);
    assert(pmat.get_global_col_size() == naos);
    c.fix_k(ik);
    const int one = 1;

    container::Tensor dm_trans(DAT::DT_DOUBLE, DEV::CpuDevice,
        { pmat.get_col_size(), pmat.get_row_size() }); // row is "inside"(memory contiguity) for pblas

    char transa = 'N', transb = 'T';
    const double beta = 0;
    int imo1_ = imo1 + 1; // fortran index
    int imo2_ = imo2 + 1;
    // for excitation => [C_virt * C_occ^T]^T = C_occ * C_virt^T  
    // (lhs for row-major Tensor, rhs for column-major pblas, occ_aos is contiguous)

    pdgemm_(&transa, &transb, &naos, &naos, &one,
        &factor, c.get_pointer(), &one, &imo1_, pc.desc,
        c.get_pointer(), &one, &imo2_, pc.desc,
        &beta, dm_trans.data<double>(), &one, &one, pmat.desc);

    return dm_trans;
}

template <>
container::Tensor cal_dm_trans_onebase_pblas(
    const psi::Psi<std::complex<double>>& c,
    const Parallel_2D& pc,
    const int& ik,
    const int& naos,
    const int& imo1, // imo1 → imo2, for excitation imo1[0,nocc), imo2[nocc,nocc+nvirt)
    const int& imo2,
    const Parallel_Orbitals& pmat,
    const std::complex<double>& factor)
{
    ModuleBase::TITLE("BSE_Util", "cal_dm_trans_onebase_pblas(complex)");
    assert(pc.comm() == pmat.comm());
    assert(pc.blacs_ctxt == pmat.blacs_ctxt);
    assert(pmat.get_local_size() > 0);
    assert(pmat.get_global_row_size() == naos);
    assert(pmat.get_global_col_size() == naos);
    c.fix_k(ik);
    const int one = 1;

    container::Tensor dm_trans(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice,
        { pmat.get_col_size(), pmat.get_row_size() }); // row is "inside"(memory contiguity) for pblas
    std::vector<std::complex<double>> dm_trans_tmp(pmat.get_local_size());
    char transa = 'N', transb = 'C';
    const std::complex<double> alpha(1.0, 0.0);
    const std::complex<double> beta(0.0, 0.0);
    int imo1_ = imo1 + 1; // fortran index
    int imo2_ = imo2 + 1;
    // for excitation => [C_virt * C_occ^\dagger]^T = C_occ^* * C_virt^T
    // (lhs for row-major Tensor, rhs for column-major pblas, occ_aos is contiguous)
    
    // since psi::get_pointer(ib) is only for global wfc, we implement the conj through pzgemm 
    // as a comparasion, see cal_dm_trans_onebase_blas(complex)
    pzgemm_(&transa, &transb, &naos, &naos, &one,
            &factor, c.get_pointer(), &one, &imo2_, pc.desc,
            c.get_pointer(), &one, &imo1_, pc.desc,
            &beta, dm_trans_tmp.data(), &one, &one, pmat.desc);

    pztranu_(&naos, &naos, &alpha,
            dm_trans_tmp.data(), &one, &one, pmat.desc,
            &beta,
            dm_trans.data<std::complex<double>>(), &one, &one, pmat.desc);

    return dm_trans;
}
#endif

template <>
container::Tensor cal_dm_trans_onebase_blas(
    const psi::Psi<double>& c,
    const int& ik,
    const int& naos,
    const int& imo1, // imo1 → imo2, for excitation imo1[0,nocc), imo2[nocc,nocc+nvirt)
    const int& imo2,
    const double& factor)
{
    ModuleBase::TITLE("BSE_Util", "cal_dm_trans_onebase_blas(double)");
    c.fix_k(ik);
    const int one = 1;

    container::Tensor dm_trans(DAT::DT_DOUBLE, DEV::CpuDevice, { naos, naos });    

    char transa = 'N', transb = 'T';
    const double beta = 0;
    // for excitation => [C_virt * C_occ^T]^T = C_occ * C_virt^T  
    // (lhs for row-major Tensor, rhs for column-major pblas, occ_aos is contiguous)

    dgemm_(&transa, &transb, &naos, &naos, &one,
        &factor, c.get_pointer(imo1), &naos,
        c.get_pointer(imo2), &naos,
        &beta, dm_trans.data<double>(), &naos);

    return dm_trans;
}

template <>
container::Tensor cal_dm_trans_onebase_blas(
    const psi::Psi<std::complex<double>>& c,
    const int& ik,
    const int& naos,
    const int& imo1, // imo1 → imo2, for excitation imo1[0,nocc), imo2[nocc,nocc+nvirt)
    const int& imo2,
    const std::complex<double>& factor)
{
    ModuleBase::TITLE("BSE_Util", "cal_dm_trans_onebase_blas(complex)");
    c.fix_k(ik);
    const int one = 1;

    container::Tensor dm_trans(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, { naos, naos });    

    char transa = 'N', transb = 'T';
    const std::complex<double> beta = 0;
    // for excitation => [C_virt * C_occ^\dagger]^T = C_occ^* * C_virt^T  
    // (lhs for row-major Tensor, rhs for column-major pblas, occ_aos is contiguous)

    std::vector<std::complex<double>> c_imo1_conj(naos);
    for (int ibasis = 0; ibasis < naos; ++ibasis) {
        c_imo1_conj[ibasis] = LR_Util::get_conj(c(imo1, ibasis));
    }
    zgemm_(&transa, &transb, &naos, &naos, &one,
        &factor, c_imo1_conj.data(), &naos,
        c.get_pointer(imo2), &naos,
        &beta, dm_trans.data<std::complex<double>>(), &naos);

    return dm_trans;
}
}