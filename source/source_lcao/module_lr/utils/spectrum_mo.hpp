#pragma once
#include "source_base/tool_title.h"
#include "source_basis/module_nao/two_center_bundle.h"
#include "source_cell/klist.h"
#include "source_io/module_parameter/parameter.h"
#include "source_lcao/module_hcontainer/hcontainer_funcs.h"
#include "source_lcao/module_lr/ao_to_mo_transformer/ao_to_mo.h"
#include "source_lcao/module_lr/utils/lr_util.h"
#include "source_lcao/module_lr/utils/lr_util_print.h"
#include "source_lcao/module_rt/velocity_op.h"
#include "source_lcao/module_lr/utils/lr_util_rR_reader.h"
#include "source_psi/psi.h"

namespace LR_Util
{
template<typename T>
std::vector<std::complex<double>> cal_velocity_mo(const UnitCell& ucell,
                                                const Grid_Driver& gd,
                                                const TwoCenterBundle& two_center_bundle,
                                                const Parallel_Orbitals& pmat,/*nbasis×nbasis*/
                                                const Parallel_2D& pc,/*nbasis×nbands*/
                                                const K_Vectors& kv,
                                                const psi::Psi<T>& psi_ks,
                                                const int nk,
                                                const int nspin_tmp,
                                                const int nbasis, 
                                                const std::vector<int> nocc, 
                                                const std::vector<int> nvirt)
{
    ModuleBase::TITLE("LR::LR_Util", "cal_velocity_mo");
    std::cout<<"Calculating velocity matrix in KS presentation..."<<std::endl;
    // get_velocity_matrix_R(ucell, gd_, pmat, two_center_bundle_);
    LCAO_Orbitals orb;
    const auto& inp = PARAM.inp;
    two_center_bundle.to_LCAO_Orbitals(orb, inp.lcao_ecut, inp.lcao_dk, inp.lcao_dr, inp.lcao_rmax);

    Velocity_op<std::complex<double>> vR(&ucell, &gd, &pmat, orb, two_center_bundle.overlap_orb.get());
    vR.calculate_grad_term();   // $<\mu, 0|-i∇r|\nu, R>$
    vR.calculate_vcomm_r(); // $<\mu, 0|i[Vnl, r]|\nu, R>$

    int nks = kv.get_nks(); // include spin
    assert(nks == nk * nspin_tmp);
    assert(psi_ks.get_nk() == nks);
    int KS_num = nocc[0] + nvirt[0];
    std::vector<std::complex<double>> velocity_mo(nspin_tmp * 3 * nk * KS_num * KS_num, 0.0);
    Parallel_2D pmo;
    LR_Util::setup_2d_division(pmo, pmat.get_block_size(), KS_num, KS_num
#ifdef __MPI
        , pc.blacs_ctxt
#endif
        );

    //1. psi_ks<T> to c_psi_ks<complex<double>>
    psi::Psi<std::complex<double>> c_psi_ks(nks,
                                            pc.get_col_size(), 
                                            pc.get_row_size(), 
                                            kv.ngk, 
                                            true);
    GlobalV::ofs_running << "new c_psi_ks" << std::endl;
    GlobalV::ofs_running << "psi_ks.get_nk(): " << psi_ks.get_nk() << " psi_ks.get_nbands(): " << psi_ks.get_nbands() << " psi_ks.get_nbasis(): " << psi_ks.get_nbasis() << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int iks = 0; iks < nks; ++iks)
    {
        for(int ic = 0; ic < pc.get_col_size(); ++ic)//band
        {
            for(int ir = 0; ir < pc.get_row_size(); ++ir)//basis
            {
                c_psi_ks(iks, ic, ir) = std::complex<double>(psi_ks(iks, ic, ir));
            }
        }
    }
    GlobalV::ofs_running<< "end c_psi_ks" << std::endl;
    
    //2. calculate v_mo = c^\dagger v c
    std::vector<ct::Tensor> vk(nks, LR_Util::newTensor<std::complex<double>>({ pmat.get_col_size(), pmat.get_row_size() }));
    for (int id = 0; id < 3; ++id)
    {
        for (auto& v : vk) v.zero();

        std::vector<std::complex<double>> v_mo(nks * pmo.get_local_size(), 0.0);

        for (int is = 0; is < nspin_tmp; ++is)
        {
            assert(KS_num == nocc[is] + nvirt[is]);

            for (int ik = is*nk; ik < nk; ++ik)
            {            
                hamilt::folding_HR(*vR.get_current_term_pointer(id), vk[ik].data<std::complex<double>>(), kv.kvec_d[ik], pmat.get_row_size(), 1/*column-major*/);
            }
        }
#ifdef __MPI
        ao_to_mo_pblas(vk, pmat, c_psi_ks, pc, nbasis,
                        nocc[0], nvirt[0], pmo, v_mo.data(),
                        false, // add_on
                        LR::MO_TYPE::ALL);
#else
        ao_to_mo_blas(vk, c_psi_ks, 
                        nocc[0], nvirt[0], v_mo.data(),
                        false , //add_on
                        LR::MO_TYPE::ALL);
#endif
        // gather local vk to global velocity_mo
        for (int is = 0; is < nspin_tmp; ++is)
        {
            for (int ik = 0; ik < nk; ++ik)
            {
                LR_Util::gather_2d_to_full(pmo, v_mo.data() + ik * pmo.get_local_size(), 
                &velocity_mo[(is * 3 * nk + id * nk + ik) * KS_num * KS_num ],
                false/*col_first*/, KS_num, KS_num);
                
                //std::cout<< "is" << is << "id: " << id << " ik: " << ik << " v_mo: " << std::endl;            
                //LR_Util::print_value(velocity_mo.data()+(is * 3 * nk + id * nk + ik) * KS_num * KS_num, KS_num, KS_num);
            }
        }
    }//id
    return velocity_mo;
}

template<typename T>
std::vector<std::complex<double>> cal_dipole_r_mo(const UnitCell& ucell,
                                                const Parallel_Orbitals& pmat,/*nbasis×nbasis*/
                                                const Parallel_2D& pc,/*nbasis×nbands*/
                                                const K_Vectors& kv,
                                                const psi::Psi<T>& psi_ks,
                                                const int nk,
                                                const int nspin_tmp,
                                                const int nbasis, 
                                                const std::vector<int> nocc, 
                                                const std::vector<int> nvirt)
{
    ModuleBase::TITLE("LR::LR_Util", "cal_dipole_r_mo");
    std::cout<<"Calculating r-dipole matrix in KS presentation..."<<std::endl;
    LR_Util::rRFileReader rRReader (PARAM.globalv.global_readin_dir + "rr.csr", pmat, ucell.nat);
    rRReader.convert_rR_HContainer();

    int nks = kv.get_nks(); // include spin
    assert(nks == nk * nspin_tmp);
    assert(psi_ks.get_nk() == nks);
    int KS_num = nocc[0] + nvirt[0];
    std::vector<std::complex<double>> dipole_mo(nspin_tmp * 3 * nk * KS_num * KS_num, 0.0);
    Parallel_2D pmo;
    LR_Util::setup_2d_division(pmo, pmat.get_block_size(), KS_num, KS_num
#ifdef __MPI
        , pc.blacs_ctxt
#endif
        );

    //1. psi_ks<T> to c_psi_ks<complex<double>>
    psi::Psi<std::complex<double>> c_psi_ks(nks,
                                            pc.get_col_size(), 
                                            pc.get_row_size(), 
                                            kv.ngk, 
                                            true);
    GlobalV::ofs_running << "new c_psi_ks" << std::endl;
    GlobalV::ofs_running << "psi_ks.get_nk(): " << psi_ks.get_nk() << " psi_ks.get_nbands(): " << psi_ks.get_nbands() << " psi_ks.get_nbasis(): " << psi_ks.get_nbasis() << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int iks = 0; iks < nks; ++iks)
    {
        for(int ic = 0; ic < pc.get_col_size(); ++ic)//band
        {
            for(int ir = 0; ir < pc.get_row_size(); ++ir)//basis
            {
                c_psi_ks(iks, ic, ir) = std::complex<double>(psi_ks(iks, ic, ir));
            }
        }
    }
    GlobalV::ofs_running<< "end c_psi_ks" << std::endl;
    
    //2. calculate r_mo = c^\dagger r c
    std::vector<ct::Tensor> rk(nks, LR_Util::newTensor<std::complex<double>>({ pmat.get_col_size(), pmat.get_row_size() }));
    for (int id = 0; id < 3; ++id)
    {
        for (auto& r : rk) r.zero();

        std::vector<std::complex<double>> r_mo(nks * pmo.get_local_size(), 0.0);

        for (int is = 0; is < nspin_tmp; ++is)
        {
            assert(KS_num == nocc[is] + nvirt[is]);

            for (int ik = is*nk; ik < nk; ++ik)
            {            
                hamilt::folding_HR(rRReader.rR[id], rk[ik].data<std::complex<double>>(), kv.kvec_d[ik], pmat.get_row_size(), 1/*column-major*/);
            }
        }
#ifdef __MPI
        ao_to_mo_pblas(rk, pmat, c_psi_ks, pc, nbasis,
                        nocc[0], nvirt[0], pmo, r_mo.data(),
                        false, // add_on
                        LR::MO_TYPE::ALL);
#else
        ao_to_mo_blas(rk, c_psi_ks, 
                        nocc[0], nvirt[0], r_mo.data(),
                        false , //add_on
                        LR::MO_TYPE::ALL);
#endif
        // gather local rk to global r_mo
        for (int is = 0; is < nspin_tmp; ++is)
        {
            for (int ik = 0; ik < nk; ++ik)
            {
                LR_Util::gather_2d_to_full(pmo, r_mo.data() + ik * pmo.get_local_size(), 
                &dipole_mo[(is * 3 * nk + id * nk + ik) * KS_num * KS_num ],
                false/*col_first*/, KS_num, KS_num);
                
                //std::cout<< "is" << is << "id: " << id << " ik: " << ik << " v_mo: " << std::endl;            
                //LR_Util::print_value(dipole_mo.data()+(is * 3 * nk + id * nk + ik) * KS_num * KS_num, KS_num, KS_num);
            }
        }
    }//id
    return dipole_mo;
}

/// @brief output the velocity matrix in KS presentation in dipole format
inline void output_spectrum_mo(const std::vector<std::complex<double>>& out_spectrum_mo,
                        const std::string& filename,
                        const double* const eig_ks,
                        const int nk,
                        const int nspin_tmp,
                        const int KS_num,
                        const K_Vectors& kv)
{
    assert(out_spectrum_mo.size() == nspin_tmp * 3 * nk * KS_num * KS_num);
    std::ofstream ofs(PARAM.globalv.global_out_dir + filename +".dat");
    ofs << "Data Unit (Hartree * Bohr)    " << filename << std::endl;
    ofs << "NOTICE: KS_index are restricted in nocc and nvirt" << std::endl;
    for (int is = 0; is < nspin_tmp; ++is)
    {
        ofs << "ispin: " << is << std::endl;
        for (int ik = 0;ik < nk;++ik)
        {
            ofs << "k-point: " << ik << " " << kv.kvec_d[ik] << std::endl;
            ofs << std::setw(4) << "KS1" << std::setw(10) << "E1(eV)" << std::setw(6) << "KS2" << std::setw(10) << "E2(eV)"
                << std::setw(16) << "x" << std::setw(23) << "|x|^2" << std::setw(18) << "y" << std::setw(23) <<"|y|^2"
                << std::setw(18) << "z" << std::setw(23) <<"|z|^2" << std::setw(13) << "average" << std::endl;
    
            for (int i = 0; i < KS_num; ++i)
            {
                for (int j = i; j < KS_num ; ++j)
                {
                    int ipair = (is * 3 * nk + ik) * KS_num * KS_num + i * KS_num + j;
                    int step = nk * KS_num * KS_num;
                    double average = (std::norm(out_spectrum_mo[ipair]) + std::norm(out_spectrum_mo[ipair + step]) + std::norm(out_spectrum_mo[ipair + 2 * step])) / 3.0;
                    ofs << std::setw(3) << i << std::setw(12) << std::setprecision(6) << eig_ks[i + ik*KS_num] * ModuleBase::Ry_to_eV
                    << std::setw(4) << j << std::setw(12) << eig_ks[j + ik*KS_num] * ModuleBase::Ry_to_eV
                    << std::setw(28) << out_spectrum_mo[ipair] << std::setw(13) << std::norm(out_spectrum_mo[ipair])
                    << std::setw(28) << out_spectrum_mo[ipair + step] << std::setw(13) << std::norm(out_spectrum_mo[ipair + step])
                    << std::setw(28) << out_spectrum_mo[ipair + 2 * step]  << std::setw(13) << std::norm(out_spectrum_mo[ipair + 2 * step] )
                    << std::setw(13) << average << std::endl; 
                }
            }
        }
    }
    ofs.close();
}

/// @brief output the velocity matrix in KS presentation in LibRPA format
inline void output_spectrum_mo_librpa(const std::vector<std::complex<double>>& out_spectrum_mo,
                        const std::string& filename,
                        const int nk,
                        const int nspin_tmp,
                        const int KS_num,
                        const K_Vectors& kv)
{
    assert(out_spectrum_mo.size() == nspin_tmp * 3 * nk * KS_num * KS_num);
    std::ofstream ofs(filename);
    ofs << nk << std::endl;
    ofs << nspin_tmp << std::endl;
    ofs << PARAM.inp.nbands << std::endl;
    ofs << PARAM.globalv.nlocal << std::endl;
    double HaBohrToEvAng = ModuleBase::Hartree_to_eV * ModuleBase::BOHR_TO_A; // 27.211396 * 0.5291770
    for (int is = 0; is < nspin_tmp; ++is)
    {
        for (int ik = 0; ik < nk; ++ik)
        {
            for (int id = 0; id < 3; ++id)
            {
                ofs <<"  " << id + 1 << "  " << ik + 1 << "  " << is + 1 << std::endl;
                for (int ib1 = 0; ib1 < KS_num; ++ib1)
                {
                    for (int ib2 = 0; ib2 < KS_num; ++ib2)
                    {
                        int ipair = (is * 3 * nk + id * nk + ik) * KS_num * KS_num + ib1 * KS_num + ib2;
                        ofs << std::setw(26) << std::fixed << std::setprecision(16) << out_spectrum_mo[ipair].real() * HaBohrToEvAng
                            << std::setw(26) << out_spectrum_mo[ipair].imag() * HaBohrToEvAng << std::endl;
                    }
                }
            }

        }
    }
}
}