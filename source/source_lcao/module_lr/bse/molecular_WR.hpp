#include "molecular_lri.h"

namespace BSE
{

/*
    MO_TYPE: OO   VO    VV  
    nmo1    nocc nocc  nvirt 
    nmo2    nocc nvirt nvirt 
    imo1    0    0     nocc  
    imo2    0    nocc  nocc  
*/

/// @brief FT transform W[R_AI][R_BJ] to global matrix WA[aik1, bjk2]
template <typename T>
void MolecularWR<T>:: cal_W_global(std::vector<T>& WA_global)
{
    ModuleBase::TITLE("MolecularWR", "cal_CVC_mo_R");
    ModuleBase::timer::tick("MolecularWR", "cal_CVC_mo_R");
    // cal cvc_mo

    std::vector<TC> Rlist = this->BvK_cells;
    
    // Rlist for the first key of WR. TODO: MPI parallelize for Rlist
    // 25-12-01 NOTE: THIS function is abandoned ! cal_cvc_mo_R is calculating Vs not Ws ! Just reserve for test!
    std::map<TC, std::map<TC, RI::Tensor<T>>> WR = LR_lri.lri.cal_cvc_mo_R(CsR_oo_mo, CsR_vv_mo, Rlist);
    ModuleBase::timer::tick("MolecularWR", "cal_CVC_mo_R");

    ModuleBase::TITLE("MolecularWR", "transform_k_global");
    ModuleBase::timer::tick("MolecularWR", "transform_k_global");

    auto add_W = [&](std::vector<T>& target,
                    const std::valarray<T>& value,
                    int k1_step,
                    int k2_step,
                    const std::complex<double>& phase) -> void
    {
        const int npair = nocc * nvirt;
        for (int j = 0; j < npair; ++j)
        {
            for (int i = 0; i < npair; ++i)
            {
                const int idx_target = (k1_step + i) + (k2_step + j) * this->ndim;
                const int idx_value = i + j * npair;
                add_c(target[idx_target], value[idx_value], phase);
            }
        }
    };
    // gather all WR and FT

#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int kai = 0; kai < this->nk; ++kai)
    {
        const int k1_step = kai * this->nocc * this->nvirt;
        const ModuleBase::Vector3<double> k1 = this->kv.kvec_d.at(kai);
        for (int kbj = 0; kbj < this->nk; ++kbj)
        {
            const int k2_step = kbj * this->nocc * this->nvirt;
            const ModuleBase::Vector3<double> k2 = this->kv.kvec_d.at(kbj);
            for (const auto& w1 : WR)
            {
                const TC R_AI = w1.first;
                ModuleBase::Vector3<double> R1(R_AI[0], R_AI[1], R_AI[2]);
                const std::complex<double> phase_ai = std::exp(-ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT * (k1 * R1));
                for (const auto& w2 : w1.second)
                {
                    const TC R_BJ = w2.first;
                    ModuleBase::Vector3<double> R2(R_BJ[0], R_BJ[1], R_BJ[2]);
                    const std::complex<double> phase_bj = std::exp(ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT * (k2 * R2));
                    const std::complex<double> fac = phase_ai * phase_bj * 2.0 / static_cast<double>(this->nk); // factor 2 for Ha → Ry

                    const RI::Tensor<T>& W_RAI_RBJ = w2.second; // row-major: (j,b,i,a)
                    add_W(WA_global, *W_RAI_RBJ.data, k1_step, k2_step, fac);
                }
            }
        }
    }
    ModuleBase::timer::tick("MolecularWR", "transform_k_global");
}

/// @brief calculate CsR_mo by 
///         C'^\mu (m1,m2)[R1,R2] = c^*(m1,s)[R1] C^\mu (s,t)[R_ts] c(m2,t)[R2-R_ts]
///         C^\mu (m1,m2)[R1,R2] = C'^\mu (m1,m2)[R1,R2] + C'^*\mu (m2,m1)[R2,R1]
/// @note this method is faster than method2, but only valid for MO1 = MO2
template <typename T>
TCsR_mo<T> MolecularWR<T>::cal_CsR_mo(const UnitCell& ucell,
                                      const TLRI<T>& Cs_ao,
                                      const psi::Psi<T>& psi_ks,
                                      const int nocc,
                                      const int nvirt,
                                      const LR_Util::MO_TYPE type,
                                      const std::string type_str)
{
    ModuleBase::TITLE("MolecularWR", "cal_CsR_mo");
    using namespace RI::Array_Operator;
    int nmo1, nmo2, imo1, imo2;
    switch(type)
    {
        case LR_Util::MO_TYPE::OO:
            nmo1 = nocc; nmo2 = nocc; imo1 = 0; imo2 = 0;
            break;
        // case LR_Util::MO_TYPE::VO:
        //     nmo1 = nocc; nmo2 = nvirt; imo1 = 0; imo2 = nocc;
        //     break;
        case LR_Util::MO_TYPE::VV:
            nmo1 = nvirt; nmo2 = nvirt; imo1 = nocc; imo2 = nocc;
            break;
        default:
            throw std::runtime_error("MolecularWR::cal_CsR_mo: only support OO, VV");
    }
    const int nbands = nocc + nvirt;
    assert(psi_ks.get_nbands() == nbands);

    // c(k) -> c(R), <{iat, R}, tensor{nmo, iat.nw}>
    std::map<TAC, RI::Tensor<T>> psi1_R = slice_psi_R(psi_ks, imo1, nmo1, "psi_"+type_str+"1_R"); 
    std::map<TAC, RI::Tensor<T>> psi2_R = slice_psi_R(psi_ks, imo2, nmo2, "psi_"+type_str+"2_R");

    // <iat, <{R_1-R_mu, R_2-R_mu}, tesnor{nabf, nmo1, nmo2}>>
    TCsR_mo<T> CsR_mo_part;  // C'^\mu (m1,m2)[R1,R2] = c^*(m1,s)[R1] C^\mu (s,t)[R_ts] c(m2,t)[R2-R_ts]
    TCsR_mo<T> CsR_mo;       // C^\mu (m1,m2)[R1,R2] = C'^\mu (m1,m2)[R1,R2] + C'^*\mu (m2,m1)[R2,R1]

    for (const auto& c1 : Cs_ao)
    {
        ModuleBase::timer::tick("MolecularWR", "cal_CsR_mo_part");
        const int iat1 = c1.first;
        const int it1 = ucell.iat2it[iat1];
        const int nw1 = ucell.atoms[it1].nw;
        for (const auto& c2 : c1.second)
        {
            const int iat2 = c2.first.first;
            const int it2 = ucell.iat2it[iat2];
            const int nw2 = ucell.atoms[it2].nw;
            const TC& R_ts = c2.first.second;

            const auto& tensor_ao = c2.second;
            std::size_t nabf = tensor_ao.shape[0];
            assert(nw1 == tensor_ao.shape[1]);
            assert(nw2 == tensor_ao.shape[2]);

            for (auto R_1_mu: this->BvK_cells) // maybe MPI parallel for R_1_mu and R_2_mu
            {
                for (auto R_2_mu: this->BvK_cells)
                {
                    if (!CsR_mo_part.count(iat1) || !CsR_mo_part[iat1].count(std::make_pair(R_1_mu, R_2_mu)))
                    {
                        CsR_mo_part[iat1][std::make_pair(R_1_mu, R_2_mu)] = RI::Tensor<T>({nabf, (std::size_t)nmo1, (std::size_t)nmo2});
                        CsR_mo[iat1][std::make_pair(R_1_mu, R_2_mu)] = RI::Tensor<T>({nabf, (std::size_t)nmo1, (std::size_t)nmo2});
                    }
                    TC R_psi1 = R_1_mu % this->period;
                    TC R_psi2 = (R_2_mu - R_ts) % this->period;
                    for (int iabf = 0; iabf < nabf; ++iabf)
                    {
                        const auto ptr = &tensor_ao(iabf, 0, 0);
                        std::vector<T> tmp(nmo1 * nw2);
                        // caution: Cs are row-major  (iw2 contiguous)
                        // C'(m1,m2) = c^*(m1,s) C(s,t) c(m2,t)         << row-major
                        // tmp_m1_t = (c_s_m1)^H (C_t_s)^T              << col-major
                        container::BlasConnector::gemm('C', 'T', nmo1, nw2, nw1,
                                                        1.0, psi1_R[std::make_pair(iat1, R_psi1)].ptr(), nw1,
                                                        ptr, nw2,
                                                        0.0, tmp.data(), nmo1);
                        // C'_m2_m1 = (c_t_m2)^T (tmp_m1_t)^T           << col-major
                        container::BlasConnector::gemm('T', 'T', nmo2, nmo1, nw2,
                                                        1.0, psi2_R[std::make_pair(iat2, R_psi2)].ptr(), nw2,
                                                        tmp.data(), nmo1,                                                        
                                                        1.0, &CsR_mo_part[iat1][std::make_pair(R_1_mu, R_2_mu)](iabf, 0, 0), nmo2);
                    }
                }
            }
        }
        ModuleBase::timer::tick("MolecularWR", "cal_CsR_mo_part");

        // C'^\mu (m1,m2)[R1,R2] is finished, now calculate C^\mu (m1,m2)[R1,R2]
        ModuleBase::timer::tick("MolecularWR", "cal_CsR_mo_add");
        assert(nmo1 == nmo2);  // ATTENTION: this method only support MO1 = MO2 case

        const auto& Cs_part_iat1 = CsR_mo_part.at(iat1);
        auto& Cs_iat1 = CsR_mo.at(iat1);

        const std::size_t nbvk = this->BvK_cells.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
        for (std::size_t i = 0; i < nbvk; ++i)
        {
            for (std::size_t j = 0; j < nbvk; ++j)
            {
                const auto& R_1_mu = this->BvK_cells[i];
                const auto& R_2_mu = this->BvK_cells[j];
                RI::Tensor<T> t = Cs_part_iat1.at(std::make_pair(R_1_mu, R_2_mu)).copy();
                const RI::Tensor<T>& t2 = Cs_part_iat1.at(std::make_pair(R_2_mu, R_1_mu));
                std::size_t nabf = t.shape[0];
                for (std::size_t iabf = 0; iabf < nabf; ++iabf)
                {
                    T* out_ptr = &t(iabf, 0, 0);
                    const T* in_ptr = &t2(iabf, 0, 0);
                    const size_t nsize = nmo1 * nmo2;
#ifdef _OPENMP
#pragma omp simd
#endif
                    for (std::size_t idx = 0; idx < nsize; ++idx)
                    {
                        const std::size_t m1 = idx / nmo2;
                        const std::size_t m2 = idx - m1 * nmo2;
                        out_ptr[idx] += LR_Util::get_conj(in_ptr[m2 * nmo1 + m1]);
                    }
                }
                Cs_iat1.at(std::make_pair(R_1_mu, R_2_mu)) = std::move(t);
            }
        }
        ModuleBase::timer::tick("MolecularWR", "cal_CsR_mo_add");
    }

    // print C_mo max
    this->print_CsR_mo_max(CsR_mo, "CsR_mo_" + type_str);
    this->print_CsR_mo_max(CsR_mo_part, "CsR_mo_part_" + type_str);
    return CsR_mo;
}

/// @brief calculate CsR_mo by C^\mu (m1,m2)[R1,R2] 
///         = c^*(m1,s)[R1] C^\mu (s,t)[R_ts] c(m2,t)[R2-R_ts]
///          +c(m2,s)[R2] C^\mu (s,t)[R_ts] c^*(m1,t)[R1-R_ts]
/// @note this method can calculate different MO1 and MO2, but slower than method1
template <typename T>
TCsR_mo<T> MolecularWR<T>::cal_CsR_mo_method2(const UnitCell& ucell,
                                              const TLRI<T>& Cs_ao,
                                              const psi::Psi<T>& psi_ks,
                                              const int nocc,
                                              const int nvirt,
                                              const LR_Util::MO_TYPE type,
                                              const std::string type_str)
{
    ModuleBase::TITLE("MolecularWR", "cal_CsR_mo_method2");
    ModuleBase::timer::tick("MolecularWR", "cal_CsR_mo_method2");
    using namespace RI::Array_Operator;
    int nmo1, nmo2, imo1, imo2;
    switch(type)
    {
    case LR_Util::MO_TYPE::OO:
        nmo1 = nocc; nmo2 = nocc; imo1 = 0; imo2 = 0;
        break;
    case LR_Util::MO_TYPE::VO:
        nmo1 = nocc; nmo2 = nvirt; imo1 = 0; imo2 = nocc;
        break;
    case LR_Util::MO_TYPE::VV:
        nmo1 = nvirt; nmo2 = nvirt; imo1 = nocc; imo2 = nocc;
        break;
    default:
        throw std::runtime_error("MolecularWR::cal_CsR_mo: only support OO, VV, VO");
    }
    const int nbands = nocc + nvirt;
    assert(psi_ks.get_nbands() == nbands);

    // c(k) -> c(R), <{iat, R}, tensor{nmo, iat.nw}>
    std::map<TAC, RI::Tensor<T>> psi1_R = slice_psi_R(psi_ks, imo1, nmo1, "psi_"+type_str+"1_R"); 
    std::map<TAC, RI::Tensor<T>> psi2_R = slice_psi_R(psi_ks, imo2, nmo2, "psi_"+type_str+"2_R");

    // <iat, <{R_1-R_mu, R_2-R_mu}, tesnor{nabf, nmo1, nmo2}>>
    TCsR_mo<T> CsR_mo;       // C^\mu (m1,m2)[R1,R2] = c^*(m1,s)[R1] C^\mu (s,t)[R_ts] c(m2,t)[R2-R_ts]
                             //                       +c(m2,s)[R2] C^\mu (s,t)[R_ts] c^*(m1,t)[R1-R_ts]

    for (const auto& c1 : Cs_ao)
    {
        const int iat1 = c1.first;
        const int it1 = ucell.iat2it[iat1];
        const int nw1 = ucell.atoms[it1].nw;
        for (const auto& c2 : c1.second)
        {
            const int iat2 = c2.first.first;
            const int it2 = ucell.iat2it[iat2];
            const int nw2 = ucell.atoms[it2].nw;
            const TC& R_ts = c2.first.second;

            const auto& tensor_ao = c2.second;
            const size_t nabf = tensor_ao.shape[0];
            assert(nw1 == tensor_ao.shape[1]);
            assert(nw2 == tensor_ao.shape[2]);
            for (auto R_1_mu: this->BvK_cells) // maybe MPI parallel for R_1_mu and R_2_mu
            {
                for (auto R_2_mu: this->BvK_cells)
                {
                    if (!CsR_mo.count(iat1) || !CsR_mo[iat1].count(std::make_pair(R_1_mu, R_2_mu)))
                    {
                        CsR_mo[iat1][std::make_pair(R_1_mu, R_2_mu)] = RI::Tensor<T>({nabf, (std::size_t)nmo1, (std::size_t)nmo2});
                    }
                    for (int iabf = 0; iabf < nabf; ++iabf)
                    {
                        const auto ptr = &tensor_ao(iabf, 0, 0);
                        const auto ptr_dest = &CsR_mo[iat1][std::make_pair(R_1_mu, R_2_mu)](iabf, 0, 0);
                        // term 1
                        std::vector<T> tmp(nmo1 * nw2);
                        TC R_psi1 = R_1_mu % this->period;
                        TC R_psi2 = (R_2_mu - R_ts) % this->period;
                        // caution: Cs are row-major  (iw2 contiguous)
                        // C(m1,m2) = c^*(m1,s) C(s,t) c(m2,t)         << row-major
                        // tmp_m1_t = (c_s_m1)^H (C_t_s)^T              << col-major
                        container::BlasConnector::gemm('C', 'T', nmo1, nw2, nw1,
                                                        1.0, psi1_R[std::make_pair(iat1, R_psi1)].ptr(), nw1,
                                                        ptr, nw2,
                                                        0.0, tmp.data(), nmo1);
                        // C_m2_m1 = (c_t_m2)^T (tmp_m1_t)^T           << col-major
                        container::BlasConnector::gemm('T', 'T', nmo2, nmo1, nw2,
                                                        1.0, psi2_R[std::make_pair(iat2, R_psi2)].ptr(), nw2,
                                                        tmp.data(), nmo1,                                                        
                                                        1.0, ptr_dest, nmo2);

                        // term 2
                        tmp.resize(nmo1 * nw1); 
                        R_psi1 = (R_1_mu - R_ts) % this->period;
                        R_psi2 = R_2_mu % this->period;
                        // C(m1,m2) = c^*(m1,t) C(s,t) c(m2,s)         << row-major
                        // tmp_m1_s = (c_t_m1)^H (C_t_s)               << col-major
                        container::BlasConnector::gemm('C', 'N', nmo1, nw1, nw2,
                                                        1.0, psi1_R[std::make_pair(iat2, R_psi1)].ptr(), nw2,
                                                        ptr, nw2,
                                                        0.0, tmp.data(), nmo1);
                        // C_m2_m1 = (c_s_m2)^T (tmp_m1_s)^T           << col-major
                        container::BlasConnector::gemm('T', 'T', nmo2, nmo1, nw1,
                                                        1.0, psi2_R[std::make_pair(iat1, R_psi2)].ptr(), nw1,
                                                        tmp.data(), nmo1,                                                        
                                                        1.0, ptr_dest, nmo2); 
                    }
                }
            }
        }
    }

    // print C_mo max
    this->print_CsR_mo_max(CsR_mo, "CsR_mo_method2_" + type_str);
    ModuleBase::timer::tick("MolecularWR", "cal_CsR_mo_method2");
    return CsR_mo;
}

template <typename T>
void MolecularWR<T>::print_CsR_mo_max(const TCsR_mo<T>& CsR_mo, const std::string file_name)
{
    ModuleBase::TITLE("MolecularWR", "print_CsR_mo_max");
    ModuleBase::timer::tick("MolecularWR", "print_CsR_mo_max");
    std::ofstream ofs_out(file_name + "_" + std::to_string(GlobalV::MY_RANK));
    for (auto& c1: CsR_mo)
    {
        int iat = c1.first;
        ofs_out << "Cs_mo: iat " << iat << std::endl;
        for (auto& c2: c1.second)
        {
            ModuleBase::Vector3<double> R_1_mu = RI_Util::array3_to_Vector3(c2.first.first) * this->ucell.latvec;
            double R_1_mu_norm = R_1_mu.norm();
            ofs_out << " R_1_mu:" << R_1_mu;

            ModuleBase::Vector3<double> R_2_mu = RI_Util::array3_to_Vector3(c2.first.second) * this->ucell.latvec;
            double R_2_mu_norm = R_2_mu.norm();
            ofs_out << " R_2_mu:" << R_2_mu << std::endl;
            auto& tensor_mo = c2.second;

            double max = tensor_mo.max_abs();
            ofs_out <<"Mu_atom: " << iat << " ,|R_1_mu|: " << R_1_mu_norm << " ,|R_2_mu|: " << R_2_mu_norm
                    << " ,max: " << max << std::endl;

            // print Cs_mo
            int nabf = tensor_mo.shape[0];
            int nmo1 = tensor_mo.shape[1];
            int nmo2 = tensor_mo.shape[2];
            int size = nabf*nmo1*nmo2;
            assert(size == LR_Util::write_value(ofs_out, tensor_mo.ptr(), nabf, nmo1, nmo2));
            // print Cs_mo
        }
    }
    ofs_out.close();
    ModuleBase::timer::tick("MolecularWR", "print_CsR_mo_max");
}


/// @brief slice psi as <{iat, R}, tensor{nmo, iat.nw}>
template <typename T>
std::map<TAC, RI::Tensor<T>>
MolecularWR<T>::slice_psi_R(const psi::Psi<T>& psi_ks,
                            const int imo,
                            const int nmo,
                            const std::string file_name)
{
    ModuleBase::TITLE("MolecularWR", "slice_psi_R");
    ModuleBase::timer::tick("MolecularWR", "slice_psi_R");
    std::map<TAC, RI::Tensor<T>> psi_R;
    for (auto cell: this->BvK_cells)
    {
        // allocate
        for (int iat = 0; iat < this->ucell.nat; ++iat)
        {
            const int it = this->ucell.iat2it[iat];
            const int nw = this->ucell.atoms[it].nw;
            psi_R[std::make_pair(iat, cell)] = RI::Tensor<T>({(std::size_t)nmo, (std::size_t)nw});
        }

        ModuleBase::Vector3<double> R(cell[0], cell[1], cell[2]);
        for (int ik = 0; ik < this->nk; ++ik)
        {
            std::complex<double> phase
                = std::exp(-ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT * (this->kv.kvec_d.at(ik) * R))
                  / static_cast<double>(this->nk);
            for (int iat = 0; iat < this->ucell.nat; ++iat)
            {
                const int it = this->ucell.iat2it[iat];
                const int nw = this->ucell.atoms[it].nw;
                auto& t = psi_R.at(std::make_pair(iat, cell));
                for (int im = 0; im < nmo; ++im)
                {
                    for (int iw = 0; iw < nw; ++iw)
                    {
                        add_c(t(im, iw), this->psi_ks(ik, im + imo, this->ucell.get_iat2iwt()[iat]+iw), phase);
                    }
                }
            }
        }
    }
    // // print psi_R
    std::ofstream ofs_out(file_name + "_" + std::to_string(GlobalV::MY_RANK));
    for (auto cell: this->BvK_cells)
    {
        ModuleBase::Vector3<double> R = RI_Util::array3_to_Vector3(cell) * this->ucell.latvec;
        double R_norm = (R).norm();
        
        // print psi_R tensor
        for (int iat = 0; iat < this->ucell.nat; ++iat)
        {
            auto& t = psi_R.at(std::make_pair(iat, cell));
            ofs_out << "psi_R: iat " << iat << " ,R:" << R << std::endl;
            const int nw = t.shape[1];
            assert(nmo*nw == LR_Util::write_value(ofs_out, t.ptr(), nmo, nw));
        }

        // print psi_R max for each mo, atom, R
        for (int im = 0; im < nmo; ++im)
        {
            for (int iat = 0; iat < this->ucell.nat; ++iat)
            {
                double max = -1.0;
                auto& t = psi_R.at(std::make_pair(iat, cell));
                const int nw = t.shape[1];
                for (size_t iw = 0; iw < nw; ++iw)
                {
                    if (std::abs(t(im, iw)) > max)
                    {
                        max = std::abs(t(im, iw));
                    }
                }
                ofs_out << "|c(R)|::IM: " << im << " ,IAT: " << iat 
                        << " ,R: " << R_norm << " ,max: " << max << std::endl;
            }
        }
    }
    ofs_out.close();
    ModuleBase::timer::tick("MolecularWR", "slice_psi_R");
    return psi_R;
}

}// namespace BSE