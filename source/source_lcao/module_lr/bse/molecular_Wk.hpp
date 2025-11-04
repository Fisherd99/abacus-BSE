#include "molecular_W.h"

namespace BSE
{

/*
LR::MO_TYPE: OO   VO    VV  
    nmo1    nocc nocc  nvirt 
    nmo2    nocc nvirt nvirt 
    imo1    0    0     nocc  
    imo2    0    nocc  nocc  
*/

/// @brief W[k_AI][k_BJ] to global matrix WA[aik1, bjk2]
template <typename T>
void MolecularWk<T>:: cal_W_global(std::vector<T>& WA_global)
{
    ModuleBase::TITLE("MolecularWk", "cal_CVC_mo_k");
    ModuleBase::timer::tick("MolecularWk", "cal_CVC_mo_k");
    // cal cvc_mo
    auto lri = this->exx_lri.lock();

    std::map<Tk, std::map<Tk, RI::Tensor<T>>>
        Wk = lri->exx_lri.lri.cal_cvc_mo_k(Csk_oo_mo, Csk_vv_mo, k1_list, k2_list);
    ModuleBase::timer::tick("MolecularWk", "cal_CVC_mo_k");

    ModuleBase::TITLE("MolecularWk", "transform_k_global");
    ModuleBase::timer::tick("MolecularWk", "transform_k_global");

    // gather all Wk
    auto gather_W = [&](std::vector<T>& target,
        const std::valarray<T>& value,
        int k1_step,
        int k2_step,
        double factor) -> void
    {
        const int npair = nocc * nvirt;
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
    for (int kai = 0; kai < this->nk; ++kai)
    {
        const int k1_step = kai * this->nocc * this->nvirt;
        const ModuleBase::Vector3<double> k1 = this->kv.kvec_d.at(kai);
        for (int kbj = 0; kbj < this->nk; ++kbj)
        {
            const int k2_step = kbj * this->nocc * this->nvirt;
            const ModuleBase::Vector3<double> k2 = this->kv.kvec_d.at(kbj);
            const RI::Tensor<T>& W_kai_kbj = Wk.at(RI_Util::Vector3_to_array3(k1)).at(RI_Util::Vector3_to_array3(k2));
            const double fac = 2.0 / static_cast<double>(this->nk); // factor 2 for Ha → Ry
            gather_W(WA_global, *W_kai_kbj.data, k1_step, k2_step, fac);
        }
    }
    ModuleBase::timer::tick("MolecularWk", "transform_k_global");
}

template <typename T>
TLRIk<T> MolecularWk<T>::cal_Csk_ao(const TLRI<T>& CsR_ao,
                                    const std::vector<Tk>& k_list)
{
    ModuleBase::timer::tick("MolecularWk", "cal_Csk_ao");
    TLRIk<T> Csk_ao; // <k, <I, <J, tensor{nabf, nwt1, nwt2}>>>
    for (const auto& k : k_list)
    {
        auto& Ck_ao = Csk_ao[k];
        for (const auto& c1 : CsR_ao)
        {
            const int iat1 = c1.first;
            auto &Ck_I = Ck_ao[iat1];
            for (const auto& c2: c1.second)
            {
                const int iat2 = c2.first.first;
                const TC& R = c2.first.second;
                double arg = 2.0 * M_PI * (k[0] * R[0] + k[1] * R[1] + k[2] * R[2]);
                std::complex<double> phase(cos(arg), sin(arg));

                const auto& tensor_ao = c2.second;
                std::size_t nabf = tensor_ao.shape[0];
                std::size_t nw1 = tensor_ao.shape[1];
                std::size_t nw2 = tensor_ao.shape[2];
                if (!Ck_I.count(iat2))
                {
                    Ck_I[iat2] = RI::Tensor<T>({nabf, nw1, nw2});
                }
                Ck_I[iat2] += tensor_ao * RI::Global_Func::convert<T>(phase);
            }
        }
    }
    ModuleBase::timer::tick("MolecularWk", "cal_Csk_ao");
    return Csk_ao;
}

/// @brief calculate Csk_mo by 
///         C'^\mu (m1,m2)[k1,k2] = c^*(m1,s)[k1] C^\mu (s,t)[k2] c(m2,t)[k2]
///         C^\mu (m1,m2)[k1,k2] = C'^\mu (m1,m2)[k1,k2] + C'^*\mu (m2,m1)[k2,k1]
/// @note this method is faster than method2, but only valid for MO1 = MO2, and k1_list = k2_list
template <typename T>
TCsk_mo<T> MolecularWk<T>::cal_Csk_mo(const UnitCell& ucell,
                                      const TLRIk<T>& Csk_ao,
                                      const psi::Psi<T>& psi_ks,
                                      const int& nocc,
                                      const int& nvirt,
                                      const LR::MO_TYPE type,
                                      const std::string& type_str,
                                      const std::vector<Tk>& kmo1_list,
                                      const std::vector<Tk>& kmo2_list)
{
    ModuleBase::TITLE("MolecularWk", "cal_Csk_mo");
    int nmo1, nmo2, imo1, imo2;
    switch(type)
    {
        case LR::MO_TYPE::OO:
            nmo1 = nocc; nmo2 = nocc; imo1 = 0; imo2 = 0;
            break;
        // case LR::MO_TYPE::VO:
        //     nmo1 = nocc; nmo2 = nvirt; imo1 = 0; imo2 = nocc;
        //     break;
        case LR::MO_TYPE::VV:
            nmo1 = nvirt; nmo2 = nvirt; imo1 = nocc; imo2 = nocc;
            break;
        default:
            throw std::runtime_error("MolecularWk::cal_CsR_mo: only support OO, VV");
    }
    const int nbands = nocc + nvirt;
    assert(psi_ks.get_nbands() == nbands);

    // <k, <iat, tensor{nmo, iat.nw}>>
    std::map<Tk, std::map<TA, RI::Tensor<T>>> psi1_k = slice_psi_k(psi_ks, imo1, nmo1, kmo1_list); 
    std::map<Tk, std::map<TA, RI::Tensor<T>>> psi2_k = slice_psi_k(psi_ks, imo2, nmo2, kmo2_list);

    // <{k1, k2}, <iat, tesnor{nabf, nmo1, nmo2}>>
    TCsk_mo<T> Csk_mo_part;  // C'^\mu (m1,m2)[k1,k2] = c^*(m1,s)[k1] C^\mu (s,t)[k2] c(m2,t)[k2]
    TCsk_mo<T> Csk_mo;       // C^\mu (m1,m2)[k1,k2] = C'^\mu (m1,m2)[k1,k2] + C'^*\mu (m2,m1)[k2,k1]
        
    ModuleBase::timer::tick("MolecularWk", "cal_Csk_mo_part");
    for (auto k1: kmo1_list)
    {
        std::map<TA, RI::Tensor<T>>& psi1_k1 = psi1_k.at(k1);
        for (auto k2: kmo2_list)
        {
            std::map<TA, RI::Tensor<T>>& psi2_k2 = psi2_k.at(k2);
            auto& Cmo_part_k1_k2 = Csk_mo_part[std::make_pair(k1, k2)];
            auto& Cmo_k1_k2 = Csk_mo[std::make_pair(k1, k2)];
            const std::map<TA, std::map<TA, RI::Tensor<T>>>& C_k2 = Csk_ao.at(k2);
            for (const auto& Ck_I : C_k2)
            {
                const int iat1 = Ck_I.first;
                const int it1 = ucell.iat2it[iat1];
                const int nw1 = ucell.atoms[it1].nw;
                const std::size_t nabf = Ck_I.second.begin()->second.shape[0];

                Cmo_part_k1_k2[iat1] = RI::Tensor<T>({nabf, (std::size_t)nmo1, (std::size_t)nmo2});
                Cmo_k1_k2[iat1] = RI::Tensor<T>({nabf, (std::size_t)nmo1, (std::size_t)nmo2});
                for (const auto& Ck_IJ: Ck_I.second)
                {
                    const int iat2 = Ck_IJ.first;
                    const int it2 = ucell.iat2it[iat2];
                    const int nw2 = ucell.atoms[it2].nw;

                    const auto& tensor_ao = Ck_IJ.second;
                    assert(nabf == tensor_ao.shape[0]);
                    assert(nw1 == tensor_ao.shape[1]);
                    assert(nw2 == tensor_ao.shape[2]);
                    
                    for (int iabf = 0; iabf < nabf; ++iabf)
                    {
                        const auto ptr = &tensor_ao(iabf, 0, 0);
                        std::vector<T> tmp(nmo1 * nw2);
                        // caution: Cs are row-major  (iw2 contiguous)
                        // C'(m1,m2) = c^*(m1,s) C(s,t) c(m2,t)         << row-major
                        // tmp_m1_t = (c_s_m1)^H (C_t_s)^T              << col-major
                        container::BlasConnector::gemm('C', 'T', nmo1, nw2, nw1,
                                                        1.0, psi1_k1[iat1].ptr(), nw1,
                                                        ptr, nw2,
                                                        0.0, tmp.data(), nmo1);
                        // C'_m2_m1 = (c_t_m2)^T (tmp_m1_t)^T           << col-major
                        container::BlasConnector::gemm('T', 'T', nmo2, nmo1, nw2,
                                                        1.0, psi2_k2[iat2].ptr(), nw2,
                                                        tmp.data(), nmo1,                                                        
                                                        1.0, &Cmo_part_k1_k2[iat1](iabf, 0, 0), nmo2);
                    }
                }
            }
        }
    }

    ModuleBase::timer::tick("MolecularWk", "cal_Csk_mo_part");

    // C'^\mu (m1,m2)[k1,k2] is finished, now calculate C^\mu (m1,m2)[k1,k2]
    ModuleBase::timer::tick("MolecularWk", "cal_Csk_mo_add");
    assert(nmo1 == nmo2);  // ATTENTION: this method only support MO1 = MO2 case

#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
    for (auto k1: kmo1_list)
    {
        for (auto k2: kmo2_list)
        {
            auto& Cmo_part_k1_k2 = Csk_mo_part.at(std::make_pair(k1, k2));
            auto& Cmo_part_k2_k1 = Csk_mo_part.at(std::make_pair(k2, k1));
            for (auto& Cmo1 : Cmo_part_k1_k2)
            {
                const int iat1 = Cmo1.first;
                RI::Tensor<T> t = Cmo1.second.copy();
                const RI::Tensor<T>& t2 = Cmo_part_k2_k1.at(iat1);
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
                Csk_mo.at(std::make_pair(k1, k2)).at(iat1) = std::move(t);
            }
        }
    }
    ModuleBase::timer::tick("MolecularWk", "cal_Csk_mo_add");

    // print C_mo max
    // this->print_Csk_mo_max(Csk_mo, "Csk_mo_" + type_str);
    // this->print_Csk_mo_max(Csk_mo_part, "Csk_mo_part_" + type_str);
    return Csk_mo;
}

/// @brief calculate Csk_mo by C^\mu (m1,m2)[k1,k2] 
///         = c^*(m1,s)[k1] C^\mu (s,t)[k2] c(m2,t)[k2]
///          +c(m2,s)[k2] C^\mu (s,t)[-k1] c^*(m1,t)[k1]
/// @note this method can calculate different MO1 and MO2, but slower than method1
template <typename T>
TCsk_mo<T> MolecularWk<T>::cal_Csk_mo_method2(const UnitCell& ucell,
                                              const TLRIk<T>& Csk_ao,
                                              const psi::Psi<T>& psi_ks,
                                              const int& nocc,
                                              const int& nvirt,
                                              const LR::MO_TYPE type,
                                              const std::string& type_str,
                                              const std::vector<Tk>& kmo1_list,
                                              const std::vector<Tk>& kmo2_list)
{
    ModuleBase::TITLE("MolecularWk", "cal_Csk_mo_method2");
    ModuleBase::timer::tick("MolecularWk", "cal_Csk_mo_method2");
    using namespace RI::Array_Operator;
    int nmo1, nmo2, imo1, imo2;
    switch(type)
    {
    case LR::MO_TYPE::OO:
        nmo1 = nocc; nmo2 = nocc; imo1 = 0; imo2 = 0;
        break;
    case LR::MO_TYPE::VO:
        nmo1 = nocc; nmo2 = nvirt; imo1 = 0; imo2 = nocc;
        break;
    case LR::MO_TYPE::VV:
        nmo1 = nvirt; nmo2 = nvirt; imo1 = nocc; imo2 = nocc;
        break;
    default:
        throw std::runtime_error("MolecularWk::cal_CsR_mo: only support OO, VV, VO");
    }

    const int nbands = nocc + nvirt;
    assert(psi_ks.get_nbands() == nbands);

    // <k, <iat, tensor{nmo, iat.nw}>>
    std::map<Tk, std::map<TA, RI::Tensor<T>>> psi1_k = slice_psi_k(psi_ks, imo1, nmo1, kmo1_list); 
    std::map<Tk, std::map<TA, RI::Tensor<T>>> psi2_k = slice_psi_k(psi_ks, imo2, nmo2, kmo2_list);

    // <{k1, k2}, <iat, tesnor{nabf, nmo1, nmo2}>>
    TCsk_mo<T> Csk_mo;       // C^\mu (m1,m2)[k1,k2] = c^*(m1,s)[k1] C^\mu (s,t)[k2] c(m2,t)[k2]
                             //                       +c(m2,s)[k2] C^\mu* (s,t)[k1] c^*(m1,t)[k1]
    for (auto k1: kmo1_list)
    {
        std::map<TA, RI::Tensor<T>>& psi1_k1 = psi1_k.at(k1);
        const std::map<TA, std::map<TA, RI::Tensor<T>>>& C_k1 = Csk_ao.at(k1);        
        for (auto k2: kmo2_list)
        {
            std::map<TA, RI::Tensor<T>>& psi2_k2 = psi2_k.at(k2);
            const std::map<TA, std::map<TA, RI::Tensor<T>>>& C_k2 = Csk_ao.at(k2);
            auto& Cmo_k1_k2 = Csk_mo[std::make_pair(k1, k2)];
            for (const auto& Ck_I : C_k2)
            {
                const int iat1 = Ck_I.first;
                const int it1 = ucell.iat2it[iat1];
                const int nw1 = ucell.atoms[it1].nw;
                const std::size_t nabf = Ck_I.second.begin()->second.shape[0];

                Cmo_k1_k2[iat1] = RI::Tensor<T>({nabf, (std::size_t)nmo1, (std::size_t)nmo2});
                for (const auto& Ck_IJ : Ck_I.second)
                {
                    const int iat2 = Ck_IJ.first;
                    const int it2 = ucell.iat2it[iat2];
                    const int nw2 = ucell.atoms[it2].nw;

                    const auto& tensor_ao = Ck_IJ.second;
                    assert(nabf == tensor_ao.shape[0]);
                    assert(nw1 == tensor_ao.shape[1]);
                    assert(nw2 == tensor_ao.shape[2]);
                    
                    const auto& tensor_ao_k1 = C_k1.at(iat1).at(iat2);
                    for (int iabf = 0; iabf < nabf; ++iabf)
                    {
                        const auto ptr = &tensor_ao(iabf, 0, 0);
                        const auto ptr_k1 = &tensor_ao_k1(iabf, 0, 0);
                        const auto ptr_desc = &Cmo_k1_k2[iat1](iabf, 0, 0);
                        // term 1
                        std::vector<T> tmp(nmo1 * nw2);
                        // caution: Cs are row-major  (iw2 contiguous)
                        // C(m1,m2) = c^*(m1,s) C(s,t) c(m2,t)          << row-major
                        // tmp_m1_t = (c_s_m1)^H (C_t_s)^T              << col-major
                        container::BlasConnector::gemm('C', 'T', nmo1, nw2, nw1,
                                                        1.0, psi1_k1[iat1].ptr(), nw1,
                                                        ptr, nw2,
                                                        0.0, tmp.data(), nmo1);
                        // C_m2_m1 = (c_t_m2)^T (tmp_m1_t)^T           << col-major
                        container::BlasConnector::gemm('T', 'T', nmo2, nmo1, nw2,
                                                        1.0, psi2_k2[iat2].ptr(), nw2,
                                                        tmp.data(), nmo1,                                                        
                                                        1.0, ptr_desc, nmo2);

                        // term2
                        tmp.resize(nmo1 * nw1);
                        // C(m1,m2) = c^*(m1,t) C^*(s,t) c(m2,s)        << row-major
                        // tmp_m1_s = (c_t_m1)^T (C_t_s)                << col-major
                        container::BlasConnector::gemm('T', 'N', nmo1, nw1, nw2,
                                                        1.0, psi1_k1[iat2].ptr(), nw2,
                                                        ptr_k1, nw2,
                                                        0.0, tmp.data(), nmo1);
                        // C_m2_m1 = (c_t_m2)^T (tmp_m1_t)^H            << col-major
                        container::BlasConnector::gemm('T', 'C', nmo2, nmo1, nw2,
                                                        1.0, psi2_k2[iat1].ptr(), nw1,
                                                        tmp.data(), nmo1,                                                        
                                                        1.0, ptr_desc, nmo2);
                    }
                }
            }
        }
    }

    // print C_mo max
    // this->print_Csk_mo_max(Csk_mo, "Csk_mo_method2_" + type_str);
    ModuleBase::timer::tick("MolecularWk", "cal_Csk_mo_method2");
    return Csk_mo;
}

template <typename T>
void MolecularWk<T>::print_Csk_mo_max(const TCsk_mo<T>& Csk_mo, const std::string& file_name)
{
    ModuleBase::TITLE("MolecularWk", "print_Csk_mo_max");
    ModuleBase::timer::tick("MolecularWk", "print_Csk_mo_max");
    std::ofstream ofs_out(file_name + "_" + std::to_string(GlobalV::MY_RANK));
    for (const auto& c1 : Csk_mo)
    {
        const Tk k1_d = c1.first.first;
        const Tk k2_d = c1.first.second;
        ModuleBase::Vector3<double> k1 = RI_Util::array3_to_Vector3(k1_d) * this->ucell.G;
        ModuleBase::Vector3<double> k2 = RI_Util::array3_to_Vector3(k2_d) * this->ucell.G;
        ofs_out << "k1: " << k1 << " k2: " << k2 << std::endl;
        double k1_norm = k1.norm();
        double k2_norm = k2.norm();
        for (const auto& c2 : c1.second)
        {
            const int iat = c2.first;
            const RI::Tensor<T>& tensor_mo = c2.second;
            double max = tensor_mo.max_abs();
            ofs_out << "Mu_atom: " << iat << " ,|k1|: " << k1_norm << " ,|k2|: " << k2_norm
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
    ModuleBase::timer::tick("MolecularWk", "print_Csk_mo_max");
}

/// @brief slice psi as <k, <iat, tensor{nmo, iat.nw}>>
template <typename T>
std::map<Tk, std::map<TA, RI::Tensor<T>>>
MolecularWk<T>::slice_psi_k(const psi::Psi<T>& psi_ks,
                            const int& imo,
                            const int& nmo,
                            const std::vector<Tk>& k_list)
{
    ModuleBase::TITLE("MolecularWk", "slice_psi_k");
    ModuleBase::timer::tick("MolecularWk", "slice_psi_k");
    std::map<Tk, std::map<TA, RI::Tensor<T>>> psi_k;
    for (const auto& k : k_list)
    {
        auto& psi_k_at = psi_k[k];
        int k_index = this->kpoint_index_map.at(k);
        for (int iat = 0; iat < this->ucell.nat; ++iat)
        {
            const int it = this->ucell.iat2it[iat];
            const int nw = this->ucell.atoms[it].nw;
            RI::Tensor<T> t({(std::size_t)nmo, (std::size_t)nw});
            for (int im = 0; im < nmo; ++im)
            {
                for (int iw = 0; iw < nw; ++iw)
                {
                    t(im, iw) = psi_ks(k_index, imo + im, this->ucell.get_iat2iwt()[iat]+iw);
                }
            }
            psi_k_at[iat] = std::move(t);
            //assert(nmo * nw == LR_Util::print_value(psi_k_at[iat].ptr(), nmo, nw));
        }
        //std::cout<<"Slice psi_k for k: " << k[0]<<","<<k[1]<<","<<k[2]<<" done."<<std::endl;
    }
    ModuleBase::timer::tick("MolecularWk", "slice_psi_k");
    return psi_k;
}

}// namespace BSE