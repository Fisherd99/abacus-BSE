#pragma once
#include <algorithm>
#include <dirent.h>
#include "ri_benchmark.h"
#include "source_base/module_container/base/third_party/blas.h"
#include "source_psi/psi.h"
#include "source_base/module_external/scalapack_connector.h"
namespace RI_Benchmark
{
    // std::cout << "the size of Cs:" << std::endl;
    // for (auto& it1: Cs)
    // {
    //     for (auto& it2: it1.second)
    //     {
    //         std::cout << "iat1=" << it1.first << ", iat2= " << it2.first.first << std::endl;
    //         auto& ts = it2.second.shape;
    //         std::cout << "Tensor shape:" << ts[0] << " " << ts[1] << " " << ts[2] << std::endl; // abf, nw1, nw2
    //     }
    // }

    template <typename TR>
    inline int count_nao_from_Cs(const TLRI<TR>& Cs_ao)
    {
        int naos = 0;
        for (const auto& it2 : Cs_ao.at(0))
        {
            assert(it2.second.shape.size() == 3);
            naos += it2.second.shape[2];
        }
        return naos;
    }

    /// @brief slice the psi from wfc_ks of all the k-points, some bands, some basis functions
    /// @return
    template <typename TK>
    inline std::vector<TK> slice_psi(const psi::Psi<TK>& wfc_ks,
        const int& ibstart,
        const int& nb,
        const int& iwstart,
        const int& nw)
    {
        std::vector<TK> psi_slice(wfc_ks.get_nk() * nb * nw);
        int i = 0;
        for (int ik = 0; ik < wfc_ks.get_nk(); ++ik)
            for (int ib = ibstart; ib < ibstart + nb; ++ib)
                for (int iw = iwstart; iw < iwstart + nw; ++iw)
                    psi_slice[i++] = wfc_ks(ik, ib, iw);
        assert(i == psi_slice.size());
        return psi_slice;
    }

    template <typename TK, typename TR>
    TLRI<TK> cal_Cs_mo(const UnitCell& ucell,
        const TLRI<TR>& Cs_ao,
        const psi::Psi<TK>& wfc_ks,
        const int& nocc,
        const int& nvirt,
        const int& occ_first,
        const bool& read_from_aims,
        const std::vector<int>& aims_nbasis)
    {
        // assert(wfc_ks.get_nk() == 1);   // currently only gamma-only is supported
        assert(nocc + nvirt <= wfc_ks.get_nbands());
        const bool use_aims_nbasis = (read_from_aims && !aims_nbasis.empty());
        TLRI<TK> Cs_mo;
        int iw1 = 0;
        for (auto& c1 : Cs_ao)
        {
            const int& iat1 = c1.first;
            const int& it1 = ucell.iat2it[iat1];
            const int& nw1 = (use_aims_nbasis ? aims_nbasis[it1] : ucell.atoms[it1].nw);
            if (!use_aims_nbasis) { assert(iw1 == ucell.get_iat2iwt()[iat1]); }
            int iw2 = 0;
            for (auto& c2 : c1.second)
            {
                const int& iat2 = c2.first.first;
                const int& it2 = ucell.iat2it[iat2];
                const int& nw2 = (use_aims_nbasis ? aims_nbasis[it2] : ucell.atoms[it2].nw);
                if (!use_aims_nbasis) { assert(iw2 == ucell.get_iat2iwt()[iat2]); }

                const auto& tensor_ao = c2.second;
                const size_t& nabf = tensor_ao.shape[0];
                assert(tensor_ao.shape.size() == 3); // abf, nw1, nw2

                std::vector<TK> psi_a1 = occ_first ? slice_psi(wfc_ks, 0, nocc, iw1, nw1)   //occ
                    : slice_psi(wfc_ks, nocc, nvirt, iw1, nw1); // virt
                std::vector<TK> psi_a2 = occ_first ? slice_psi(wfc_ks, nocc, nvirt, iw2, nw2)   //virt
                    : slice_psi(wfc_ks, 0, nocc, iw2, nw2); // occ

                Cs_mo[c1.first][c2.first] = RI::Tensor<TK>({ nabf, (std::size_t)nocc, (std::size_t)nvirt });
                for (int iabf = 0; iabf < nabf; ++iabf)
                {
                    const auto ptr = &tensor_ao(iabf, 0, 0);
                    std::vector<TK> tmp(nw1 * (occ_first ? nvirt : nocc));
                    // caution: Cs are row-major  (ia2 contiguous)
                    if (occ_first)
                    {
                        container::BlasConnector::gemm('T', 'N', nvirt, nw1, nw2, 1.0, psi_a2.data(), nw2, ptr, nw2, 0.0, tmp.data(), nvirt);
                        container::BlasConnector::gemm('N', 'N', nvirt, nocc, nw1, 1.0, tmp.data(), nvirt, psi_a1.data(), nw1, 0.0, &Cs_mo[c1.first][c2.first](iabf, 0, 0), nvirt);
                    }
                    else
                    {
                        container::BlasConnector::gemm('T', 'N', nw1, nocc, nw2, 1.0, ptr, nw2, psi_a2.data(), nw2, 0.0, tmp.data(), nw1);
                        container::BlasConnector::gemm('T', 'N', nvirt, nocc, nw1, 1.0, psi_a1.data(), nw1, tmp.data(), nw1, 0.0, &Cs_mo[c1.first][c2.first](iabf, 0, 0), nvirt);
                    }
                }
                iw2 += nw2;
            }
            iw1 += nw1;
        }
        return Cs_mo;
    }

    template <typename TK, typename TR>
    std::vector<TK> cal_Amat_full(const TLRI<TK>& Cs_a,
        const TLRI<TK>& Cs_b,
        const TLRI<TR>& Vs)
    {
        assert(Cs_a.size() > 0);
        assert(Cs_b.size() > 0);
        assert(Vs.size() > 0);
        auto& Cs_shape = Cs_a.at(0).begin()->second.shape;
        auto& Vs_shape = Vs.at(0).begin()->second.shape;
        assert(Cs_shape.size() == 3); // abf, nocc, nvirt
        assert(Cs_shape.size() == 3); // abf, nocc, nvirt
        assert(Vs_shape.size() == 2); // abf, abf

        const int& npairs = Cs_shape[1] * Cs_shape[2];
        std::vector<TK> Amat_full(npairs * npairs, 0.0);

        for (auto& itv1 : Vs)
        {
            const int& iat1 = itv1.first;
            for (auto& itv2 : itv1.second)
            {
                const int& iat2 = itv2.first.first;
                const auto& tensor_v = itv2.second; // (nabf2, nabf1), T
                for (auto& itca2 : Cs_a.at(iat1))
                {
                    const int& iat3 = itca2.first.first;
                    const auto& tensor_ca = itca2.second; // (nvirt*nocc, nabf1), N
                    for (auto& itcb2 : Cs_b.at(iat2))
                    {
                        const int& iat4 = itcb2.first.first;
                        const auto& tensor_cb = itcb2.second; // (nvirt*nocc, nabf2), T
                        const int& nabf1 = tensor_v.shape[0];
                        const int& nabf2 = tensor_v.shape[1];
                        assert(tensor_ca.shape[0] == nabf1);   //abf1
                        assert(tensor_cb.shape[0] == nabf2); //abf2
                        std::vector<TK> tmp(npairs * nabf1);
                        container::BlasConnector::gemm('T', 'T', nabf1, npairs, nabf2, 1.0, tensor_v.ptr(), nabf2, tensor_cb.ptr(), npairs, 0.0, tmp.data(), nabf1);
                        container::BlasConnector::gemm('N', 'N', npairs, npairs, nabf1, 2.0/*Hartree to Ry*/, tensor_ca.ptr(), npairs, tmp.data(), nabf1, 1.0, Amat_full.data(), npairs);
                    }
                }
            }
        }
        return Amat_full;
    }
    template <typename TK>
    TLRIX<TK> cal_CsX(const TLRI<TK>& Cs_mo, const TK* X)
    {
        TLRIX<TK> CsX;
        for (auto& it1 : Cs_mo)
        {
            const int& iat1 = it1.first;
            for (auto& it2 : it1.second)
            {
                const int& iat2 = it2.first.first;
                auto& tensor_c = it2.second;
                const int& nabf = tensor_c.shape[0];
                const int& npairs = tensor_c.shape[1] * tensor_c.shape[2];
                std::vector<TK> CX(nabf);
                for (int iabf = 0;iabf < nabf;++iabf)
                    CX[iabf] = container::BlasConnector::dot(npairs, &tensor_c(iabf, 0, 0), 1, X, 1);
                CsX[iat1][it2.first] = CX;
            }
        }
        return CsX;
    }

    template <typename TK, typename TR>
    TLRI<TK> cal_CV(const TLRI<TK>& Cs_a_mo,
        const TLRI<TR>& Vs)
    {
        TLRI<TK> CV;
        for (auto& it1 : Cs_a_mo)   //the atom on which ABFs locate
        {
            const int& iat1 = it1.first;
            for (auto& it2 : it1.second)
            {
                const int& iat2 = it2.first.first;
                // const auto& Rc=it2.first.second;
                auto& tensor_c = it2.second;
                const int& nabf1 = tensor_c.shape[0];
                const int& npairs = tensor_c.shape[1] * tensor_c.shape[2];
                for (auto& it3 : Vs.at(iat1))
                {
                    const int& iat3 = it3.first.first;
                    // const auto& Rv=it3.first.second;
                    const auto& tensor_v = it3.second;
                    assert(nabf1 == tensor_v.shape[0]);
                    const size_t& nabf2 = tensor_v.shape[1];
                    std::vector<TK> tmp(nabf2 * npairs);
                    // const auto& Rcv = (Rv - Rc) % period;
                    if (CV.count(iat2) && CV.at(iat2).count({ iat3, {0, 0, 0} }))   // add-up, sum over iat1
                    {
                        auto& tensor_cv = CV.at(iat2).at({ iat3, {0, 0, 0} });
                        container::BlasConnector::gemm('N', 'T', npairs, nabf2, nabf1, 1.0, tensor_c.ptr(), npairs, tensor_v.ptr(), nabf2, 1.0, tensor_cv.ptr(), npairs);
                    }
                    else
                    {
                        RI::Tensor<TK> tmp({ nabf2, tensor_c.shape[1], tensor_c.shape[2] });  // (nabf2, nocc, nvirt)
                        container::BlasConnector::gemm('N', 'T', npairs, nabf2, nabf1, 1.0, tensor_c.ptr(), npairs, tensor_v.ptr(), nabf2, 0.0, tmp.ptr(), npairs);
                        CV[iat2][{iat3, { 0, 0, 0 }}] = tmp;
                    }
                }
            }
        }
        return CV;
    }
    template <typename TK, typename TR>
    void cal_AX(const TLRI<TK>& Cs_a,
        const TLRIX<TK>& Cs_bX,
        const TLRI<TR>& Vs,
        TK* AX,
        const double& scale)
    {
        const int& npairs = Cs_a.at(0).begin()->second.shape[1] * Cs_a.at(0).begin()->second.shape[2];
        for (auto& itv1 : Vs)
        {
            const int& iat1 = itv1.first;
            for (auto& itv2 : itv1.second)
            {
                const int& iat2 = itv2.first.first;
                const auto& tensor_v = itv2.second; // (nabf2, nabf1), T
                for (auto& itca2 : Cs_a.at(iat1))
                {
                    const int& iat3 = itca2.first.first;
                    const auto& tensor_ca = itca2.second; // (nvirt*nocc, nabf1), N
                    for (auto& itcb2 : Cs_bX.at(iat2))
                    {
                        const int& iat4 = itcb2.first.first;
                        const auto& vector_cb = itcb2.second; // (nvirt*nocc, nabf2), T
                        const int& nabf1 = tensor_v.shape[0];
                        const int& nabf2 = tensor_v.shape[1];
                        assert(tensor_ca.shape[0] == nabf1);   //abf1
                        assert(vector_cb.size() == nabf2); //abf2
                        std::vector<TK> tmp(nabf1);
                        container::BlasConnector::gemv('T', nabf1, nabf2, 1.0, tensor_v.ptr(), nabf2, vector_cb.data(), 1, 0.0, tmp.data(), 1);
                        container::BlasConnector::gemv('N', npairs, nabf1, scale/*Hartree to Ry; singlet*/, tensor_ca.ptr(), npairs, tmp.data(), 1, 1.0, AX, 1);
                    }
                }
            }
        }
    }

    template <typename TK>
    void cal_AX(const TLRI<TK>& CV,
        const TLRIX<TK>& Cs_bX,
        TK* AX,
        const double& scale)
    {
        for (auto& it1 : CV)
        {
            const int& iat1 = it1.first;
            for (auto& it2 : it1.second)
            {
                const int& iat2 = it2.first.first;
                const auto& tensor_cv = it2.second; // (nabf, nocc, nvirt)
                const int& npairs = tensor_cv.shape[1] * tensor_cv.shape[2];
                for (auto& it3 : Cs_bX.at(iat2))
                {
                    const int& iat3 = it3.first.first;
                    const auto& vector_cx = it3.second; // (nabf)
                    const int& nabf = tensor_cv.shape[0];
                    assert(vector_cx.size() == nabf); //abf on at2
                    container::BlasConnector::gemv('N', npairs, nabf, scale/*Hartree to Ry; singlet*/, tensor_cv.ptr(), npairs, vector_cx.data(), 1, 1.0, AX, 1);
                }
            }
        }
    }

    template <typename FPTYPE>
    std::vector<FPTYPE> read_aims_ebands(const std::string& file, const int nocc, const int nvirt, int& ncore)
    {
        std::vector<FPTYPE> bands;
        std::vector<FPTYPE> bands_final;
        std::ifstream ifs;
        ifs.open(file);
        std::string tmp;
        FPTYPE ene, occ;
        for (int i = 0;i < 6;++i) { std::getline(ifs, tmp); } // skip the first 6 lines
        int ivirt = 0;
        while (ifs.peek() != EOF) {
            ifs >> tmp >> occ >> ene >> tmp;
            std::cout << "occ=" << occ << ", ene=" << ene << std::endl;
            bands.push_back(ene * 2);//Hartree to Ry
            if (occ < 0.1) { ++ivirt; }
            if (ivirt == nvirt) { break; }
        }
        ncore = bands.size() - nocc - nvirt;
        std::cout << "bands_final:" << std::endl;
        for (int i = ncore;i < bands.size();++i)
        {
            bands_final.push_back(bands[i]);
            std::cout << bands[i] << "  ";
        }
        std::cout << std::endl;
        return bands_final;
    }

    /// @brief  read the eigenvectors from librpa, only for spin degenerate
    template <typename TK>
    void read_librpa_eigenvectors(psi::Psi<TK>& wfc_ks, const std::string& path, const int ncore, const int nbands_file,
        const int nspin_tmp, const int nspin_file, Parallel_Orbitals& pmat) {
        int nbands = pmat.get_wfc_global_nbands();// nbands = nocc + nvirt
        int nbasis = pmat.get_wfc_global_nbasis();
        const size_t nk = PARAM.inp.nspin == 2 ? wfc_ks.get_nk() / 2 : wfc_ks.get_nk();
        std::vector<std::vector<TK>> wfc_ks_tot(
            wfc_ks.get_nk(),
            std::vector<TK>(((GlobalV::MY_RANK == 0) ? nbands * nbasis : 0), 0.0)); // glboal wfc
            if (GlobalV::MY_RANK == 0) {
            struct dirent *ptr;
            DIR *dir;
            dir = opendir(path.c_str());
            std::vector<bool> readen_k(nk, false);

            while ((ptr = readdir(dir)) != NULL){// read all the files in the directory
                std::string fm(ptr->d_name);
                if (fm.find("KS_eigenvector") == 0)// find file KS_eigenvectorXXX
                {
                    std::cout << "found librpa_eigenvector file:" << fm << std::endl;
                    std::ifstream file_librpa_ks(path + fm);
                    std::string tmp;
                    while (file_librpa_ks.peek() != EOF)
                    {
                        int ik;
                        file_librpa_ks >> ik;
                        ik = ik - 1;
                        assert(readen_k[ik] == false);
                        for (int iw = 0; iw < nbasis; ++iw) {
                            for (int ib = 0; ib < nbands_file; ++ib) {
                                for (int is = 0; is < nspin_file; ++is) {
                                    if (ib >= ncore && ib< (ncore+nbands)) {
                                        RI_Benchmark::read_one_data(file_librpa_ks, wfc_ks_tot[ik+is*nk][(ib-ncore)*nbasis + iw]);
                                        file_librpa_ks >> std::ws; // skip the blank if there is
                                    }
                                    else {
                                        std::getline(file_librpa_ks, tmp); //skip the useless bands
                                    }
                                }
                            }
                        }
                        if (nspin_tmp == 2 && nspin_file == 1) {
                            wfc_ks_tot[ik + nk] = wfc_ks_tot[ik];
                        }
                        readen_k[ik] = true;
                    }
                }
            }
            closedir(dir);
            for(int ik = 0; ik < nk; ++ik) {
                if (!readen_k[ik])
                    throw std::runtime_error("librpa_eigenvector file not found for k-point " + std::to_string(ik+1));
            }
            
        }// end of if (GlobalV::MY_RANK == 0) ;
        for (int iks = 0; iks < wfc_ks.get_nk(); ++iks){
            // test: output wfc
            if (GlobalV::MY_RANK == 0) {
                std::cout << "wfc_gs_read_from_librpa for iks:" << iks << std::endl;
                for (int ib = 0;ib < nbands;++ib)
                {
                    for (int iw = 0;iw < nbasis;++iw)
                    {
                        std::cout << wfc_ks_tot[iks][ib * nbasis + iw] << "  ";
                    }
                    std::cout << std::endl;
                }
            }
            // test: output wfc
            wfc_ks.fix_k(iks);
#ifdef __MPI
            Parallel_2D pv_glb;
            pv_glb.set(nbasis, nbands, std::max(nbasis, nbands), pmat.blacs_ctxt);
            Cpxgemr2d(nbasis, nbands, wfc_ks_tot[iks].data(), 1, 1, pv_glb.desc,
                        wfc_ks.get_pointer(), 1, 1, const_cast<int*>(pmat.desc_wfc),
                        pv_glb.blacs_ctxt);
#else
            BlasConnector::copy(nbands*nlocal, wfc_ks_tot[iks].data(), 1, wfc_ks.get_pointer(), 1);
#endif
        }
    }

    /// @brief  read the eigenvectors from FHI-aims, only for gamma_only and spin degenerate
    template <typename TK>
    void read_aims_eigenvectors(psi::Psi<TK>& wfc_ks, const std::string& file, const int ncore, const int nbands, const int nbasis)
    {
        std::ifstream ifs;
        ifs.open(file);
        std::string tmp;
        int nbands_last = 0;
        while (ifs.peek() != EOF)
        {
            std::getline(ifs, tmp); //the first line
            std::stringstream ss(tmp);
            while (std::getline(ss, tmp, ' ')) {};
            int nbands_file = std::stoi(tmp);
            for (int iw = 0;iw < nbasis;++iw)
            {
                ifs >> tmp >> tmp >> tmp >> tmp >> tmp >> tmp;  //useless cols
                for (int ib = nbands_last; ib < nbands_file;++ib)
                {
                    ifs >> tmp;
                    if (ib >= ncore && ib < ncore + nbands)
                    {
                        for (int is = 0;is < wfc_ks.get_nk();++is)
                        {   //only for gamma_only and spin degenerate
                            wfc_ks(is, ib - ncore, iw) = std::stod(tmp);
                        }
                    }
                }
            }
            std::getline(ifs, tmp); // the interval line between two blocks
            std::getline(ifs, tmp); // the interval line between two blocks
            nbands_last = nbands_file;
        }
        // output wfc
        std::cout << "wfc_gs_read_from_aims:" << std::endl;
        for (int ib = 0;ib < nbands;++ib)
        {
            for (int iw = 0;iw < nbasis;++iw)
            {
                std::cout << wfc_ks(0, ib, iw) << "  ";
            }
            std::cout << std::endl;
        }
    }
    
    template <typename TCs, typename TR> // only for blocking by atom pairs (abacus type)
    TLRI<TR> read_coulomb_mat(const std::string& file, const TLRI<TCs>& Cs, const BSE::RI_kRlist& kRlist )
    {
        std::ifstream ifs;
        ifs.open(file);
        size_t nk = 0, nabf = 0, istart = 0, jstart = 0, iend = 0, jend = 0;
        std::string tmp;
        const std::unique_ptr<K_Vectors>& klist = kRlist.klist;
        ifs >> nk;//   nkstot(actually nk)
        assert(nk == klist->get_nks());
        int ik_readin = -1;
        TLRI<TR> Vs;
        std::map<int, std::map<std::pair<int,int>, RI::Tensor<std::complex<double>>>> Vq; // <iat1, <<iat2,ik>, T>>
        const int nat = Cs.size();
        for (int iat1 = 0;iat1 < nat;++iat1)
        {
            for (int ik =0;ik < nk;++ik)
            {
                const size_t nabf1 = Cs.at(iat1).at({ 0, {0,0,0} }).shape[0];
                for (int iat2 = 0;iat2 < nat;++iat2)
                {
                    if (iat1 > iat2)
                    {   // coulomb_mat has only the upper triangle part
                        Vq[iat1][{iat2, ik}] = Vq[iat2][{iat1, ik}].dagger();
                        continue;
                    }
                    const size_t nabf2 = Cs.at(iat2).at({ 0, {0,0,0} }).shape[0];
                    ifs >> nabf >> istart >> iend >> jstart >> jend >> ik_readin >> klist->wk[ik];
                    assert(ik_readin == ik+1);
                    assert(nabf1 == iend - istart + 1);
                    assert(nabf2 == jend - jstart + 1);
                    RI::Tensor<std::complex<double>> t({ nabf1, nabf2 });
                    for (int i = 0;i < nabf1;++i)
                    {
                        for (int j = 0;j < nabf2;++j)
                        {
                            RI_Benchmark::read_one_data(ifs, t(i, j));
                        }
                    }
                    Vq[iat1][{iat2, ik}] = t;
                }
            }
        }

        auto array3_to_Vector3_double = [](const std::array<int, 3>& v) -> ModuleBase::Vector3<double> {
            return ModuleBase::Vector3<double>{static_cast<double>(v[0]), 
                                            static_cast<double>(v[1]), 
                                            static_cast<double>(v[2])};
        };
        for ( const TC& iR : kRlist.Rlist )
        {
            std::cout<<"FISH_OUTPUT: in read V: iR="<<iR[0]<<" "<<iR[1]<<" "<<iR[2]<<std::endl;
            
            for (int iat1 = 0;iat1 < nat;++iat1)
            {
                for (int iat2 = 0;iat2 < nat;++iat2)
                {
                    Vs[iat1][{iat2, iR}] = RI::Tensor<TR>({ Vq[iat1][{iat2, 0}].shape[0], Vq[iat1][{iat2, 0}].shape[1] });
                    for (int ik = 0;ik < nk;++ik)
                    {
                    const double arg = -1.0 * ModuleBase::TWO_PI * (klist->kvec_d[ik] * array3_to_Vector3_double(iR));
                    const std::complex<double> kphase (cos(arg), sin(arg));
                    Vs[iat1][{iat2, iR}] += RI::Global_Func::convert<TR> (Vq[iat1][{iat2, ik}] * kphase) * RI::Global_Func::convert<TR>(klist->wk[ik]);
                    }
                }
            }
        }
        return Vs;
    }

    template <typename TCs, typename TR> // any blocking (aims type)
    TLRI<TR> read_coulomb_mat_general(const std::string& file, const TLRI<TCs>& Cs, const BSE::RI_kRlist& kRlist)
    {
        std::ifstream ifs;
        ifs.open(file);
        size_t nk = 0, nabf = 0, istart = 0, jstart = 0, iend = 0, jend = 0;
        std::string tmp;
        const std::unique_ptr<K_Vectors>& klist = kRlist.klist;
        ifs >> nk;//   nkstot(actually nk)
        assert(nk == klist->get_nks());
        int ik_readin = -1;
        TLRI<TR> Vs;
        std::map<int, std::map<std::pair<int,int>, RI::Tensor<std::complex<double>>>> Vq; // <iat1, <<iat2,ik>, T>>
        std::map<int,std::vector<std::complex<double>>> Vq_tmp; //<ik, vector> 
        while (ifs.peek() != EOF)
        {
            ifs >> nabf >> istart >> iend >> jstart >> jend >> ik_readin >> klist->wk[ik_readin-1];
            if (ifs.peek() == EOF) { break; }
            int ik = ik_readin - 1;
            if (Vq_tmp[ik].empty()) { Vq_tmp[ik].resize(nabf * nabf, 0.0); }
            for (int i = istart - 1;i < iend;++i)
            {
                for (int j = jstart - 1;j < jend;++j)
                {
                    RI_Benchmark::read_one_data(ifs, Vq_tmp.at(ik)[i * nabf + j]);
                }
            }
        }
        const int nat = Cs.size();
        istart = 0;    // 
        for (int iat1 = 0;iat1 < nat;++iat1)
        {
            const size_t nabf1 = Cs.at(iat1).at({ 0, {0,0,0} }).shape[0];
            jstart = 0;
            for (int iat2 = 0;iat2 < nat;++iat2)
            {
                const size_t nabf2 = Cs.at(iat2).at({ 0, {0,0,0} }).shape[0];
                for (int ik = 0; ik < nk; ++ik){                    
                    if (iat1 > iat2)
                    {   // coulomb_mat has only the upper triangle part
                        Vq[iat1][{iat2, ik}] = Vq[iat2][{iat1, ik}].dagger();
                    }
                    else
                    {
                        RI::Tensor<std::complex<double>> t({ nabf1, nabf2 });
                        for (int i = 0;i < nabf1;++i)
                        {
                            for (int j = 0;j < nabf2;++j)
                            {
                                t(i, j) = Vq_tmp[ik][(istart + i) * nabf + jstart + j];
                            }
                        }
                        Vq[iat1][{iat2, ik}] = t;
                    }
                }
                jstart += nabf2;
            }
            assert(jstart == nabf);
            istart += nabf1;
        }
        assert(istart == nabf);

        auto array3_to_Vector3_double = [](const std::array<int, 3>& v) -> ModuleBase::Vector3<double> {
            return ModuleBase::Vector3<double>{static_cast<double>(v[0]), 
                                            static_cast<double>(v[1]), 
                                            static_cast<double>(v[2])};
        };
        for ( const TC& iR : kRlist.Rlist )
        {
            std::cout<<"FISH_OUTPUT: in read V: iR="<<iR[0]<<" "<<iR[1]<<" "<<iR[2]<<std::endl;
            for (int iat1 = 0;iat1 < nat;++iat1)
            {
                for (int iat2 = 0;iat2 < nat;++iat2)
                {
                    Vs[iat1][{iat2, iR}] = RI::Tensor<TR>({ Vq[iat1][{iat2, 0}].shape[0], Vq[iat1][{iat2, 0}].shape[1] });
                    for (int ik = 0; ik < nk; ++ik)
                    {
                    const double arg = -1.0 * ModuleBase::TWO_PI * (klist->kvec_d[ik] * array3_to_Vector3_double(iR));
                    const std::complex<double> kphase (cos(arg), sin(arg));
                    Vs[iat1][{iat2, iR}] += RI::Global_Func::convert<TR>(Vq[iat1][{iat2, ik}] * kphase) * RI::Global_Func::convert<TR>(klist->wk[ik]);
                    }
                }
            }
        }
        return Vs;
    }

    template < typename TR>
    bool compare_Vs(const TLRI<TR>& Vs1, const TLRI<TR>& Vs2, const double thr)
    {
        for (auto& tmp1 : Vs1)
        {
            const int& iat1 = tmp1.first;
            for (auto& tmp2 : tmp1.second)
            {
                const int& iat2 = tmp2.first.first;
                const RI::Tensor<TR>& t1 = tmp2.second;
                const RI::Tensor<TR>& t2 = Vs2.at(iat1).at({ iat2, {0,0,0} });
                if (t1.shape[0] != t2.shape[0]) { return false; }
                if (t1.shape[1] != t2.shape[1]) { return false; }
                for (int i = 0;i < t1.shape[0];++i)
                {
                    for (int j = 0;j < t1.shape[1];++j)
                    {
                        if (std::abs(t1(i, j) - t2(i, j)) > thr) { std::cout << "element (" << i << ", " << j << ") are differernt: " << t1(i, j) << ", " << t2(i, j) << std::endl;return false; }
                    }
                }
            }
        }
        return true;
    }
    template <typename TR>
    std::vector<TLRI<TR>> split_Ds(const std::vector<std::vector<TR>>& Ds, const std::vector<int>& aims_nbasis, const UnitCell& ucell) // vector index: ispin
    {
        // Due to the hard-coded constructor of elecstate::DensityMatrix, singlet-triplet with nspin=2 cannot use DM_trans with size 1
        // if(Ds.size()>1) { throw std::runtime_error("split_Ds only supports gamma-only spin-1 Ds now."); }
        std::vector<TLRI<TR>> Ds_split;
        for (const auto& D : Ds)
        {
            TLRI<TR> D_split;
            const int nbasis = std::sqrt(D.size());
            int iw1_start = 0;
            for (int iat1 = 0;iat1 < ucell.nat;++iat1)
            {
                const int& it1 = ucell.iat2it[iat1];
                const size_t& nw1 = aims_nbasis[it1];
                int iw2_start = 0;
                for (int iat2 = 0;iat2 < ucell.nat;++iat2)
                {
                    const int& it2 = ucell.iat2it[iat2];
                    const size_t& nw2 = aims_nbasis[it2];
                    D_split[iat1][{iat2, { 0,0,0 }}] = RI::Tensor<TR>({ nw1, nw2 });
                    for (int i = 0;i < nw1;++i)
                    {
                        for (int j = 0;j < nw2;++j)
                        {
                            D_split[iat1][{iat2, { 0,0,0 }}](i, j) = D[(iw1_start + i)*nbasis+(iw2_start + j)] * 0.5; // consistent with split_m2D_ktoR
                        }
                    }
                    iw2_start += nw2;
                }
                assert(iw2_start == nbasis);
                iw1_start += nw1;
            }
            assert(iw1_start == nbasis);
            Ds_split.push_back(D_split);
        }
        return Ds_split;
    }
}
