#include "molecular_lri.h"
#include <RI/distribute/Distribute_Equally.h>
#include <cstddef>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __MKL
#include <mkl_service.h>
#endif
namespace BSE
{

template <typename T>
void MolecularLRI<T>::init(TLRI<T>& Cs_in, TLRI<T>& Vs_in, TLRI<T>& Ws_in, const Exx_Info::Exx_Info_RI& info_ri)
{
    ModuleBase::TITLE("MolecularLRI", "init");
    // 1-1. distribute atom and k-point tasks among MPI processes
    ModuleBase::timer::tick("MolecularLRI", "distribute_atom_and_k");
    int nproc = GlobalV::NPROC;
    std::set<int> set_I, set_J, set_IJ, set_k;
    std::vector<int> list_k1_index, list_k2_index;
    int task_sizes = this->ucell.nat * this->ucell.nat * this->nk * this->nk;
    std::cout << "Total Molecular LRI tasks: " << task_sizes << ", number of MPI processes: " << nproc << std::endl;
    RI::Distribute_Equally::distribute_atom_and_k_pair(MPI_COMM_WORLD,
                                                       (std::size_t)this->ucell.nat,
                                                       (std::size_t)this->nk,
                                                       this->list_I,
                                                       this->list_J,
                                                       list_k1_index,
                                                       list_k2_index,
                                                       false);

    set_I.insert(this->list_I.begin(), this->list_I.end());
    set_J.insert(this->list_J.begin(), this->list_J.end());
    set_IJ.insert(set_I.begin(), set_I.end());
    set_IJ.insert(set_J.begin(), set_J.end());
    // transform set to vector, since openmp cannot handle set for parallelization
    this->list_IJ.assign(set_IJ.begin(), set_IJ.end());

    for (int k1 : list_k1_index)
    {
        this->k1_list.push_back(RI_Util::Vector3_to_array3(this->kv.kvec_d.at(k1)));
        this->is_local_k1[k1] = true;
    }
    for (int k2 : list_k2_index)
    {
        this->k2_list.push_back(RI_Util::Vector3_to_array3(this->kv.kvec_d.at(k2)));
    }
    set_k.insert(list_k1_index.begin(), list_k1_index.end());
    set_k.insert(list_k2_index.begin(), list_k2_index.end());
    for (int k : set_k)
    {
        this->k_list.push_back(RI_Util::Vector3_to_array3(this->kv.kvec_d.at(k)));
    }

    int proc_ntasks = this->list_I.size() * this->list_J.size() * this->k1_list.size() * this->k2_list.size();
    GlobalV::ofs_running << "Molecular LRI init: Process " << GlobalV::MY_RANK
        << " handles " << proc_ntasks << " tasks." << std::endl;
    print_a(GlobalV::ofs_running, this->list_I, "list_I");
    print_a(GlobalV::ofs_running, this->list_J, "list_J");
    print_k(GlobalV::ofs_running, this->k1_list, "k1_list");
    print_k(GlobalV::ofs_running, this->k2_list, "k2_list");
    // LRI_CV_Tools::write_Cs_ao(Cs_in, PARAM.globalv.global_out_dir + "Cs_in_test_" + std::to_string(GlobalV::MY_RANK));
    // LRI_CV_Tools::write_Vs_abf(Vs_in, PARAM.globalv.global_out_dir + "Vs_in_test_" + std::to_string(GlobalV::MY_RANK));
    // LRI_CV_Tools::write_Vs_abf(Ws_in, PARAM.globalv.global_out_dir + "Ws_in_test_" + std::to_string(GlobalV::MY_RANK));
    
    // 1-2. move R tensors to nearest image
    double dist;
    std::set<int> all_atoms;
    for (int i = 0; i < this->ucell.nat; ++i)
    {
        all_atoms.insert(i);
        for (int j = 0; j < this->ucell.nat; ++j)
        {
            for (const TC R_original : this->kRlist.Rlist)
            {
				const TC R = cell_nearest.cell_nearest_check(i, j, R_original, dist);
                if (R != R_original)
                {
                    BSE_Util::move_R_tensor(Cs_in, i, j, R_original, R);
                    BSE_Util::move_R_tensor(Vs_in, i, j, R_original, R);
                    BSE_Util::move_R_tensor(Ws_in, i, j, R_original, R);
                }
            }
        }
    }
    // 1-3. set tensors, in these functions MPI distribution will be performed
    this->LR_lri.set_Cs(Cs_in, info_ri.C_threshold, set_IJ, all_atoms);
    this->LR_lri.set_Vs(Vs_in, info_ri.V_threshold, set_I, set_J);
    this->LR_lri.set_Ws(Ws_in, info_ri.V_threshold, set_I, set_J);


    if (PARAM.inp.out_ri_cv)        // out LR_lri tensors
    {        
        TLRI<T>& Cs_LRI = this->LR_lri.lri.data_pool.at("Cs_").Ds_ab;// see LRI::set_tensor_map2
        LRI_CV_Tools::write_Cs_ao(Cs_LRI, PARAM.globalv.global_out_dir + "Cs_lri_test_" + std::to_string(GlobalV::MY_RANK));
        TLRI<T>& Vs_LRI = this->LR_lri.lri.data_pool.at("Vs_").Ds_ab;
        LRI_CV_Tools::write_Vs_abf(Vs_LRI, PARAM.globalv.global_out_dir + "Vs_lri_test_" + std::to_string(GlobalV::MY_RANK));
        TLRI<T>& Ws_LRI = this->LR_lri.lri.data_pool.at("Ws_").Ds_ab;
        LRI_CV_Tools::write_Vs_abf(Ws_LRI, PARAM.globalv.global_out_dir + "Ws_lri_test_" + std::to_string(GlobalV::MY_RANK));
    }
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "distribute_atom_and_k");
    ModuleBase::timer::tick("MolecularLRI", "distribute_atom_and_k");

    // 2. calculate Csk_ao_mo
    this->map_psi = this->transform_psi_k(this->psi_ks, this->k_list);
    TLRI<T>& Cs_ao = this->LR_lri.lri.data_pool.at("Cs_").Ds_ab;
    TLRIk<T> Csk_ao = cal_Csk_ao(Cs_ao, this->k_list, this->list_IJ);
    this->LR_lri.free_Cs(); // free Cs_ao to save memory
    this->Csk_ao_mo = cal_Csk_ao_mo(ucell, Csk_ao, this->k_list, this->list_IJ);
    //this->Csk_mo = cal_Csk_mo(ucell, Csk_ao, this->psi_ks, this->k_list, this->list_IJ);
}

template <typename T>
TLRIk<T> MolecularLRI<T>::cal_Csk_ao(const TLRI<T>& CsR_ao,
                                    const std::vector<Tk>& k_list,
                                    const std::vector<TA>& list_IJ)
{
    ModuleBase::TITLE("MolecularLRI", "cal_Csk_ao");
    ModuleBase::timer::tick("MolecularLRI", "cal_Csk_ao");
    TLRIk<T> Csk_ao; // <k, <I, <J, tensor{nabf, nwt1, nwt2}>>>
    for (const auto& k : k_list)
    {
        for (const auto& iat1 : list_IJ)
        {
            Csk_ao[k][iat1]; // initialize
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
    for (const auto& k : k_list)
    {
        auto& Ck_ao = Csk_ao.at(k);
        for (const auto& iat1 : list_IJ)
        {
            auto &Ck_I = Ck_ao.at(iat1);
            auto &CR_I = CsR_ao.at(iat1);
            for (const auto& CI_JR: CR_I)
            {
                const int iat2 = CI_JR.first.first;
                const TC& R = CI_JR.first.second;
                double arg = 2.0 * M_PI * (k[0] * R[0] + k[1] * R[1] + k[2] * R[2]);
                std::complex<double> phase(cos(arg), sin(arg));

                const auto& tensor_ao = CI_JR.second;
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
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "calculate Csk_ao");
    ModuleBase::timer::tick("MolecularLRI", "cal_Csk_ao");
    return Csk_ao;
}

/// @brief calculate Csk_ao_mo by C^\mu (s,m)[k] = C^\mu (s,t)[k] c(m,t)[k]
/// s,t: atom orbital index; m: band index
template <typename T>
TCsk_ao_mo<T> MolecularLRI<T>::cal_Csk_ao_mo(const UnitCell& ucell,
                                      const TLRIk<T>& Csk_ao,
                                      const std::vector<Tk>& k_list,
                                      const std::vector<TA>& list_IJ)
{
    ModuleBase::TITLE("MolecularLRI", "cal_Csk_ao_mo");            
    ModuleBase::timer::tick("MolecularLRI", "cal_Csk_ao_mo");
    const std::size_t nmo = psi_ks.get_nbands();

#ifdef __MKL
    const std::size_t mkl_threads = mkl_get_max_threads();
    std::cout << "MKL threads max: " << mkl_threads << std::endl;
    mkl_set_num_threads(1);
#endif

    // <k, <iat, tesnor{nmo, nao}>>
    std::map<Tk, std::map<TA, RI::Tensor<T>>> Csk_ao_mo;  // C'^\mu (s,m)[k] = C^\mu (s,t)[k] c(m,t)[k]

    for (auto k: k_list)
    {
        auto& Ck_ao_mo = Csk_ao_mo[k];
        for (auto iat : list_IJ){
            Ck_ao_mo[iat];
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
    for (auto k: k_list)
    {
        auto& psi_k = this->map_psi.at(k);        
        auto& Ck_ao_mo = Csk_ao_mo.at(k);
        const std::map<TA, std::map<TA, RI::Tensor<T>>>& Ck_ao = Csk_ao.at(k);
        for (auto iat1 : list_IJ)
        {
            auto& Ck_I = Ck_ao.at(iat1);
            const int it1 = ucell.iat2it[iat1];
            const std::size_t nw1 = ucell.atoms[it1].nw;
            const std::size_t nabf = Ck_I.begin()->second.shape[0];
            auto& tensor_ao_mo = Ck_ao_mo.at(iat1); // C'^\mu (s,m)[k]
            tensor_ao_mo = RI::Tensor<T>({nabf, nw1, nmo}); // initialize to zero
            for (const auto& Ck_IJ: Ck_I)
            {
                const int iat2 = Ck_IJ.first;
                const int it2 = ucell.iat2it[iat2];
                const int nw2 = ucell.atoms[it2].nw;

                const auto& tensor_ao = Ck_IJ.second; // C^\mu (s,t)[k]
                assert(nabf == tensor_ao.shape[0]);
                assert(nw1 == tensor_ao.shape[1]);
                assert(nw2 == tensor_ao.shape[2]);
                const auto& psi_k_J = psi_k.at(iat2); // c(m,t)[k]
                assert(nw2 == psi_k_J.shape[1]);

                // caution: Cs are row-major  (iw2 contiguous)
                // C'(mu,s,m) = C(mu,s,t) c(m,t)         << row-major
                // C'_m_s_mu = (c_t_m)^T (C_t_s_mu)      << col-major
                container::BlasConnector::gemm('T', 'N', nmo, nw1*nabf, nw2,
                                                1.0, psi_k_J.ptr(), nw2,
                                                tensor_ao.ptr(), nw2,
                                                1.0, tensor_ao_mo.ptr(), nmo);

            }
        }
    }
#ifdef __MKL
    mkl_set_num_threads(mkl_threads);
#endif
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "cal_Csk_ao_mo");
    ModuleBase::timer::tick("MolecularLRI", "cal_Csk_ao_mo");
    return Csk_ao_mo;
}

/// @brief transform psi to <k, <iat, tensor{nmo, iat.nw}>>, mo is not sliced
template <typename T>
std::map<Tk, std::map<TA, RI::Tensor<T>>>
MolecularLRI<T>::transform_psi_k(const psi::Psi<T>& psi_ks, const std::vector<Tk>& k_list)
{
    ModuleBase::TITLE("MolecularLRI", "transform_psi_k");
    ModuleBase::timer::tick("MolecularLRI", "transform_psi_k");
    std::map<Tk, std::map<TA, RI::Tensor<T>>> psi_map;
    const std::size_t nmo = psi_ks.get_nbands();
    for (const auto& k : k_list) // initialize
    {
        auto& psi_map_k = psi_map[k];
        for (int iat = 0; iat < this->ucell.nat; ++iat)
        {
            psi_map_k[iat];
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
    for (const auto& k : k_list)
    {
        auto& psi_map_k = psi_map.at(k);
        int k_index = this->kpoint_index_map.at(k);
        for (int iat = 0; iat < this->ucell.nat; ++iat)
        {
            const int it = this->ucell.iat2it[iat];
            const std::size_t nw = this->ucell.atoms[it].nw;
            RI::Tensor<T> t({nmo, nw});
            for (int im = 0; im < nmo; ++im)
            {
                for (int iw = 0; iw < nw; ++iw)
                {
                    t(im, iw) = psi_ks(k_index, im, this->ucell.get_iat2iwt()[iat]+iw);
                }
            }
            psi_map_k.at(iat) = std::move(t);
        }
    }
    ModuleBase::timer::tick("MolecularLRI", "transform_psi_k");
    return psi_map;
}


/// ===== Below are functions not used, just reserve for reference =====

/// @brief transform psi to <k, <iat, tensor{nmo, iat.nw}>>, mo is sliced according to imo and nmo
template <typename T>
std::map<Tk, std::map<TA, RI::Tensor<T>>>
MolecularLRI<T>::slice_psi_k(const psi::Psi<T>& psi_ks,
                            const int imo,
                            const std::size_t nmo,
                            const std::vector<Tk>& k_list)
{
    ModuleBase::TITLE("MolecularLRI", "slice_psi_k");
    ModuleBase::timer::tick("MolecularLRI", "slice_psi_k");
    std::map<Tk, std::map<TA, RI::Tensor<T>>> psi_map;
    for (const auto& k : k_list)
    {
        auto& psi_map_k = psi_map[k];
        int k_index = this->kpoint_index_map.at(k);
        for (int iat = 0; iat < this->ucell.nat; ++iat)
        {
            const int it = this->ucell.iat2it[iat];
            const std::size_t nw = this->ucell.atoms[it].nw;
            RI::Tensor<T> t({nmo, nw});
            for (int im = 0; im < nmo; ++im)
            {
                for (int iw = 0; iw < nw; ++iw)
                {
                    t(im, iw) = psi_ks(k_index, imo + im, this->ucell.get_iat2iwt()[iat]+iw);
                }
            }
            psi_map_k[iat] = std::move(t);
            //assert(nmo * nw == LR_Util::print_value(psi_map_k[iat].ptr(), nmo, nw));
        }
        //std::cout<<"Slice psi for k: " << k[0]<<","<<k[1]<<","<<k[2]<<" done."<<std::endl;
    }
    ModuleBase::timer::tick("MolecularLRI", "slice_psi_k");
    return psi_map;
}

/// @brief calculate Csk_mo by 
///         C'^\mu (m1,m2)[k1,k2] = c^*(m1,s)[k1] C^\mu (s,t)[k2] c(m2,t)[k2]
///         C^\mu (m1,m2)[k1,k2] = C'^\mu (m1,m2)[k1,k2] + C'^*\mu (m2,m1)[k2,k1]
/// @note this method is faster than method2, but only valid for m1_list = m2_list, and k1_list = k2_list
template <typename T>
TCsk_mo<T> MolecularLRI<T>::cal_Csk_mo(const UnitCell& ucell,
                                      const TLRIk<T>& Csk_ao,
                                      const psi::Psi<T>& psi_ks,
                                      const std::vector<Tk>& k_list,
                                      const std::vector<TA>& list_IJ)
{
    ModuleBase::TITLE("MolecularLRI", "cal_Csk_mo");
    const std::size_t nmo1 = psi_ks.get_nbands();
    const std::size_t nmo2 = nmo1;
#ifdef __MKL
    const std::size_t mkl_threads = mkl_get_max_threads();
    std::cout << "MKL threads max: " << mkl_threads << std::endl;
    mkl_set_num_threads(1);
#endif
    // <k, <iat, tensor{nmo, iat.nw}>>
    std::map<Tk, std::map<TA, RI::Tensor<T>>> psi_k = transform_psi_k(psi_ks, k_list);

    // <{k1, k2}, <iat, tesnor{nabf, nmo1, nmo2}>>
    TCsk_mo<T> Csk_mo_part;  // C'^\mu (m1,m2)[k1,k2] = c^*(m1,s)[k1] C^\mu (s,t)[k2] c(m2,t)[k2]
    TCsk_mo<T> Csk_mo;       // C^\mu (m1,m2)[k1,k2] = C'^\mu (m1,m2)[k1,k2] + C'^*\mu (m2,m1)[k2,k1]
        
    ModuleBase::timer::tick("MolecularLRI", "cal_Csk_mo_part");
    for (auto k1: k_list)
    {
        for (auto k2: k_list)
        {
            auto& Csk_mo_part_k1_k2 = Csk_mo_part[std::make_pair(k1, k2)];
            auto& Csk_mo_k1_k2 = Csk_mo[std::make_pair(k1, k2)];
            for (auto iat1 : list_IJ){
                Csk_mo_part_k1_k2[iat1];
                Csk_mo_k1_k2[iat1];
            }
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(3)
#endif
    for (auto k1: k_list)
    {
        std::map<TA, RI::Tensor<T>>& psi1_k1 = psi_k.at(k1);
        for (auto k2: k_list)
        {            
            std::map<TA, RI::Tensor<T>>& psi2_k2 = psi_k.at(k2);
            auto& Cmo_part_k1_k2 = Csk_mo_part.at(std::make_pair(k1, k2));
            const std::map<TA, std::map<TA, RI::Tensor<T>>>& C_k2 = Csk_ao.at(k2);
            for (auto iat1 : list_IJ)
            {
                auto& Ck_I = C_k2.at(iat1);
                const int it1 = ucell.iat2it[iat1];
                const int nw1 = ucell.atoms[it1].nw;
                const std::size_t nabf = Ck_I.begin()->second.shape[0];
                Cmo_part_k1_k2.at(iat1) = RI::Tensor<T>({nabf, nmo1, nmo2});
                for (const auto& Ck_IJ: Ck_I)
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
                                                        1.0, psi1_k1.at(iat1).ptr(), nw1,
                                                        ptr, nw2,
                                                        0.0, tmp.data(), nmo1);
                        // C'_m2_m1 = (c_t_m2)^T (tmp_m1_t)^T           << col-major
                        container::BlasConnector::gemm('T', 'T', nmo2, nmo1, nw2,
                                                        1.0, psi2_k2.at(iat2).ptr(), nw2,
                                                        tmp.data(), nmo1,                                                        
                                                        1.0, &Cmo_part_k1_k2[iat1](iabf, 0, 0), nmo2);
                    }
                }
            }
        }
    }
#ifdef __MKL
    mkl_set_num_threads(mkl_threads);
#endif
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "calculate Csk_mo_part");
    ModuleBase::timer::tick("MolecularLRI", "cal_Csk_mo_part");

    // C'^\mu (m1,m2)[k1,k2] is finished, now calculate C^\mu (m1,m2)[k1,k2]
    ModuleBase::timer::tick("MolecularLRI", "cal_Csk_mo_add");

#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(3)
#endif
    for (auto k1: k_list)
    {
        for (auto k2: k_list)
        {
            auto& Cmo_part_k1_k2 = Csk_mo_part.at(std::make_pair(k1, k2));
            auto& Cmo_part_k2_k1 = Csk_mo_part.at(std::make_pair(k2, k1));
            for (auto iat1 : list_IJ)
            {
                RI::Tensor<T> t = Cmo_part_k1_k2.at(iat1).copy();
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
            ModuleBase::TITLE("MolecularLRI", "Csk_mo_add");
        }
    }
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "calculate Csk_mo_add");
    ModuleBase::timer::tick("MolecularLRI", "cal_Csk_mo_add");
    // print C_mo max
    // this->print_Csk_mo_max(Csk_mo, "Csk_mo_" + type_str);
    // this->print_Csk_mo_max(Csk_mo_part, "Csk_mo_part_" + type_str);
    return Csk_mo;
}

/* the first index is contiguous in memory
MO_TYPE: OO   VO    OV    VV    ALL(not used in mo_lri)
nmo1     nocc nocc  nvirt nvirt nocc+nvirt
nmo2     nocc nvirt nocc  nvirt nocc+nvirt
imo1     0    0     nocc  nocc  0
imo2     0    nocc  0     nocc  0
*/
template <typename T>
TCsk_mo<T> MolecularLRI<T>::slice_Csk_mo(const int nocc,
                                         const int nvirt,
                                         const LR_Util::MO_TYPE type,
                                         const std::string type_str,
                                         const std::vector<Tk>& kmo1_list,
                                         const std::vector<Tk>& kmo2_list,
                                         const std::vector<TA>& list_I_or_J)
{
    ModuleBase::TITLE("MolecularLRI", "slice_Cmo");
    ModuleBase::timer::tick("MolecularLRI", "slice_Cmo");

    std::size_t nmo1, nmo2, imo1, imo2;
    switch(type)
    {
    case LR_Util::MO_TYPE::OO:
        nmo1 = nocc; nmo2 = nocc; imo1 = 0; imo2 = 0;
        break;
    case LR_Util::MO_TYPE::VO:
        nmo1 = nocc; nmo2 = nvirt; imo1 = 0; imo2 = nocc;
        break;
    case LR_Util::MO_TYPE::OV:
        nmo1 = nvirt; nmo2 = nocc; imo1 = nocc; imo2 = 0;
        break;
    case LR_Util::MO_TYPE::VV:
        nmo1 = nvirt; nmo2 = nvirt; imo1 = nocc; imo2 = nocc;
        break;
    default:
        throw std::runtime_error("MolecularLRI::cal_CsR_mo: only support OO, VV, VO, OV");
    }

    TCsk_mo<T> slice_Csk_mo;
    for (auto k1: kmo1_list)
    {
        for (auto k2: kmo2_list)
        {
            auto& slice_Cmo_k1_k2 = slice_Csk_mo[std::make_pair(k1, k2)];
            for (auto iat1 : list_I_or_J){ slice_Cmo_k1_k2[iat1]; }
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(3)
#endif
    for (auto k1: kmo1_list)
    {
        for (auto k2: kmo2_list)
        {
            auto& Cmo_k1_k2 = this->Csk_mo.at(std::make_pair(k1, k2));
            auto& slice_Cmo_k1_k2 = slice_Csk_mo.at(std::make_pair(k1, k2));
            for (auto iat1 : list_I_or_J)
            {
                const RI::Tensor<T>& Ck_I = Cmo_k1_k2.at(iat1);
                RI::Tensor<T>& slice_Ck_I = slice_Cmo_k1_k2.at(iat1);

                const std::size_t nabf = Ck_I.shape[0];

                slice_Ck_I = RI::Tensor<T>({nabf, nmo1, nmo2});
                for (std::size_t iabf = 0; iabf < nabf; ++iabf)
                {
                    for (int m1 = 0; m1 < nmo1; ++m1)
                    {
                        for (int m2 = 0; m2 < nmo2; ++m2)
                        {
                            slice_Ck_I(iabf, m1, m2) = Ck_I(iabf, imo1 + m1, imo2 + m2);
                        }
                    }
                }
            }
        }
    }
    
    // print C_mo max
    // this->print_Csk_mo_max(Csk_mo, "Csk_mo_" + type_str);
    ModuleBase::timer::tick("MolecularLRI", "slice_Cmo");
    return slice_Csk_mo;
}

/// @brief calculate Csk_mo by C^\mu (m1,m2)[k1,k2] 
///         = c^*(m1,s)[k1] C^\mu (s,t)[k2] c(m2,t)[k2]
///          +c(m2,s)[k2] C^\mu (s,t)[-k1] c^*(m1,t)[k1]
/// @note this method can calculate different MO1 and MO2, but slower than method1
template <typename T>
TCsk_mo<T> MolecularLRI<T>::cal_Csk_mo_method2(const UnitCell& ucell,
                                               const TLRIk<T>& Csk_ao,
                                               const psi::Psi<T>& psi_ks,
                                               const int nocc,
                                               const int nvirt,
                                               const LR_Util::MO_TYPE type,
                                               const std::string type_str,
                                               const std::vector<Tk>& kmo1_list,
                                               const std::vector<Tk>& kmo2_list)
{
    ModuleBase::TITLE("MolecularLRI", "cal_Csk_mo_method2");
    ModuleBase::timer::tick("MolecularLRI", "cal_Csk_mo_method2");

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
        throw std::runtime_error("MolecularLRI::cal_CsR_mo: only support OO, VV, VO");
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

                Cmo_k1_k2[iat1] = RI::Tensor<T>({nabf, nmo1, nmo2});
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
    ModuleBase::timer::tick("MolecularLRI", "cal_Csk_mo_method2");
    return Csk_mo;
}

template <typename T>
void MolecularLRI<T>::print_Csk_mo_max(const TCsk_mo<T>& Csk_mo, const std::string file_name)
{
    ModuleBase::TITLE("MolecularLRI", "print_Csk_mo_max");
    ModuleBase::timer::tick("MolecularLRI", "print_Csk_mo_max");
    std::ofstream ofs_out(file_name + "_" + std::to_string(GlobalV::MY_RANK));
    for (const auto& C_kpair : Csk_mo)
    {
        const Tk k1_d = C_kpair.first.first;
        const Tk k2_d = C_kpair.first.second;
        ModuleBase::Vector3<double> k1 = RI_Util::array3_to_Vector3(k1_d) * this->ucell.G;
        ModuleBase::Vector3<double> k2 = RI_Util::array3_to_Vector3(k2_d) * this->ucell.G;
        ofs_out << "k1: " << k1 << " k2: " << k2 << std::endl;
        double k1_norm = k1.norm();
        double k2_norm = k2.norm();
        for (const auto& C_kk_I : C_kpair.second)
        {
            const int iat = C_kk_I.first;
            const RI::Tensor<T>& tensor_mo = C_kk_I.second;
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
    ModuleBase::timer::tick("MolecularLRI", "print_Csk_mo_max");
}


}// namespace BSE