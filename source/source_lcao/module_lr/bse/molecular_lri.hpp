//=======================
// AUTHOR : Ziqing Guan
// DATE :   2026-03-22
//=======================
#include "molecular_lri.h"
#include <RI/distribute/Distribute_Equally.h>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <limits>
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
    ModuleBase::TITLE("MolecularLRI", "before_set_Cs");
    this->LR_lri.set_Cs(Cs_in, info_ri.C_threshold, set_IJ, all_atoms);
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "set_Cs");
    ModuleBase::TITLE("MolecularLRI", "before_set_Vs");
    this->LR_lri.set_Vs(Vs_in, info_ri.V_threshold, set_I, set_J);
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "set_Vs");
    ModuleBase::TITLE("MolecularLRI", "before_set_Ws");
    this->LR_lri.set_Ws(Ws_in, info_ri.V_threshold, set_I, set_J);
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "set_Ws");

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
    TLRI<T>& CsR_ao = this->LR_lri.lri.data_pool.at("Cs_").Ds_ab;
    this->Csk_ao_mo = cal_Csk_ao_mo(CsR_ao, this->k_list, this->list_IJ);
    ModuleBase::TITLE("MolecularLRI", "before_free_Cs");
    this->LR_lri.free_Cs(); // free Cs_ao to save memory
    ModuleBase::TITLE("MolecularLRI", "after_free_Cs");
}

/// @brief calculate Csk_ao_mo by C'^\mu (s,m)[k] = C^\mu (s,t)[k] c(m,t)[k]
/// s,t: atom orbital index; m: band index
template <typename T>
TCsk_ao_mo<T> MolecularLRI<T>::cal_Csk_ao_mo(const TLRI<T>& CsR_ao,
                                    const std::vector<Tk>& k_list,
                                    const std::vector<TA>& list_IJ)
{
    ModuleBase::TITLE("MolecularLRI", "cal_Csk_ao_mo");
    ModuleBase::timer::tick("MolecularLRI", "cal_Csk_ao_mo");
    const std::size_t nmo = this->psi_ks.get_nbands();

    using U64 = unsigned long long;
    auto read_proc_status_kb = [](const std::string& key) -> U64 {
        std::ifstream ifs("/proc/self/status");
        std::string line;
        while (std::getline(ifs, line))
        {
            if (line.compare(0, key.size(), key) == 0)
            {
                const std::size_t p = line.find(':');
                if (p == std::string::npos) return 0;
                std::istringstream iss(line.substr(p + 1));
                U64 kb = 0; std::string unit;
                iss >> kb >> unit;
                return kb;
            }
        }
        return 0;
    };
    auto safe_mul = [](const U64 a, const U64 b) -> U64 {
        if (a == 0 || b == 0) return 0;
        if (a > std::numeric_limits<U64>::max() / b) 
            throw std::overflow_error("Multiplication would overflow");
        return a * b;
    };
    U64 prealloc_count = 0;
    U64 prealloc_est_bytes = 0;
    U64 prealloc_max_bytes = 0;
    TCsk_ao_mo<T> Csk_ao_mo; // <k, <I, tensor{nabf, nwt1, nmo}>>>
    for (const auto& k : k_list)
    {
        for (const auto& iat1 : list_IJ)
        {
            const std::size_t nabf = CsR_ao.at(iat1).begin()->second.shape[0];
            const std::size_t nw1 = CsR_ao.at(iat1).begin()->second.shape[1];
            const U64 tensor_bytes = safe_mul(safe_mul(static_cast<U64>(nabf), static_cast<U64>(nw1)),
                                     safe_mul(static_cast<U64>(nmo), static_cast<U64>(sizeof(T))));
            prealloc_count += 1;
            prealloc_est_bytes += tensor_bytes;
            if (tensor_bytes > prealloc_max_bytes) prealloc_max_bytes = tensor_bytes;
            try
            {
                Csk_ao_mo[k][iat1] = RI::Tensor<T>({nabf, nw1, nmo}); // initialize
            }
            catch (const std::bad_alloc&)
            {
                const U64 vmrss_kb = read_proc_status_kb("VmRSS");
                const U64 vmhwm_kb = read_proc_status_kb("VmHWM");
                GlobalV::ofs_running << "[BSE_MEMDBG] bad_alloc(prealloc): rank=" << GlobalV::MY_RANK
                                     << ", tensors=" << prealloc_count
                                     << ", iat1=" << iat1
                                     << ", shape={" << nabf << "," << nw1 << "," << nmo << "}"
                                     << ", tensorMB=" << (tensor_bytes / 1024.0 / 1024.0)
                                     << ", preallocGB=" << (prealloc_est_bytes / 1024.0 / 1024.0 / 1024.0)
                                     << ", VmRSSGB=" << (vmrss_kb / 1024.0 / 1024.0)
                                     << ", VmHWMGB=" << (vmhwm_kb / 1024.0 / 1024.0)
                                     << std::endl;
                throw;
            }
        }
    }
    ModuleBase::TITLE("MolecularLRI", "Csk_ao_mo keys prepared");
    {
        const U64 vmrss_kb = read_proc_status_kb("VmRSS");
        const U64 vmhwm_kb = read_proc_status_kb("VmHWM");
        GlobalV::ofs_running << "[BSE_MEMDBG] prealloc summary: rank=" << GlobalV::MY_RANK
                             << ", tensors=" << prealloc_count
                             << ", estGB=" << (prealloc_est_bytes / 1024.0 / 1024.0 / 1024.0)
                             << ", maxTensorMB=" << (prealloc_max_bytes / 1024.0 / 1024.0)
                             << ", VmRSSGB=" << (vmrss_kb / 1024.0 / 1024.0)
                             << ", VmHWMGB=" << (vmhwm_kb / 1024.0 / 1024.0)
                             << std::endl;
    }
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "Csk_ao_mo keys has been prepared.");

#ifdef __MKL
    const std::size_t mkl_threads = mkl_get_max_threads();
    std::cout << "MKL threads max: " << mkl_threads << std::endl;
    mkl_set_num_threads(1);
#endif
#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
    for (const auto& k : k_list)
    {
        for (const auto& iat1 : list_IJ)
        {
            auto& tensor_ao_mo = Csk_ao_mo.at(k).at(iat1); // C'^\mu (s,m)[k]
            auto& psi_k = this->map_psi.at(k);
            std::map<TA, RI::Tensor<T>> Ck_I_thread;
            auto& CR_I = CsR_ao.at(iat1);
            const std::size_t nabf = CR_I.begin()->second.shape[0];
            const std::size_t nw1 = CR_I.begin()->second.shape[1];
            // FT to get C^\mu (s,t)[k] = sum_R C^\mu (s,t)[R] exp(i k R)
            for (const auto& CI_JR: CR_I)
            {
                const int iat2 = CI_JR.first.first;                
                const TC& R = CI_JR.first.second;
                double arg = 2.0 * M_PI * (k[0] * R[0] + k[1] * R[1] + k[2] * R[2]);
                std::complex<double> phase(cos(arg), sin(arg));
                if (!Ck_I_thread.count(iat2))
                {
                    Ck_I_thread[iat2] = std::move(CI_JR.second * RI::Global_Func::convert<T>(phase));
                }
                else { Ck_I_thread[iat2] += CI_JR.second * RI::Global_Func::convert<T>(phase); }
            }
            // C'^\mu (s,m)[k] = sum_t C^\mu (s,t)[k] c(m,t)[k]
            for (const auto& Ck_IJ: Ck_I_thread)
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
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "calculate Csk_ao_mo");
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
#pragma omp parallel for schedule(static)
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

}// namespace BSE