#include "bse.h"
// dm_trans in this file is simplified as single excitation j→b
dm_trans[isk] = c c

namespace BSE
{
    template <typename T>
    class BSE_Matrix
    {
    public:
        cal_Matrix(const int& nspin,//only for spin=2
                   const int& naos,
                   const std::vector<int>& nocc,
                   const std::vector<int>& nvirt,
                   const UnitCell& ucell_in,
                   const psi::Psi<T>& psi_ks_in,
                   const Modubase::matrix& eig_gw,
                   std::weak_ptr<Exx_LRI<T>> exx_lri_in,
                   const K_Vectors& kv_in,
                   const Parallel_2D& pX_in,// vector for spin
                   const Parallel_2D& pc_in,
                   const Parallel_Orbitals& pmat_in,
                   const std::string& spin_type,
                   const std::string& ri_hartree_benchmark = "none",
                   const std::vector<int>& aims_nbasis = {})
            : nspin(nspin), nocc(nocc), nvirt(nvirt), nk(kv_in.get_nks() / nspin)
        {
            ModuleBase::TITLE("BSE", "BSE_Matrix");

            int ndim = nk * nocc[0] * nvirt[0];
            this->BSE_A_global.resize( ndim * ndim, 0.0);

            // this->pA.init(ndim, ndim, 1/*nb*/, MPI_COMM_WORLD, false/*dim0<dim1*/);
            LR_Util::setup_2d_division(pA, /*nb=*/1, ndim, ndim);
            this->BSE_A_local.resize(pA.get_local_size() * pA.get_local_size(), 0.0);
            
            // 1.add the diag part            
            #pragma omp parallel for
            for(int ik = 0;ik < nk;++ik)
            {
                for (int i = 0;i < nocc[0];++i)
                {
                    for(int a = 0;a < nvirt[0];++a)
                    {
                        int index = ik * nocc[0] * nvirt[0] + i * nvirt[0] + a;
                        this->BSE_A_global[index * ndim + index] = eig_gw(ik, i) - eig_gw(ik, nocc[0] + a);
                    }
                }
            }
            // 2.add the exchange term V
            this->add_V();
            // 3.add the direct term W
            this->add_W();
        }

        void add_V(){
            if (ri_hartree_benchmark == "aims"){
                throw std::runtime_error("this BSE routine is not supported for aims benchmark");
            }
            else if (ri_hartree_benchmark =="aims-librpa") {
                assert(!aims_nbasis.empty());
            
            }
            else if (ri_hartree_benchmark == "none") {
                LR_Util::initialize_DMR(*this->DM_trans, pmat_in, ucell_in, gd_in, orb_cutoff); 
                // always use nspin=1 for transition density matrix
                this->DM_trans = LR_Util::make_unique<elecstate::DensityMatrix<T, T>>(&pmat_in, 1, kv_in.kvec_d, nk);
            }
        }

        void add_W(){
            
            this->exx_lri = std::make_shared<Exx_LRI<T>>(exx_info.info_ri);
            this->exx_lri->init(MPI_COMM_WORLD, ucell,this->kv, ks_sol.orb_);
            std::cout << "check bse_ri_pca_threshold: " << this->exx_info.info_ri.pca_threshold << std::endl;
            std::cout << "check bse_ri_ccp_rmesh_times: " << this->exx_info.info_ri.ccp_rmesh_times << std::endl;
            std::cout << "FISH_output: prepare W matrix for BSE in esolver_lrtd_lcao.cpp 1" << std::endl;
            this->exx_lri->cal_exx_ions(ucell,input.out_ri_cv, true); //ture for read_W
        }

        void global2local(T* lvec, const T* gvec, const int& nband) const
        {
            const int npairs = nocc[0] * nvirt[0];
            for (int ib = 0;ib < nband;++ib)
            {
                const int loffset_b = ib * nk * pX[0].get_local_size();
                const int goffset_b = ib * nk * npairs;
                for (int ik = 0;ik < nk;++ik)
                {
                    const int loffset = loffset_b + ik * pX[0].get_local_size();
                    const int goffset = goffset_b + ik * npairs;
                    for (int lo = 0;lo < pX[0].get_col_size();++lo)
                    {
                        const int go = pX[0].local2global_col(lo);
                        for (int lv = 0;lv < pX[0].get_row_size();++lv)
                        {
                            const int gv = pX[0].local2global_row(lv);
                            lvec[loffset + lo * pX[0].get_row_size() + lv] = gvec[goffset + go * nvirt[0] + gv];
                        }
                    }
                }
            }
        }
    
    hamilt::Operator<T>* bse_W = new OperatorLREXX<T>(nspin,
                                                        naos,
                                                        nocc[0],
                                                        nvirt[0],
                                                        ucell_in,
                                                        psi_ks_in,
                                                        this->DM_trans,
                                                        exx_lri_in,
                                                        kv_in,
                                                        pX_in[0],
                                                        pc_in,
                                                        pmat_in,
                                                        1.0, // alpha
                                                        aims_nbasis);
    }
}