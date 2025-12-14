#include "hamilt_bse.h"

#include "hamilt_bse_solver.h"
#include "source_lcao/module_gint/temp_gint/gint_interface.h"
#include "source_lcao/module_lr/utils/lr_util_hcontainer.h"

#include <cassert>

namespace BSE
{
template <typename T>
HamiltBSE<T>::HamiltBSE(const int& nspin,
                const int& naos,
                const std::vector<int>& nocc,
                const std::vector<int>& nvirt,
                const UnitCell& ucell_in,
                const std::vector<double>& orb_cutoff_in,
                const Grid_Driver& gd_in,
                const psi::Psi<T>& psi_in,
                const psi::Psi<T>& psi_glb_in,
                const ModuleBase::matrix& eig_gw_in,
                MolecularLRI<T>& mo_lri_in,
                std::weak_ptr<LR::PotHxcLR> pot_in,
                const K_Vectors& kv_in,
                const std::vector<Parallel_2D>& pX_in,// vector for spin, parallel as {nvirt, nocc}
                const Parallel_2D& pc_in, //parallel as {nbasis, nbands}
                const Parallel_Orbitals& pmat_in, //parallel as {nbasis, nbasis}
                const std::vector<std::string>& spin_types_in, //can be singlet and triplet
                const std::string& tda, // can be: "tda", "full", "both"
                const std::string& ri_hartree_benchmark_in)
    : nspin(nspin), naos(naos), nocc(nocc), nvirt(nvirt), ucell(ucell_in),
    orb_cutoff(orb_cutoff_in), gd(gd_in), psi_ks(psi_in), psi_ks_glb(psi_glb_in), eig_gw(eig_gw_in),
    mo_lri(mo_lri_in),
    pot(pot_in), kv(kv_in),
    pX(pX_in), pc(pc_in), pmat(pmat_in),
    spin_types(spin_types_in), ri_hartree_benchmark(ri_hartree_benchmark_in)
{
    ModuleBase::TITLE("BSE", "HamiltBSE");
    if (this->pX[0].get_local_size() == 0) {
        std::cerr<< "Warning: Parallel_2D in RANK "+std::to_string(GlobalV::MY_RANK) +" has no local size, please use less mpi." << std::endl;
        std::cerr<< " [File:"<<__FILE__<< ", Function: " << __FUNCTION__ << ", Line: " << __LINE__ << "]" << std::endl;
    }
    assert(naos == pmat.get_global_row_size() && naos == pmat.get_global_col_size());
    this->nk = this->nspin == 2 ? this->kv.get_nks() / 2 : this->kv.get_nks();
    this->ndim = nk * nocc[0] * nvirt[0];
    this->BSE_A_global.resize(ndim * ndim, 0.0);
    if (tda == "both" || tda == "full") { this->BSE_B_global.resize(ndim * ndim, 0.0); }

    this->DM_trans = LR_Util::make_unique<elecstate::DensityMatrix<T, T>>(&pmat, 1/*nspin*/, kv_in.kvec_d, nk);
    this->DM_trans->set_DMK_zero();
    LR_Util::initialize_DMR(*this->DM_trans, this->pmat, this->ucell, this->gd, this->orb_cutoff);

    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "HamiltBSE is ready to calculate V and W");

    if (PARAM.inp.bse_continue >= 1) {
        this->VA_global.resize( this->ndim * this->ndim, 0.0);
        this->read_AB_matrix("A_V_matrix.dat", this->VA_global.data(), this->ndim, this->ndim);
        ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "read_V_for_A");
    }
    if (PARAM.inp.bse_continue >= 2) {
        this->WA_global.resize( this->ndim * this->ndim, 0.0);
        this->read_AB_matrix("A_W_matrix.dat", this->WA_global.data(), this->ndim, this->ndim);
        ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "read_W_for_A");
    }
    if (PARAM.inp.bse_continue >= 3) {
        this->VB_global.resize( this->ndim * this->ndim, 0.0);
        this->read_AB_matrix("B_V_matrix.dat", this->VB_global.data(), this->ndim, this->ndim);
        ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "read_V_for_B");
    }
    if (PARAM.inp.bse_continue >= 4) {
        this->WB_global.resize( this->ndim * this->ndim, 0.0);
        this->read_AB_matrix("B_W_matrix.dat", this->WB_global.data(), this->ndim, this->ndim);
        ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "read_W_for_B");
    }
    for (const auto& st : this->spin_types) {
        if (st == "singlet" || st == "triplet") {
            // Hartree term V (exchange electron and hole)
            if (st == "singlet"){
                this->cal_V_for_A();
                if (tda == "both" || tda == "full") { this->cal_V_for_B(); }
            }
            else if (st == "triplet") {
                std::cout << "Hatree term is not needed for triplet." << std::endl;
            }

            // direct term W (electron-electron and hole-hole)
            this->cal_W_for_A();
            if (tda == "both" || tda == "full") { this->cal_W_for_B(); }
        }
        else if(st == "rpa") {
            this->cal_V_for_A();
            if (tda == "both" || tda == "full") { this->cal_V_for_B(); }
        }
        else if(st != "ipa") {
            throw std::runtime_error("Unsupported type in BSE: " + st);
        }
    }
}

template <typename T>
void HamiltBSE<T>::cal_V_for_A(){
    ModuleBase::TITLE("HamiltBSE", "cal_V_for_A");
    ModuleBase::timer::tick("HamiltBSE", "cal_V_for_A");
    std::cout<<"in cal_V_for_A"<<std::endl;
    if (! this->VA_global.empty()) {
        std::cout<< "V for A has been calculated, skip." <<std::endl;
        return;
    }
    this->VA_global.resize( this->ndim * this->ndim, 0.0);
    if (this->ri_hartree_benchmark == "aims" || this->ri_hartree_benchmark == "abacus") {
        throw std::runtime_error("this BSE routine only supports aims/abacus-librpa benchmark");
    }
    else if (PARAM.inp.bse_ri_hartree || this->ri_hartree_benchmark =="aims-librpa" || this->ri_hartree_benchmark == "abacus-librpa") {
        std::cout << "Calculating Hartree term for A with RI approximation" << std::endl;
        this->mo_lri.cal_hartree_for_A(this->VA_global);
    }
    else if (this->ri_hartree_benchmark == "none") { // do things like OperatorLRHxc
        std::cout << "Calculating Hartree term for A with grid integration" << std::endl;
        // 1. initialize HContainer VR
        const int& is = 0; //spin index, only support 1 spin now
        const auto psi_is = LR_Util::get_psi_spin(psi_ks, is, nk);
        std::unique_ptr<hamilt::HContainer<T>> VR = std::unique_ptr<hamilt::HContainer<T>>(new hamilt::HContainer<T>(&this->pmat));
        LR_Util::initialize_HR<T, T>(*VR, this->ucell, this->gd, this->orb_cutoff);

        for (int ik2 = 0; ik2 < nk; ++ik2) {                    
            for (int j = 0; j < nocc[0]; ++j) {
                for (int b = 0; b < nvirt[0]; ++b) {//calculate row {aik1} for each column {bjk2}
                    ModuleBase::timer::tick("HamiltBSE", "cal_V_column_by_grid");
                    int bjk = ik2 * nocc[0] * nvirt[0] + j * nvirt[0] + b; // column index in BSE matrix
                    // 2. calculate transition matrix jk2→bk2, D(k)=c_b(k)c^†_j(k)
        #ifdef __MPI
                    ct::Tensor dm_trans_2d = 
                        BSE_Util::cal_dm_trans_onebase_pblas(psi_is, pc, ik2, naos, j, b+nocc[0], pmat, (T)1.0 / (T)nk);
        #else
                    ct::Tensor dm_trans_2d = 
                        BSE_Util::cal_dm_trans_onebase_blas(psi_is, pc, ik2, naos, j, b+nocc[0], (T)1.0 / (T)nk);
        #endif
                    // LR_Util::print_tensor<T>(dm_trans_2d, "dm_trans_2d", &pmat);
                    this->DM_trans->set_DMK_pointer(ik2, dm_trans_2d.data<T>());
                    // 3. D(k)→D(R)
                    this->DM_trans->cal_DMR(ik2);
                    // LR_Util::print_DMR(*DM_trans, ucell.nat, "DMR");

                    // 4. D(R)→V(R)
                    this->grid_calculation(*VR);

                    // 5. V(R)→V(k) 
                    std::vector<ct::Tensor> v_k_2d(nk, LR_Util::newTensor<T>({ pmat.get_col_size(), pmat.get_row_size() }));
                    for (auto& v : v_k_2d) v.zero();
                    int nrow = ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER(PARAM.inp.ks_solver) ? 
                        this->pmat.get_row_size() : this->pmat.get_col_size();
                    for (int ik1 = 0;ik1 < nk;++ik1) {
                        folding_HR(*VR, v_k_2d[ik1].data<T>(), this->kv.kvec_d[ik1], nrow, 1);
                    }
                    // for (int ik1 = 0;ik1 < nk;++ik1)
                    //     LR_Util::print_tensor<T>(v_k_2d[ik1], "V(k)[ik=" + std::to_string(ik1) + "]", &this->pmat);
        #ifdef __MPI
                    std::vector<T> V_col_local( this->nk * this->pX[is].get_local_size(), 0.0); // V_col(bjk2)
                    LR::ao_to_mo_pblas(v_k_2d, this->pmat, psi_is, this->pc, this->naos,
                                    nocc[is], nvirt[is], this->pX[is], V_col_local.data(), false, LR::MO_TYPE::VO);

                    for (int ik1 = 0; ik1 < this->nk; ++ik1) {
                        LR_Util::gather_2d_to_full(this->pX[is],
                            V_col_local.data() + ik1 * this->pX[is].get_local_size(),
                            &this->VA_global[bjk * this->ndim /*col*/ + ik1 * nocc[is] * nvirt[is]/*row*/],
                            false, nvirt[is], nocc[is]);
                    }
        #else
                    LR::ao_to_mo_blas(v_k_2d, psi_is, nocc[is], nvirt[is], this->VA_global.data()+bjk * this->ndim, false, LR::MO_TYPE::VO);
        #endif
                    ModuleBase::timer::tick("HamiltBSE", "cal_V_column_by_grid");
                }
            }
        }
    }
    if (GlobalV::MY_RANK == 0){
        this->write_AB_matrix("A_V_matrix.dat", 6, this->VA_global.data(), this->ndim, this->ndim);
    }
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "cal_V_for_A");
    ModuleBase::timer::tick("HamiltBSE", "cal_V_for_A");
}

template <typename T>
void HamiltBSE<T>::cal_V_for_B(){
    ModuleBase::TITLE("HamiltBSE", "cal_V_for_B");
    ModuleBase::timer::tick("HamiltBSE", "cal_V_for_B");
    std::cout<<"in cal_V_for_B"<<std::endl;
    if (! this->VB_global.empty()) {
        std::cout<< "V for B has been calculated, skip." <<std::endl;
        return;
    }
    this->VB_global.resize( this->ndim * this->ndim, 0.0);
    if (this->ri_hartree_benchmark == "aims" || this->ri_hartree_benchmark == "abacus") {
        throw std::runtime_error("this BSE routine only supports aims/abacus-librpa benchmark");
    }
    else if (PARAM.inp.bse_ri_hartree || this->ri_hartree_benchmark =="aims-librpa" || this->ri_hartree_benchmark == "abacus-librpa") {
        std::cout << "Calculating Hartree term for B with RI approximation" << std::endl;
        this->mo_lri.cal_hartree_for_B(this->VB_global);
    }
    else if (this->ri_hartree_benchmark == "none") { // do things like OperatorLRHxc
        std::cout << "Calculating Hartree term for B with grid integration" << std::endl;
        // 1. initialize HContainer VR
        const int& is = 0; //spin index, only support 1 spin now
        const auto psi_is = LR_Util::get_psi_spin(psi_ks, is, nk);
        std::unique_ptr<hamilt::HContainer<T>> VR = std::unique_ptr<hamilt::HContainer<T>>(new hamilt::HContainer<T>(&this->pmat));
        LR_Util::initialize_HR<T, T>(*VR, this->ucell, this->gd, this->orb_cutoff);

        for (int ik2 = 0; ik2 < nk; ++ik2) {                    
            for (int j = 0; j < nocc[0]; ++j) {
                for (int b = 0; b < nvirt[0]; ++b) {//calculate row {aik1} for each column {bjk2}
                    ModuleBase::timer::tick("HamiltBSE", "cal_V_column_by_grid");
                    int bjk = ik2 * nocc[0] * nvirt[0] + j * nvirt[0] + b; // column index in BSE matrix
                    // 2. calculate transition matrix jk2←bk2, D(k)=c_j(k)c^†_b(k)
        #ifdef __MPI
                    ct::Tensor dm_trans_2d = 
                        BSE_Util::cal_dm_trans_onebase_pblas(psi_is, pc, ik2, naos, b+nocc[0], j, pmat, (T)1.0 / (T)nk);
        #else
                    ct::Tensor dm_trans_2d = 
                        BSE_Util::cal_dm_trans_onebase_blas(psi_is, pc, ik2, naos, b+nocc[0], j, (T)1.0 / (T)nk);
        #endif
                    // LR_Util::print_tensor<T>(dm_trans_2d, "dm_trans_2d", &pmat);
                    this->DM_trans->set_DMK_pointer(ik2, dm_trans_2d.data<T>());
                    // 3. D(k)→D(R)
                    this->DM_trans->cal_DMR(ik2);
                    // LR_Util::print_DMR(*DM_trans, ucell.nat, "DMR");

                    // 4. D(R)→V(R)
                    this->grid_calculation(*VR);

                    // 5. V(R)→V(k) 
                    std::vector<ct::Tensor> v_k_2d(nk, LR_Util::newTensor<T>({ pmat.get_col_size(), pmat.get_row_size() }));
                    for (auto& v : v_k_2d) v.zero();
                    int nrow = ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER(PARAM.inp.ks_solver) ? 
                        this->pmat.get_row_size() : this->pmat.get_col_size();
                    for (int ik1 = 0;ik1 < nk;++ik1) {
                        folding_HR(*VR, v_k_2d[ik1].data<T>(), this->kv.kvec_d[ik1], nrow, 1);
                    }
                    // for (int ik1 = 0;ik1 < nk;++ik1)
                    //     LR_Util::print_tensor<T>(v_k_2d[ik1], "V(k)[ik=" + std::to_string(ik1) + "]", &this->pmat);
        #ifdef __MPI
                    std::vector<T> V_col_local( this->nk * this->pX[is].get_local_size(), 0.0); // V_col(bjk2)
                    LR::ao_to_mo_pblas(v_k_2d, this->pmat, psi_is, this->pc, this->naos,
                                    nocc[is], nvirt[is], this->pX[is], V_col_local.data(), false, LR::MO_TYPE::VO);

                    for (int ik1 = 0; ik1 < this->nk; ++ik1) {
                        LR_Util::gather_2d_to_full(this->pX[is],
                            V_col_local.data() + ik1 * this->pX[is].get_local_size(),
                            &this->VB_global[bjk * this->ndim /*col*/ + ik1 * nocc[is] * nvirt[is]/*row*/],
                            false, nvirt[is], nocc[is]);
                    }
        #else
                    LR::ao_to_mo_blas(v_k_2d, psi_is, nocc[is], nvirt[is], this->VB_global.data()+bjk * this->ndim, false, LR::MO_TYPE::VO);
        #endif
                    ModuleBase::timer::tick("HamiltBSE", "cal_V_column_by_grid");
                }
            }
        }
    }
    if (GlobalV::MY_RANK == 0){
        this->write_AB_matrix("B_V_matrix.dat", 6, this->VB_global.data(), this->ndim, this->ndim);
    }
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "cal_V_for_B");
    ModuleBase::timer::tick("HamiltBSE", "cal_V_for_B");
}

template <typename T>
void HamiltBSE<T>::cal_W_for_A(){
    ModuleBase::TITLE("HamiltBSE", "cal_W_for_A");
    ModuleBase::timer::tick("HamiltBSE", "cal_W_for_A");
    std::cout<<"in cal_W_for_A"<<std::endl;
    if (! this->WA_global.empty()) {
        std::cout<< "W for A has been calculated, skip." <<std::endl;
        return;
    }
    this->WA_global.resize( this->ndim * this->ndim, 0.0);
    
    // BSE::MolecularWR<T> WR(this->ucell, this->naos, this->nk, this->kv, this->nocc[0], this->nvirt[0],
    //                 this->psi_ks_glb, this->LR_lri);
    // WR.cal_W_global(this->WA_global);

    this->mo_lri.cal_W_for_A(this->WA_global);    
    if (GlobalV::MY_RANK == 0){
        this->write_AB_matrix("A_W_matrix.dat", 6, this->WA_global.data(), this->ndim, this->ndim);
    }
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "cal_W_for_A");
    ModuleBase::timer::tick("HamiltBSE", "cal_W_for_A");
}

template <typename T>
void HamiltBSE<T>::cal_W_for_B(){
    ModuleBase::TITLE("HamiltBSE", "cal_W_for_B");
    ModuleBase::timer::tick("HamiltBSE", "cal_W_for_B");
    std::cout<<"in cal_W_for_B"<<std::endl;
    if (! this->WB_global.empty()) {
        std::cout<< "W for B has been calculated, skip." <<std::endl;
        return;
    }
    this->WB_global.resize( this->ndim * this->ndim, 0.0);
    this->mo_lri.cal_W_for_B(this->WB_global);
    
    if (GlobalV::MY_RANK == 0){
        this->write_AB_matrix("B_W_matrix.dat", 6, this->WB_global.data(), this->ndim, this->ndim);
    }
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "cal_W_for_B");
    ModuleBase::timer::tick("HamiltBSE", "cal_W_for_B");
}


template <typename T>
void HamiltBSE<T>::init_bse_matrix(const bool is_full, const int & st_index){
    ModuleBase::TITLE("HamiltBSE", "init_bse_matrix");
    ModuleBase::timer::tick("HamiltBSE", "init_bse_matrix");

    std::fill(this->BSE_A_global.begin(), this->BSE_A_global.end(), 0.0);
    if (this->VA_global.empty()){
        std::cout<<"A_V matrix is not calculated, fill zero now!"<<std::endl;
        this->VA_global.resize( this->ndim * this->ndim, 0.0);
    }
    if (this->WA_global.empty()){
        std::cout<<"A_W matrix is not calculated, fill zero now!"<<std::endl;
        this->WA_global.resize( this->ndim * this->ndim, 0.0);
    }
    if (is_full) {
        std::fill(this->BSE_B_global.begin(), this->BSE_B_global.end(), 0.0);
        if (this->VB_global.empty()){
            std::cout<<"B_V matrix is not calculated, fill zero now!"<<std::endl;
            this->VB_global.resize( this->ndim * this->ndim, 0.0);
        }
        if (this->WB_global.empty()){
            std::cout<<"B_W matrix is not calculated, fill zero now!"<<std::endl;
            this->WB_global.resize( this->ndim * this->ndim, 0.0);
        }
    }

    double alpha, beta;
    const std::string& st = this->spin_types[st_index];
    if (st == "singlet") {
        alpha = 2.0; beta = -1.0;
    }
    else if (st == "triplet") {
        alpha = 0.0; beta = -1.0;
    }
    else if (st == "rpa") {
        alpha = 2.0; beta = 0.0;
    }
    else if (st == "ipa") {
        alpha = 0.0; beta = 0.0;
    }
    else {
        throw std::runtime_error("Unsupported type in BSE: " + st);
    }
    
    std::string tda_type = is_full ? "full" : "TDA";
    std::cout<<"| init "<< tda_type << " BSE for type: "<<this->spin_types[st_index]<<std::endl;
    std::cout<<"| A(ai,bj) = (Ea-Ei) δ_ij δ_ab + alpha (ai|V|jb)  +  beta (ji|W|ab)" << std::endl;
    std::cout<<"|   term coefficient: (Exchange) alpha: "<<std::setw(2)<<alpha<<", (Direct) beta: "<<beta<<std::endl;
    if (is_full) {
        std::cout<<"| B(ai,jb) = alpha (ai|V|bj)  +  beta (bi|W|aj)" << std::endl;
    }
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for(int ik = 0;ik < nk;++ik)
    {
        for (int i = 0;i < nocc[0];++i)
        {
            for(int a = 0;a < nvirt[0];++a)
            {
                int index = ik * nocc[0] * nvirt[0] + i * nvirt[0] + a;
                this->BSE_A_global[index * ndim + index] = this->eig_gw(ik, nocc[0] + a) - this->eig_gw(ik, i);
            }
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < this->BSE_A_global.size(); ++i) {
        this->BSE_A_global[i] +=(alpha * this->VA_global[i] + beta * this->WA_global[i]);
        if (is_full) { this->BSE_B_global[i] +=(alpha * this->VB_global[i] + beta * this->WB_global[i]); }
    }

    constexpr double threshold = 1.0e-6;
    if (LR_Util::is_hermitian(this->BSE_A_global.data(), this->ndim, threshold)) {
        if (GlobalV::MY_RANK == 0) {
            std::cout << "|  CHECK PASS: Matrix A is hermitian under threshold " << threshold << std::endl;
        }
    }
    else {
        std::cout << "| Matrix A is not hermitian under threshold " << threshold << std::endl;
    }
    if (GlobalV::MY_RANK == 0){
        this->write_AB_matrix("A_matrix.dat", 6, this->BSE_A_global.data(), this->ndim, this->ndim);
    }

    if (is_full) {
        if (LR_Util::is_symmetric(this->BSE_B_global.data(), this->ndim, threshold)) {
            if (GlobalV::MY_RANK == 0) {
                std::cout << "| CHECK PASS: Matrix B is symmetric under threshold " << threshold << std::endl;
            }
        }
        else { std::cout << "|  CHECK WARNING: Matrix B is not symmetric under threshold " << threshold << std::endl; }
        if (GlobalV::MY_RANK == 0){
            this->write_AB_matrix("B_matrix.dat", 6, this->BSE_B_global.data(), this->ndim, this->ndim);
        }
    }

    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "init_bse_matrix");
    ModuleBase::timer::tick("HamiltBSE", "init_bse_matrix");
}

template <typename T>
void HamiltBSE<T>::tda_solver(const int & st_index, const int& nstates, double* ene_out, T* X_out){
    ModuleBase::TITLE("HamiltBSE", "tda_solver");
    ModuleBase::timer::tick("HamiltBSE", "tda_solver");

    std::fill(this->BSE_A_global.begin(), this->BSE_A_global.end(), 0.0);

    std::cout<<"solve tda for spin type: "<<this->spin_types[st_index]<<std::endl;
    std::vector<T> global_X_tda(this->ndim * this->ndim, 0.0);
    std::vector<double> ev(this->ndim, 0.0);

    this->init_bse_matrix(false, st_index);

    // this->pA.init(ndim, ndim, 1/*nb*/, MPI_COMM_WORLD, false/*dim0<dim1*/);
    LR_Util::setup_2d_division(this->pA, 1/*nb*/, ndim, ndim
        #ifdef __MPI
                , this->pX[0].blacs_ctxt
        #endif
            );

    BSE::solve_tda(GlobalV::MY_RANK,
                    this->BSE_A_global,
                    this->pA,
                    this->ndim,
                    ev,
                    global_X_tda);
    // copy to output
    std::copy_n(ev.data(), nstates, ene_out);
    LR_Util::global2local_X(X_out, global_X_tda.data(), nstates, this->nk,
                            this->nocc, this->nvirt, this->pX, false/*openshell*/);
    
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "BSE TDA solver");
    ModuleBase::timer::tick("HamiltBSE", "tda_solver");
}

template <>
void HamiltBSE<double>::full_solver(const int& st_index, const int& nstates,
                                    double* ene_out,
                                    double* X_out,
                                    double* Y_out){
    ModuleBase::TITLE("HamiltBSE", "full_solver(double)");
    ModuleBase::timer::tick("HamiltBSE", "full_solver(double)");

    this->init_bse_matrix(true, st_index);

    // convert to complex
    std::vector<std::complex<double>> BSE_A_global_complex = BSE_Util::to_complex(this->BSE_A_global);
    std::vector<std::complex<double>> BSE_B_global_complex = BSE_Util::to_complex(this->BSE_B_global);
    std::vector<std::complex<double>> global_v_full(4 * this->ndim * this->ndim, 0.0);
    std::vector<double> ev(2 * this->ndim, 0.0);
    BSE::solve_full(GlobalV::MY_RANK,
                    BSE_A_global_complex,
                    BSE_B_global_complex,
                    this->ndim,
                    ev,
                    global_v_full);

    // copy positive eigenvalues
    std::vector<double> global_X_full(this->ndim * nstates, 0.0);
    std::vector<double> global_Y_full(this->ndim * nstates, 0.0);
    for (int i = 0; i < this->ndim; ++i) {
        assert(ev[i+this->ndim] >= 0.0);
        for (int j = 0; j < this->ndim; ++j) {
            assert(std::abs(global_v_full[(i+this->ndim)*2*this->ndim + j].imag()) < 1e-10);
            assert(std::abs(global_v_full[(i+this->ndim)*2*this->ndim + j+this->ndim].imag()) < 1e-10);
            global_X_full[i * this->ndim + j] = global_v_full[(i+this->ndim)*2*this->ndim + j].real();
            global_Y_full[i * this->ndim + j] = global_v_full[(i+this->ndim)*2*this->ndim + j+this->ndim].real();
        }
    }

    // copy to output
    std::copy_n(&ev[this->ndim], nstates, ene_out);
    LR_Util::global2local_X(X_out, global_X_full.data(), nstates, this->nk,
                            this->nocc, this->nvirt, this->pX, false/*openshell*/);
    LR_Util::global2local_X(Y_out, global_Y_full.data(), nstates, this->nk,
                            this->nocc, this->nvirt, this->pX, false/*openshell*/);

    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "BSE Full solver");
    ModuleBase::timer::tick("HamiltBSE", "full_solver(double)");
}

template <>
void HamiltBSE<std::complex<double>>::full_solver(const int& st_index, const int& nstates,
                                                    double* ene_out,
                                                    std::complex<double>* X_out,
                                                    std::complex<double>* Y_out){
    ModuleBase::TITLE("HamiltBSE", "full_solver(complex)");
    ModuleBase::timer::tick("HamiltBSE", "full_solver(complex)");

    this->init_bse_matrix(true, st_index);

    std::vector<std::complex<double>> global_v_full(4 * this->ndim * this->ndim, 0.0);
    std::vector<double> ev(2 * this->ndim, 0.0);
    BSE::solve_full(GlobalV::MY_RANK,
                    this->BSE_A_global,
                    this->BSE_B_global,
                    this->ndim,
                    ev,
                    global_v_full);

    // copy positive eigenvalues
    std::vector<std::complex<double>> global_X_full(this->ndim * nstates, 0.0);
    std::vector<std::complex<double>> global_Y_full(this->ndim * nstates, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < this->ndim; ++i) {
        assert(ev[i+this->ndim] >= 0.0);
        for (int j = 0; j < this->ndim; ++j) {
            global_X_full[i * this->ndim + j] = global_v_full[(i+this->ndim)*2*this->ndim + j];
            global_Y_full[i * this->ndim + j] = global_v_full[(i+this->ndim)*2*this->ndim + j+this->ndim];
        }
    }

    // copy to output
    std::copy_n(&ev[this->ndim], nstates, ene_out);
    LR_Util::global2local_X(X_out, global_X_full.data(), nstates, this->nk,
                            this->nocc, this->nvirt, this->pX, false/*openshell*/);
    LR_Util::global2local_X(Y_out, global_Y_full.data(), nstates, this->nk,
                            this->nocc, this->nvirt, this->pX, false/*openshell*/);
    
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "BSE FULL solver");
    ModuleBase::timer::tick("HamiltBSE", "full_solver(complex)");
}

template<>
void HamiltBSE<double>::grid_calculation(hamilt::HContainer<double>& VR) const
{
    ModuleBase::TITLE("HamiltBSE", "grid_calculation(double)");
    ModuleBase::timer::tick("HamiltBSE", "grid_calculation(double)");

    // 4.1. transition density rho on grid
    double** rho_trans;
    const int& nrxx = this->pot.lock()->nrxx;

    LR_Util::_allocate_2order_nested_ptr(rho_trans, 1, nrxx); // nspin=1 for transition density
    ModuleBase::GlobalFunc::ZEROS(rho_trans[0], nrxx);
    ModuleGint::cal_gint_rho(this->DM_trans->get_DMR_vector(), 1, rho_trans, false);

    // 4.2. v_hxc = f_hxc * rho_trans
    ModuleBase::matrix vr_hxc(1, nrxx);   //grid
    std::vector<int> ispin_ks = { 0 }; //for close-shell dft-xc kerenl, actually placeholder for bse 
    this->pot.lock()->cal_v_eff(rho_trans, ucell, vr_hxc, ispin_ks);// in this function, unit changes from Ha to Ry
    LR_Util::_deallocate_2order_nested_ptr(rho_trans, 1);

    // 4.3 V^{Hxc}_{\mu,\nu}=\int{dr} \phi_\mu(r) v_{Hxc}(r) \phi_\nu(r)
    VR.set_zero();
    ModuleGint::cal_gint_vl(vr_hxc.c, &VR);
    // LR_Util::print_HR(VR, this->ucell.nat, "VR(real, 2d)");

    ModuleBase::timer::tick("HamiltBSE", "grid_calculation(double)");
}

template<>
void HamiltBSE<std::complex<double>>::grid_calculation(hamilt::HContainer<std::complex<double>>& VR) const
{
    ModuleBase::TITLE("HamiltBSE", "grid_calculation(complex)");
    ModuleBase::timer::tick("HamiltBSE", "grid_calculation(complex)");

    elecstate::DensityMatrix<std::complex<double>, double> DM_trans_real_imag(&this->pmat, 1, this->kv.kvec_d, this->nk);
    DM_trans_real_imag.init_DMR(VR);
    hamilt::HContainer<double> HR_real_imag(ucell, &this->pmat);
    LR_Util::initialize_HR<std::complex<double>, double>(HR_real_imag, ucell, gd, orb_cutoff);

    auto dmR_to_hR = [&, this](const char& type) -> void
        {
            LR_Util::get_DMR_real_imag_part(*this->DM_trans, DM_trans_real_imag, ucell.nat, type);
            // if (this->first_print)LR_Util::print_DMR(DM_trans_real_imag, ucell.nat, "DMR(2d, real)");

            // 4.1. transition density rho on grid
            double** rho_trans;
            const int& nrxx = this->pot.lock()->nrxx;

            LR_Util::_allocate_2order_nested_ptr(rho_trans, 1, nrxx); // nspin=1 for transition density
            ModuleBase::GlobalFunc::ZEROS(rho_trans[0], nrxx);
            ModuleGint::cal_gint_rho(DM_trans_real_imag.get_DMR_vector(), 1, rho_trans, false);

            // 4.2. v_hxc = f_hxc * rho_trans
            ModuleBase::matrix vr_hxc(1, nrxx);   //grid
            std::vector<int> ispin_ks = { 0 }; //for close-shell dft-xc kerenl, actually placeholder for bse 
            this->pot.lock()->cal_v_eff(rho_trans, ucell, vr_hxc, ispin_ks);// in this function, unit changes from Ha to Ry
            LR_Util::_deallocate_2order_nested_ptr(rho_trans, 1);

            // 4.3 V^{Hxc}_{\mu,\nu}=\int{dr} \phi_\mu(r) v_{Hxc}(r) \phi_\nu(r)
            HR_real_imag.set_zero();
            ModuleGint::cal_gint_vl(vr_hxc.c, &HR_real_imag);
            // LR_Util::print_HR(HR_real_imag, this->ucell.nat, "VR(real, 2d)");
            LR_Util::set_HR_real_imag_part(HR_real_imag, VR, ucell.nat, type);
        };
    VR.set_zero();
    dmR_to_hR('R');   //real
    if (this->nk > 1) { dmR_to_hR('I'); }   //imag for multi-k
    ModuleBase::timer::tick("HamiltBSE", "grid_calculation(complex)");
}

template class HamiltBSE<double>;
template class HamiltBSE<std::complex<double>>;
}//namespace BSE