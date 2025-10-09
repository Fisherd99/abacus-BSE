#include "source_lcao/module_lr/bse/esolver_bse_lcao.h"

namespace BSE
{
using TA = int;
using Tcell = int;
using TC = std::array<Tcell, 3>;
using TAC = std::pair<TA, TC>;
using TatomR = std::array<double, 3>;
template <typename T>
using TLRI = std::map<int, std::map<TAC, RI::Tensor<T>>>;

template <typename T, typename TR>
ESolver_BSE<T, TR>::ESolver_BSE(const Input_para& inp, UnitCell& ucell) :
    LR::ESolver_LR<T, TR>(inp, ucell, GlobalC::exx_info)
{
    ModuleBase::TITLE("ESolver_BSE", "ESolver_BSE(from scratch)");

    // xc kernel
    this->xc_kernel = LR_Util::tolower(inp.xc_kernel);

    // necessary steps in ESolver_FP
    ModuleESolver::ESolver_FP::before_all_runners(ucell, inp);
    this->pelec = new elecstate::ElecStateLCAO<T>();

    // necessary steps in ESolver_KS::before_all_runners : symmetry and k-points
    if (ModuleSymmetry::Symmetry::symm_flag == 1)
    {
        ucell.symm.analy_sys(ucell.lat, ucell.st, ucell.atoms, GlobalV::ofs_running);
        ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "SYMMETRY");
    }
    this->kv.set(ucell, ucell.symm, PARAM.inp.kpoint_file, PARAM.inp.nspin, ucell.G, ucell.latvec, GlobalV::ofs_running);
    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "INIT K-POINTS");
    ModuleIO::setup_parameters(ucell, this->kv);

    this->parameter_check();

    /// read orbitals and build the interpolation table
    this->two_center_bundle_.build_orb(ucell.ntype, ucell.orbital_fn.data());

    LCAO_Orbitals orb;
    this->two_center_bundle_.to_LCAO_Orbitals(orb, inp.lcao_ecut, inp.lcao_dk, inp.lcao_dr, inp.lcao_rmax);
    this->orb_cutoff_ = orb.cutoffs();
    if (LR_Util::tolower(this->input.abs_gauge) == "velocity")
    {
        this->setup_2center_table(this->two_center_bundle_, orb, ucell);
    }

    this->set_dimension();
    //  setup 2d-block distribution for AO-matrix and KS wfc
    LR_Util::setup_2d_division(this->paraMat_, 1, this->nbasis, this->nbasis);
#ifdef __MPI
    this->paraMat_.set_desc_wfc_Eij(this->nbasis, this->nbands, this->paraMat_.get_row_size());
    int err = this->paraMat_.set_nloc_wfc_Eij(this->nbands, GlobalV::ofs_running, GlobalV::ofs_warning);
    this->paraMat_.set_atomic_trace(ucell.get_iat2iwt(), ucell.nat, this->nbasis);
#else
    this->paraMat_.nrow_bands = this->nbasis;
    this->paraMat_.ncol_bands = this->nbands;
#endif

    this->psi_ks = new psi::Psi<T>(this->kv.get_nks(),
                                   this->paraMat_.ncol_bands,
                                   this->paraMat_.get_row_size(),
                                   this->kv.ngk,
                                   true);
    this->read_ks_wfc();
    // FISH_NOTE: openshell is not implemented in BSE
    if (this->nspin == 2)
    {
        this->nupdown = this->cal_nupdown_form_occ(this->pelec->wg);
        this->reset_dim_spin2();
    }

    LR_Util::setup_2d_division(this->paraC_, this->paraMat_.get_block_size(), this->nbasis, this->nbands
#ifdef __MPI
            , this->paraMat_.blacs_ctxt
#endif
        );

    // read the ground state charge density and calculate xc kernel
    this->Pgrid.init(this->pw_rho->nx,
               this->pw_rho->ny,
               this->pw_rho->nz,
               this->pw_rho->nplane,
               this->pw_rho->nrxx,
               this->pw_big->nbz,
               this->pw_big->bz);
    Charge chg_gs;
    if (this->input.ri_hartree_benchmark == "none")
    {
        this->read_ks_chg(chg_gs);
    }
    this->init_pot(chg_gs);

    // search adjacent atoms and init Gint
    double search_radius = -1.0;
    search_radius = atom_arrange::set_sr_NL(GlobalV::ofs_running,
                                            PARAM.inp.out_level,
                                            orb.get_rcutmax_Phi(),
                                            ucell.infoNL.get_rcutmax_Beta(),
                                            PARAM.globalv.gamma_only_local);
    atom_arrange::search(PARAM.globalv.search_pbc,
                         GlobalV::ofs_running,
                         this->gd,
                         this->ucell,
                         search_radius,
                         PARAM.inp.test_atom_input);
    // new_GINT
    this->gint_info_.reset(new ModuleGint::GintInfo(this->pw_big->nbx,
                                              this->pw_big->nby,
                                              this->pw_big->nbz,
                                              this->pw_rho->nx,
                                              this->pw_rho->ny,
                                              this->pw_rho->nz,
                                              0,
                                              0,
                                              this->pw_big->nbzp_start,
                                              this->pw_big->nbx,
                                              this->pw_big->nby,
                                              this->pw_big->nbzp,
                                              orb.Phi,
                                              ucell,
                                              this->gd));
    ModuleGint::Gint::set_gint_info(this->gint_info_.get());
    // new_GINT
    // read 2-center integral and calculate Cs, Vs
    if (this->input.lr_solver != "spectrum")
    {
        this->exx_info.info_global.ccp_type = Conv_Coulomb_Pot_K::Ccp_Type::Hf;
        this->exx_info.info_global.hybrid_alpha = 1;
        this->exx_info.info_ri.ccp_rmesh_times = 10;
        // code below is for origianl `cal_exx_ions`
        // this->exx_info.info_global.coulomb_param[Conv_Coulomb_Pot_K::Coulomb_Type::Fock].resize(1);
        // this->exx_info.info_global.coulomb_param[Conv_Coulomb_Pot_K::Coulomb_Type::Fock]
        //     = {{{"alpha", "1"}, {"singularity_correction", "spencer"}}};

        this->exx_lri = std::make_shared<Exx_LRI<T>>(this->exx_info.info_ri);
        std::cout << "check bse_ri_pca_threshold: " << this->exx_info.info_ri.pca_threshold << std::endl;
        std::cout << "check bse_ri_ccp_rmesh_times: " << this->exx_info.info_ri.ccp_rmesh_times << std::endl;
        this->exx_init();
    }
}

template<typename T, typename TR>
void ESolver_BSE<T, TR>::exx_init()
{
    // ======= exx_lri->init() without constructing abfs =======
    this->exx_lri->mpi_comm = MPI_COMM_WORLD;
    this->exx_lri->p_kv = &this->kv;
    // ======= exx_lri->init() without constructing abfs =======

    // do things similar to `cal_exx_ions` but read Ws and Cs from file
    std::vector<TA> atoms(this->ucell.nat);
    for (int iat = 0; iat < this->ucell.nat; ++iat)
    {
        atoms[iat] = iat;
    }
    std::map<TA, TatomR> atoms_pos;
    for (int iat = 0; iat < this->ucell.nat; ++iat)
    {
        atoms_pos[iat] = RI_Util::Vector3_to_array3(this->ucell.atoms[this->ucell.iat2it[iat]].tau[this->ucell.iat2ia[iat]]);
    }
    const std::array<TatomR, 3> latvec = {RI_Util::Vector3_to_array3(this->ucell.a1),
                                             RI_Util::Vector3_to_array3(this->ucell.a2),
                                             RI_Util::Vector3_to_array3(this->ucell.a3)};
    const std::array<Tcell, 3> period = {this->kv.nmp[0], this->kv.nmp[1], this->kv.nmp[2]};
    this->exx_lri->exx_lri.set_parallel(MPI_COMM_WORLD, atoms_pos, latvec, period);
    // const std::array<Tcell,Ndim> period_Vs = LRI_CV_Tools::cal_latvec_range<Tcell>(1+this->info.ccp_rmesh_times,
    // ucell, orb_cutoff_); 
    // const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA,std::array<Tcell,Ndim>>>>>
    //     list_As_Vs = RI::Distribute_Equally::distribute_atoms_periods(this->mpi_comm, atoms, period_Vs, 2, false);

    // anaylyze c(k) -> c(R), will develop sparse c(R)
    std::vector<std::array<Tcell, 3>> BvK_cells = RI_Util::get_Born_von_Karmen_cells(period);
    std::map<int, std::map<TAC, RI::Tensor<T>>> psi_R; // <mo, <{at, R}, tensor{ao.nw}>
    for (int im = 0; im < this->nbands; ++im)
    {
        for (int iat = 0; iat < this->ucell.nat; ++iat)
        {
            const int it = this->ucell.iat2it[iat];
            for (auto cell: BvK_cells)
            {
                psi_R[im][std::make_pair(iat, cell)] = RI::Tensor<T>({static_cast<size_t>(this->ucell.atoms[it].nw)});
            }
        }
    }

    for (auto cell: BvK_cells)
    {
        ModuleBase::Vector3<double> R = RI_Util::array3_to_Vector3(cell) * this->ucell.latvec;
        for (int ik = 0; ik < this->kv.get_nks(); ++ik)
        {
            std::complex<double> phase
                = std::exp(-ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT * (this->kv.kvec_c.at(ik) * R))
                  / static_cast<double>(this->kv.get_nks());
            for (int im = 0; im < this->nbands; ++im)
            {
                for (int it = 0; it < this->ucell.ntype; ++it)
                {
                    for (int ia = 0; ia < this->ucell.atoms[it].na; ++ia)
                    {
                        const int iat = this->ucell.itia2iat(it, ia);
                        auto& t = psi_R[im][std::make_pair(iat, cell)];
                        const int nw = this->ucell.atoms[it].nw;
                        for (int iw = 0; iw < nw; ++iw)
                        {
                            add_c(t(iw), (*this->psi_ks)(ik, im, this->ucell.iat2ia[iat]), phase);
                        }
                    }
                }
            }
        }
        double R_norm = (R).norm();
        for (int im = 0; im < this->nbands; ++im)
        {
            for (int iat = 0; iat < this->ucell.nat; ++iat)
            {
                const int it = this->ucell.iat2it[iat];
                const int nw = this->ucell.atoms[it].nw;
                double max = -1.0;
                auto& t = psi_R[im][std::make_pair(iat, cell)];
                for (size_t iw = 0; iw < nw; ++iw)
                {
                    if (std::abs(t(iw)) > max)
                    {
                        max = std::abs(t(iw));
                    }
                }
                std::cout << "|c(R)|::IM: " << im << " , IAT: " << iat << " , R: " << R_norm << " ,max: " << max
                          << std::endl;
            }
        }
    }
    // anaylyze c(k) -> c(R)

    // start read Ws and Cs
    std::cout << "prepare W matrix for BSE in ESolver_BSE(from scratch)" << std::endl;
    BSE_IO::RI_kRlist kRlist("stru_out", this->ucell);
    std::map<TA, std::map<TAC, RI::Tensor<T>>> Cs_in = LRI_CV_Tools::read_Cs_ao<T>("Cs_data_0.txt");
    std::map<TA, std::map<TAC, RI::Tensor<TR>>> Vs_in;
    if (this->input.ri_hartree_benchmark == "aims-librpa" ){
        Vs_in = RI_Benchmark::read_coulomb_mat_general<T, TR>("coulomb_mat_0.txt", Cs_in, kRlist);
    }
    else if (this->input.ri_hartree_benchmark == "none" || this->input.ri_hartree_benchmark == "abacus-librpa" ){
        Vs_in = RI_Benchmark::read_coulomb_mat<T, TR>("coulomb_mat_0.txt", Cs_in, kRlist);
    }
    
    // LRI_CV_Tools::write_Vs_abf(Vs_in, PARAM.globalv.global_out_dir + "Vs_test_" + std::to_string(GlobalV::MY_RANK));
    std::map<TA, std::map<TAC, RI::Tensor<T>>> Ws_in = BSE_IO::read_Ws<T, TR>(Vs_in, kRlist.Rlist);
    this->exx_lri->exx_lri.set_Vs(std::move(Ws_in), this->exx_lri->info.V_threshold);
    this->exx_lri->exx_lri.set_Cs(std::move(Cs_in), this->exx_lri->info.C_threshold);
    if (this->input.out_ri_cv)
    {
        LRI_CV_Tools::write_Vs_abf(Ws_in, PARAM.globalv.global_out_dir + "Ws_test_" + std::to_string(GlobalV::MY_RANK));
        LRI_CV_Tools::write_Cs_ao(Cs_in, PARAM.globalv.global_out_dir + "Cs_test_" + std::to_string(GlobalV::MY_RANK));
    }
}

template <typename T, typename TR>
void ESolver_BSE<T, TR>::runner(UnitCell& ucell, const int istep)
{
    ModuleBase::TITLE("ESolver_BSE", "runner");
    ModuleBase::timer::tick("ESolver_BSE", "runner");
    //allocate 2-particle state and setup 2d division
    this->allocate_eigen_infos();

    auto efile_out = [&](const std::string& label)->std::string {return PARAM.globalv.global_out_dir + "Excitation_Energy_" + label + ".dat";};
    auto vfile_out = [&](const std::string& label)->std::string {return PARAM.globalv.global_out_dir + "Excitation_Amplitude_" + label + "_" + std::to_string(GlobalV::MY_RANK) + ".dat";};
    auto efile_in = [&](const std::string& label)->std::string {return PARAM.globalv.global_readin_dir + "Excitation_Energy_" + label + ".dat";};
    auto vfile_in = [&](const std::string& label)->std::string {return PARAM.globalv.global_readin_dir + "Excitation_Amplitude_" + label + "_" + std::to_string(GlobalV::MY_RANK) + ".dat";};

    if (this->input.lr_solver == "elpa")
    {
        std::cout << "Calculating Casida/BSE matrix directly." << std::endl;
        assert(this->xc_kernel == "bse");

        HamiltBSE<T> bse_matrix(this->nspin, this->nbasis, this->nocc, this->nvirt,
                                     this->ucell, this->orb_cutoff_, this->gd, *this->psi_ks, this->eig_gw,
#ifdef __EXX
                                     this->exx_lri,
#endif
                                     /*this->gint_,*/ this->pot[0], this->kv, this->paraX_, this->paraC_, this->paraMat_,
                                     this->input.bse_spin_types,
                                     this->input.bse_tda,
                                     this->input.ri_hartree_benchmark);
        auto write_tda_states = [&](const std::string& label, const Real<T>* e, const T* v, const int& dim, const int& nst, const int& prec = 8)->void
        {
            if (GlobalV::MY_RANK == 0) {
                assert(nst == LR_Util::write_value(efile_out(label), prec, e, nst));
            }
            assert(nst * dim == LR_Util::write_value(vfile_out(label), prec, v, nst, dim));
        };
        auto write_full_states = [&](const std::string& label, const Real<T>* e, const T* X, const T* Y, const int& dim, const int& nst, const int& prec = 8)->void
        {
            if (GlobalV::MY_RANK == 0) {
                assert(nst == LR_Util::write_value(efile_out(label), prec, e, nst));
            }
            assert(nst * dim == LR_Util::write_value(vfile_out("full_X_"+label), prec, X, nst, dim));
            assert(nst * dim == LR_Util::write_value(vfile_out("full_Y_"+label), prec, Y, nst, dim));
        };

        if ((this->input.bse_tda == "both" || this->input.bse_tda == "tda")) {
            for (int is = 0; is < this->input.bse_spin_types.size(); ++is) {
                bse_matrix.tda_solver(is, this->nstates, &this->tda_ene[is * this->nstates], this->X[is].template data<T>());
                if (this->input.out_wfc_lr) {
                    write_tda_states(this->input.bse_spin_types[is], &this->tda_ene[is * this->nstates],
                        this->X[is].template data<T>(), this->nloc_per_state, this->nstates);
                }
            }
        }
        if ((this->input.bse_tda == "both" || this->input.bse_tda == "full")) {
            for (int is = 0; is < this->input.bse_spin_types.size(); ++is) {
                bse_matrix.full_solver(is, this->nstates, &this->full_ene[is * this->nstates],
                    this->full_X[is].template data<T>(), this->full_Y[is].template data<T>());
                if (this->input.out_wfc_lr) {
                    write_full_states(this->input.bse_spin_types[is], &this->full_ene[is * this->nstates],
                        this->full_X[is].template data<T>(), this->full_Y[is].template data<T>(), this->nloc_per_state, this->nstates);
                }
            }
        }
    }
    else if (this->input.lr_solver == "spectrum")
    {
        auto read_tda_states = [&](const std::string& label, Real<T>* e, T* v, const int& dim, const int& nst)->void
        {
            if (GlobalV::MY_RANK == 0) {
                assert(nst == LR_Util::read_value(efile_in(label), e, nst));
                std::cout <<"Rank "<< GlobalV::MY_RANK << ": finish reading " << efile_in(label) << std::endl;
            }
#ifdef __MPI
// in velocity gauge, the eigenvalues are used to calculate the transition dipole, so we'd better broadcast them
            MPI_Bcast(e, nst, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
            assert(nst * dim == LR_Util::read_value(vfile_in(label), v, nst, dim));
            std::cout <<"Rank "<< GlobalV::MY_RANK << ": finish reading " << vfile_in(label) << std::endl;
        };
        auto read_full_states = [&](const std::string& label, Real<T>* e, T* X, T* Y, const int& dim, const int& nst)->void
        {
            if (GlobalV::MY_RANK == 0) {
                assert(nst == LR_Util::read_value(efile_in(label), e, nst));
                std::cout <<"Rank "<< GlobalV::MY_RANK << ": finish reading " << efile_in(label) << std::endl;
            }
#ifdef __MPI
            MPI_Bcast(e, nst, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
            assert(nst * dim == LR_Util::read_value(vfile_in("full_X_"+label), X, nst, dim));
            std::cout <<"Rank "<< GlobalV::MY_RANK << ": finish reading " << vfile_in("full_X_"+label) << std::endl;
            assert(nst * dim == LR_Util::read_value(vfile_in("full_Y_"+label), Y, nst, dim));
            std::cout <<"Rank "<< GlobalV::MY_RANK << ": finish reading " << vfile_in("full_Y_"+label) << std::endl;
        };

        std::cout << "reading the excitation states from file: \n";
        if (this->input.bse_tda == "both" || this->input.bse_tda == "tda") {
            for (int is = 0; is < this->input.bse_spin_types.size(); ++is) {
                read_tda_states(this->input.bse_spin_types[is], &this->tda_ene[is * this->nstates],
                    this->X[is].template data<T>(), this->nloc_per_state, this->nstates);
            }
        }
        if (this->input.bse_tda == "both" || this->input.bse_tda == "full") {
            for (int is = 0; is < this->input.bse_spin_types.size(); ++is) {
                read_full_states(this->input.bse_spin_types[is], &this->full_ene[is * this->nstates],
                    this->full_X[is].template data<T>(), this->full_Y[is].template data<T>(), this->nloc_per_state, this->nstates);
            }
        }
    }
    else
    {
        ModuleBase::WARNING_QUIT("ESolver_BSE", "lr_solver must be elpa or spectrum");
    }
    ModuleBase::timer::tick("ESolver_BSE", "runner");
    return;
}

template <typename T, typename TR>
void ESolver_BSE<T, TR>::after_all_runners(UnitCell& ucell)
{
    ModuleBase::TITLE("ESolver_BSE", "after_all_runners");
    if (this->input.ri_hartree_benchmark != "none") { return; } //no need to calculate the spectrum in the benchmark routine
    if (this->input.bse_tda == "both" || this->input.bse_tda == "tda"){
        for (int is = 0;is < this->X.size();++is)
        {
            LR::LR_Spectrum<T> spectrum(this->nspin, this->nbasis, this->nocc, this->nvirt, this->gint_, *this->pw_rho, *this->psi_ks,
                this->ucell, this->kv, this->gd, this->orb_cutoff_, this->two_center_bundle_,
                this->paraX_, this->paraC_, this->paraMat_,
                &this->tda_ene[is * this->nstates], this->X[is].template data<T>(), this->nstates, false/*openshell*/,
                LR_Util::tolower(this->input.abs_gauge));
            spectrum.transition_analysis(this->input.bse_spin_types[is]);
            if (this->input.bse_spin_types[is] != "triplet")        // triplets has no transition dipole and no contribution to the spectrum
            {
                // spectrum.optical_absorption_method1(freq, input.abs_broadening);
                spectrum.write_transition_dipole(PARAM.globalv.global_out_dir + "transition_dipole.dat");

                if (LR_Util::tolower(this->input.abs_gauge) == "velocity")
                {
                    spectrum.test_transition_dipoles_velocity_ks(this->eig_ks.c);
                    spectrum.write_transition_dipole(PARAM.globalv.global_out_dir + "transition_dipole_velocity_ks.dat");
                }
            }
        }        
    }
    else if (this->input.bse_tda == "both" || this->input.bse_tda == "full"){
        for (int is = 0;is < this->full_X.size();++is)
        {    //FISH_TODO: full spectrum
            LR::LR_Spectrum<T> spectrum(this->nspin, this->nbasis, this->nocc, this->nvirt, this->gint_, *this->pw_rho, *this->psi_ks,
                this->ucell, this->kv, this->gd, this->orb_cutoff_, this->two_center_bundle_,
                this->paraX_, this->paraC_, this->paraMat_,
                &this->full_ene[is * this->nstates], this->full_X[is].template data<T>(), this->nstates, false/*openshell*/,
                LR_Util::tolower(this->input.abs_gauge));
            spectrum.transition_analysis(this->input.bse_spin_types[is]);
            if (this->input.bse_spin_types[is] != "triplet")        // triplets has no transition dipole and no contribution to the spectrum
            {
                // spectrum.optical_absorption_method1(freq, input.abs_broadening);
                spectrum.write_transition_dipole(PARAM.globalv.global_out_dir + "transition_dipole_full.dat");

                if (LR_Util::tolower(this->input.abs_gauge) == "velocity")
                {
                    spectrum.test_transition_dipoles_velocity_ks(this->eig_ks.c);
                    spectrum.write_transition_dipole(PARAM.globalv.global_out_dir + "transition_dipole_velocity_ks_full.dat");
                }
            }
        }
    }
}

template<typename T, typename TR>
void ESolver_BSE<T, TR>::read_ks_wfc()
{
    assert(this->psi_ks != nullptr);
    this->pelec->ekb.create(this->kv.get_nks(), this->nbands);
    this->pelec->wg.create(this->kv.get_nks(), this->nbands);
    this->eig_gw.create(this->kv.get_nks(), this->nbands);

    int ncore = 0; // skip core bands
    int nbands_file = 0;
    int nk_file = 0;
    int nspin_file = 0;
    int nspin_tmp = PARAM.inp.nspin == 2 ? 2 : 1;
    BSE_IO::parse_band_out_file("band_out", nbands_file, nk_file, nspin_file);
    if (nk_file != this->nk) {
        ModuleBase::WARNING_QUIT("ESolver_BSE", "The nk in band_out is not consistent with BSE::nk.");
    }
    auto eig_gw_info = BSE_IO::read_energy_qp("energy_qp", this->nocc[0], this->nvirt[0], ncore, this->nk, nspin_tmp, nspin_file);
    for (int iks = 0; iks < this->kv.get_nks(); ++iks) {
        for (int ib = 0; ib < this->nbands; ++ib) {
        this->pelec->wg(iks, ib) = eig_gw_info[iks * this->nbands *3 + ib * 3 + 0];
        this->pelec->ekb(iks, ib) = eig_gw_info[iks * this->nbands *3 + ib * 3 + 1];
        this->eig_gw(iks, ib) = eig_gw_info[iks * this->nbands *3 + ib * 3 + 2];
        }
    }
    RI_Benchmark::read_librpa_eigenvectors<T>(*this->psi_ks, "./", ncore, nbands_file, nspin_tmp, nspin_file, this->paraMat_);

    this->eig_ks = std::move(this->pelec->ekb);
}

template<typename T, typename TR>
void ESolver_BSE<T, TR>::init_pot(const Charge& chg_gs)
{
    this->pot.resize(this->nspin, nullptr);
    if (this->input.ri_hartree_benchmark != "none") { return; } //no need to initialize potential for Hxc kernel in the RI-benchmark routine
    switch (this->nspin)
    {
        using ST = LR::PotHxcLR::SpinType;
    case 1: case 2:
        this->pot[0] = std::make_shared<LR::PotHxcLR>(this->xc_kernel, *this->pw_rho, this->ucell, chg_gs, this->Pgrid,
            ST::S1, this->input.lr_init_xc_kernel);
        break;
    // case 2:
    //     this->pot[0] = std::make_shared<PotHxcLR>(xc_kernel, *this->pw_rho, ucell, chg_gs, Pgrid, openshell ? ST::S2_updown : ST::S2_singlet, input.lr_init_xc_kernel);
    //     this->pot[1] = std::make_shared<PotHxcLR>(xc_kernel, *this->pw_rho, ucell, chg_gs, Pgrid, openshell ? ST::S2_updown : ST::S2_triplet, input.lr_init_xc_kernel);
    //     break;
    default:
        throw std::invalid_argument("ESolver_BSE: nspin must be 1 or 2");
    }
}

template<typename T, typename TR>
void ESolver_BSE<T, TR>::allocate_eigen_infos()
{
    ModuleBase::TITLE("ESolver_BSE", "allocate_eigen_infos");

    for (int is = 0;is < this->nspin;++is)
    {
        Parallel_2D px;
        LR_Util::setup_2d_division(px, /*nb2d=*/1, this->nvirt[is], this->nocc[is]
#ifdef __MPI
            , this->paraC_.blacs_ctxt
#endif
        );
        this->paraX_.emplace_back(std::move(px));
    }
    this->nloc_per_state = this->nk * (this->openshell ? 
        this->paraX_[0].get_local_size() + this->paraX_[1].get_local_size() : 
        this->paraX_[0].get_local_size());

    int n_spin_types = this->input.bse_spin_types.size();
    if (this->input.bse_tda == "both" || this->input.bse_tda == "tda") {
        this->tda_ene.resize(n_spin_types * this->nstates);
        this->X.resize(n_spin_types, LR_Util::newTensor<T>({ this->nstates, this->nloc_per_state }));
        for (auto& x : this->X) { x.zero(); }
    }
    if (this->input.bse_tda == "both" || this->input.bse_tda == "full") {
        this->full_ene.resize(n_spin_types * this->nstates);
        this->full_X.resize(n_spin_types, LR_Util::newTensor<T>({ this->nstates, this->nloc_per_state }));
        this->full_Y.resize(n_spin_types, LR_Util::newTensor<T>({ this->nstates, this->nloc_per_state }));
        for (auto& x : this->full_X) { x.zero(); }
        for (auto& y : this->full_Y) { y.zero(); }
    }
}

template class ESolver_BSE<double, double>;
template class ESolver_BSE<std::complex<double>, double>;
} // namespace BSE

