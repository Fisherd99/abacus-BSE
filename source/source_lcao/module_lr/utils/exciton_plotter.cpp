#include "exciton_plotter.h"

namespace LR_Util
{

template <typename T>
elecstate::DensityMatrix<T, T> ExcitonPlotter<T>::cal_transition_density_matrix(const int istate)
{
    assert(this->nspin_x == 1);               // only close shell temperarily 25-12-22
    const int offset_b = istate * this->ldim; // start index of band istate
    elecstate::DensityMatrix<T, T> DM_trans(&this->pmat, this->nspin_x, this->kv.kvec_d, this->nk);
    for (int is = 0; is < this->nspin_x; ++is)
    {
        const int offset_x = offset_b + is * nk * this->pX[0].get_local_size();

#ifdef __MPI
        std::vector<container::Tensor> dm_trans_2d = LR::cal_dm_trans_pblas(this->X + offset_x,
                                                                            this->pX[is],
                                                                            this->psi_ks_vec[is],
                                                                            this->pc,
                                                                            this->naos,
                                                                            this->nocc[is],
                                                                            this->nvirt[is],
                                                                            this->pmat,
                                                                            (T)1.0 / (T)nk);
#else
        std::vector<container::Tensor> dm_trans_2d = LR::cal_dm_trans_blas(this->X + offset_x,
                                                                            this->psi_ks_vec[is],
                                                                            this->nocc[is],
                                                                            this->nvirt[is],
                                                                            (T)1.0 / (T)nk);
#endif
        for (int ik = 0; ik < this->nk; ++ik)
        {
            DM_trans.set_DMK_pointer(ik + is * nk, dm_trans_2d[ik].data<T>());
        }
    }
    LR_Util::initialize_DMR(DM_trans, this->pmat, this->ucell, this->gd_, this->orb_cutoff_);
    DM_trans.cal_DMR();

    return DM_trans;
}

template <>
void ExcitonPlotter<double>::plot_exciton(const int istate, const std::string& type)
{
    const elecstate::DensityMatrix<double, double> DM_trans = this->cal_transition_density_matrix(istate);
    double** rho_trans;
    LR_Util::_allocate_2order_nested_ptr(rho_trans, this->nspin_x, this->rho_basis.nrxx);
    for (int is = 0;is < this->nspin_x;++is)
    {
        ModuleBase::GlobalFunc::ZEROS(rho_trans[is], this->rho_basis.nrxx);
    }
    // cal_gint_rho is not suitable for exciton \psi_e(r_e) * \psi_h(r_h), since r_e ≠ r_h
    ModuleGint::cal_gint_rho(DM_trans.get_DMR_vector(), nspin_x, rho_trans, false);

    for (int is = 0;is < this->nspin_x;++is)
    {
        for (int ixx = 0; ixx < this->rho_basis.nrxx; ++ixx)
        {
            rho_trans[is][ixx] = std::pow(rho_trans[is][ixx],2);
        }
        std::string fn = PARAM.globalv.global_out_dir + "/Exciton_" + type + std::to_string(istate) + "_spin" + std::to_string(is) + ".cube";
        ModuleIO::write_vdata_palgrid(this->Pgrid,
            rho_trans[is],
            is,
            this->nspin_x,
            0/*iter*/,
            fn,
            eig[istate],/*shown as fermi energy*/
            &this->ucell);
    }
    LR_Util::_deallocate_2order_nested_ptr(rho_trans, this->nspin_x);
}

template <>
void ExcitonPlotter<std::complex<double>>::plot_exciton(const int istate, const std::string& type)
{
    const elecstate::DensityMatrix<std::complex<double>, std::complex<double>> DM_trans = this->cal_transition_density_matrix(istate);
    elecstate::DensityMatrix<std::complex<double>, double> DM_trans_real_imag(&this->pmat, this->nspin_x, this->kv.kvec_d, this->nk);
    LR_Util::initialize_DMR(DM_trans_real_imag, this->pmat, this->ucell, this->gd_, this->orb_cutoff_);

    double **rho_trans, **rho_trans_real, **rho_trans_imag;
    LR_Util::_allocate_2order_nested_ptr(rho_trans, nspin_x, this->rho_basis.nrxx);
    LR_Util::_allocate_2order_nested_ptr(rho_trans_real, 1, this->rho_basis.nrxx);
    LR_Util::_allocate_2order_nested_ptr(rho_trans_imag, 1, this->rho_basis.nrxx);

    for (int is = 0;is < this->nspin_x;++is)
    {
        ModuleBase::GlobalFunc::ZEROS(rho_trans[is], this->rho_basis.nrxx);
        ModuleBase::GlobalFunc::ZEROS(rho_trans_real[0], this->rho_basis.nrxx);
        ModuleBase::GlobalFunc::ZEROS(rho_trans_imag[0], this->rho_basis.nrxx);

        // cal_gint_rho is not suitable for exciton \psi_e(r_e) * \psi_h(r_h), since r_e ≠ r_h
        LR_Util::get_DMR_real_imag_part(DM_trans, DM_trans_real_imag, ucell.nat, 'R');
        ModuleGint::cal_gint_rho({ DM_trans_real_imag.get_DMR_vector().at(is) }, 1, rho_trans_real, false);
        for (int ixx=0; ixx < 10; ++ixx)
            { std::cout<<rho_trans_real[0][ixx]<<std::endl; std::cout<<rho_trans_imag[0][ixx]<<std::endl; }
        std::cout<<"111"<<std::endl;
        LR_Util::get_DMR_real_imag_part(DM_trans, DM_trans_real_imag, ucell.nat, 'I');
        ModuleGint::cal_gint_rho({ DM_trans_real_imag.get_DMR_vector().at(is) }, 1, rho_trans_imag, false);
        for (int ixx=0; ixx < 10; ++ixx)
            { std::cout<<rho_trans_real[0][ixx]<<std::endl; std::cout<<rho_trans_imag[0][ixx]<<std::endl; }
        std::cout<<"222"<<std::endl;
        for (int ixx=0; ixx < this->rho_basis.nrxx; ++ixx)
        {
            rho_trans[is][ixx] = std::pow(rho_trans_real[0][ixx],2) + std::pow(rho_trans_imag[0][ixx],2);
            std::cout<<rho_trans[is][ixx]<<std::endl;
        }
        std::string fn = PARAM.globalv.global_out_dir + "/Exciton_" + type + std::to_string(istate) + "_spin" + std::to_string(is) + ".cube";
        ModuleIO::write_vdata_palgrid(this->Pgrid,
            rho_trans[is],
            is,
            this->nspin_x,
            0/*iter*/,
            fn,
            eig[istate],/*shown as fermi energy*/
            &this->ucell);
    }
    LR_Util::_deallocate_2order_nested_ptr(rho_trans, nspin_x);
    LR_Util::_deallocate_2order_nested_ptr(rho_trans_real, 1);
    LR_Util::_deallocate_2order_nested_ptr(rho_trans_imag, 1);
}

} //namespace LR_Util