#pragma once
#include "source_lcao/module_lr/esolver_lrtd_lcao.h"
#include "source_lcao/module_lr/ri_benchmark/ri_benchmark.h"
#include "source_lcao/module_ri/Exx_LRI.h"
#include "source_io/module_parameter/parameter.h"
#include "source_io/cube_io.h"
#include "source_io/print_info.h"
#include "../utils/lr_io.h"
#include "hamilt_bse.h"
#include "source_lcao/module_lr/lr_spectrum.h"
namespace BSE
{
    template<typename T> using Real = typename GetTypeReal<T>::type;

    template<typename T, typename TR = double>
    class ESolver_BSE : public LR::ESolver_LR<T, TR> {
    public:
        LR_IO::RI_kRlist kRlist;
        psi::Psi<T>* psi_ks_global; ///< global version of psi_ks
        ModuleBase::matrix eig_gw; ///< GW energy
        std::vector<double> tda_ene, full_ene; // in Rydberg

        /// @brief  - [nspin_types][{nstates, nk* (locc* lvirt}]
        std::vector<ct::Tensor> full_X, full_Y;

        /// @brief a from-scratch constructor
        ESolver_BSE(const Input_para& inp, UnitCell& ucell);

        ~ESolver_BSE() override {
            //delete this->psi_ks; // already deleted in ESolver_LR
            delete this->psi_ks_global;
        }

        void exx_init();

        virtual void runner(UnitCell& ucell, int istep) override;
        virtual void after_all_runners(UnitCell& ucell) override;

        /// @brief read in the ground state wave function, gw band energy and occupation
        virtual void read_ks_wfc() override;

        /// @brief init Hartree potential for BSE,
        /// @attention here we don't multiply 2 for singlet or 0 for triplet, we will do it in HamiltBSE
        virtual void init_pot(const Charge& chg_gs) override;

        
        /// @brief X for tda and also Y for full BSE excitation
        void allocate_eigen_infos();

    };

}