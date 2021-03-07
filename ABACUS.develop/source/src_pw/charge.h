#ifndef CHARGE_H
#define CHARGE_H

#include "tools.h"
#include "../src_parallel/parallel_global.h"

//==========================================================
// Electron Charge Density 
//==========================================================
class Charge
{

public:

    Charge();
    ~Charge();

//==========================================================
// MEMBER VARIABLES :
// NAME : rho (nspin,ncxyz), the charge density in real space
// NAME : rho_save (nspin,ncxyz), for charge mixing
// NAME : rhog, charge density in G space
// NAME : rhog_save, chage density in G space
// NAME : rho_core [nrxx], the core charge in real space
// NAME : rhog_core [ngm], the core charge in reciprocal space
//==========================================================

    double** rho;
    double** rho_save;

    complex<double>** rhog;
    complex<double>** rhog_save;

    double *rho_core;
	complex<double> *rhog_core;

	//  output charge if out_charge > 0, and output every "out_charge" elec step.
    int out_charge;

    double *start_mag_type;
    double *start_mag_atom;

	// mohan update 2021-02-20
	void allocate(const int &nspin_in, const int &nrxx_in, const int &ngmc_in);

    void atomic_rho(const int spin_number_need, double **rho_in)const;

    void set_rho_core(const ComplexMatrix &structure_factor);

    void sum_band(void);

    void renormalize_rho(void);

    void save_rho_before_sum_band(void);

	// for non-linear core correction
    void non_linear_core_correction
    (
        const bool &numeric,
        const int mesh,
        const double *r,
        const double *rab,
        const double *rhoc,
        double *rhocg
    );

	double check_ne(const double *rho_in) const;

    void init_final_scf(); //LiuXh add 20180619

	public:

    void write_rho(const double* rho_save, const int &is, const int &iter, const string &fn, 
		const int &precision = 11, const bool for_plot = false);//mohan add 2007-10-17

    void write_rho_dipole(const double* rho_save, const int &is, const int &iter, const string &fn, 
		const int &precision = 11, const bool for_plot = false);//fuxiang add 2017-3-15    

    bool read_rho(const int &is, const string &fn, double* rho);//mohan add 2007-10-17


	private:

	// mohan add 2021-02-20
	int nrxx; // number of r vectors in this processor
	int ngmc; // number of g vectors in this processor
	int nspin; // number of spins

    void sum_band_k();

    void rho_mpi(void);

    double sum_rho(void) const;

	bool allocate_rho;

	bool allocate_rho_final_scf; //LiuXh add 20180606

};

#endif //charge

