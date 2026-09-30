#include "berryphase.h"

#include "source_base/module_external/lapack_connector.h"

#include "source_io/module_parameter/parameter.h"
#include "source_cell/klist.h"
#include "source_cell/kpoint_strings.h"

namespace
{
constexpr double ELEMENTARY_CHARGE_SI = 1.602176634e-19;
constexpr double BOHR_RADIUS_SI = 5.29177210903e-11;
constexpr double POLARIZATION_AU_TO_SI
    = ELEMENTARY_CHARGE_SI / (BOHR_RADIUS_SI * BOHR_RADIUS_SI);
}

bool berryphase::berry_phase_flag = false;

berryphase::berryphase()
{
    GDIR = PARAM.inp.gdir;
}

#ifdef __LCAO
berryphase::berryphase(const Parallel_Orbitals* paraV_in) : paraV(paraV_in)
{
    GDIR = PARAM.inp.gdir;
}
#endif

berryphase::~berryphase()
{
}

void berryphase::get_occupation_bands()
{
    double occupied_bands = static_cast<double>(PARAM.inp.nelec / ModuleBase::DEGSPIN);
    if ((occupied_bands - std::floor(occupied_bands)) > 0.0)
    {
        occupied_bands = std::floor(occupied_bands) + 1.0;
    }

    occ_nbands = (int)occupied_bands;
    if (occ_nbands > PARAM.inp.nbands)
    {
        ModuleBase::WARNING_QUIT("berryphase::get_occupation_bands",
                                 "not enough bands for berryphase, increase band numbers.");
    }
}

#ifdef __LCAO
void berryphase::lcao_init(const UnitCell& ucell,
                           const Grid_Driver& gd,
                           const K_Vectors& kv,
                           const LCAO_Orbitals& orb)
{
    ModuleBase::TITLE("berryphase", "lcao_init");
    lcao_method.init(ucell, kv.get_nkstot(), orb);
    lcao_method.cal_R_number(ucell, gd);
    lcao_method.cal_orb_overlap(ucell);
    return;
}
#endif

// this routine depends on kpoint's mp method
void berryphase::set_kpoints(const K_Vectors& kv, const int direction)
{
    ModuleBase::TITLE("berryphase", "set_kpoints");

    const std::array<int, 3> mesh = {{kv.nmp[0], kv.nmp[1], kv.nmp[2]}};
    const int spin_channels = PARAM.inp.nspin == 2 ? 2 : 1;
    const ModuleCell::KPointStrings strings
        = ModuleCell::build_periodic_kpoint_strings(mesh, direction, spin_channels);
    this->direction = strings.direction;
    this->nppstr = strings.points_per_string;
    this->k_index = strings.indices;
    this->total_string = static_cast<int>(this->k_index.size());
}

#include "../../source_base/complexmatrix.h"
double berryphase::stringPhase(const UnitCell& ucell,
                               int index_str,
                               int nbands,
                               const int npwx,
                               const psi::Psi<std::complex<double>>* psi_in,
                               const ModulePW::PW_Basis* rhopw,
                               const ModulePW::PW_Basis_K* wfcpw,
                               const K_Vectors& kv)
{
    std::complex<double> zeta(1.0, 0.0);
    ModuleBase::ComplexMatrix mat(nbands, nbands);
    int ik_1 = 0;
    int ik_2 = 0;
    ModuleBase::Vector3<double> G(0.0, 0.0, 0.0);
    ModuleBase::Vector3<double> dk = kv.kvec_c[k_index[index_str][1]] - kv.kvec_c[k_index[index_str][0]];

    for (int k_start = 0; k_start < (nppstr - 1); k_start++)
    {
        ik_1 = k_index[index_str][k_start];
        ik_2 = k_index[index_str][k_start + 1];

        if (PARAM.inp.basis_type == "pw")
        {
            for (int mb = 0; mb < nbands; mb++)
            {

                for (int nb = 0; nb < nbands; nb++)
                {

                    if (PARAM.inp.nspin != 4)
                    {
                        if (k_start == (nppstr - 2))
                        {
                            if (direction == 1)
                            {
                                ModuleBase::Vector3<double> tem_G(1.0, 0.0, 0.0);
                                G = tem_G;
                            }
                            if (direction == 2)
                            {
                                ModuleBase::Vector3<double> tem_G(0.0, 1.0, 0.0);
                                G = tem_G;
                            }
                            if (direction == 3)
                            {
                                ModuleBase::Vector3<double> tem_G(0.0, 0.0, 1.0);
                                G = tem_G;
                            }

                            mat(nb, mb) = pw_method.unkdotp_G0(rhopw, wfcpw, ik_1, ik_2, nb, mb, psi_in, G);
                        }
                        else
                        {
                            mat(nb, mb) = pw_method.unkdotp_G(wfcpw, ik_1, ik_2, nb, mb, psi_in);
                        }
                    }
                    else
                    {
                        if (k_start == (nppstr - 2))
                        {
                            if (direction == 1)
                            {
                                ModuleBase::Vector3<double> tem_G(1.0, 0.0, 0.0);
                                G = tem_G;
                            }
                            if (direction == 2)
                            {
                                ModuleBase::Vector3<double> tem_G(0.0, 1.0, 0.0);
                                G = tem_G;
                            }
                            if (direction == 3)
                            {
                                ModuleBase::Vector3<double> tem_G(0.0, 0.0, 1.0);
                                G = tem_G;
                            }

                            mat(nb, mb) = pw_method.unkdotp_soc_G0(rhopw, wfcpw, ik_1, ik_2, nb, mb, psi_in, G);
                        } else {
                            mat(nb, mb) = pw_method.unkdotp_soc_G(wfcpw,
                                                                  ik_1,
                                                                  ik_2,
                                                                  nb,
                                                                  mb,
                                                                  npwx,
                                                                  psi_in);
                        }
                    }

                } // nb

            } // mb

            std::complex<double> det(1.0, 0.0);
            int info = 0;
            std::vector<int> ipiv(nbands);
            LapackConnector::zgetrf(nbands, nbands, mat, nbands, ipiv.data(), &info);
            for (int ib = 0; ib < nbands; ib++)
            {
                if (ipiv[ib] != (ib + 1)) {
                    det = -det * mat(ib, ib);
                } else {
                    det = det * mat(ib, ib);
                }
            }

            zeta = zeta * det;

            // delete[] ipiv;
        }
#ifdef __LCAO
        else if (PARAM.inp.basis_type == "lcao")
        {
            if (PARAM.inp.nspin != 4)
            {
                zeta = zeta * lcao_method.det_berryphase(ucell,ik_1, ik_2, dk, nbands, *(this->paraV), psi_in, kv);
            }
            else
            {
            }
        }
#endif
    }

    return log(zeta).imag();
}

void berryphase::Berry_Phase(const UnitCell& ucell,
                             int nbands,
                             double& pdl_elec_tot,
                             int& mod_elec_tot,
                             const int npwx,
                             const psi::Psi<std::complex<double>>* psi_in,
                             const ModulePW::PW_Basis* rhopw,
                             const ModulePW::PW_Basis_K* wfcpw,
                             const K_Vectors& kv)
{
    std::complex<double> cave = 0.0;

    std::vector<double> phik(total_string);
    double phik_ave = 0.0;
    std::vector<std::complex<double>> cphik(total_string);
    std::vector<double> wistring(total_string);

    // electron polarization

    // get weight of every std::string
    for (int istring = 0; istring < total_string; istring++)
    {
        wistring[istring] = 1.0 / total_string;
        if (PARAM.inp.nspin == 2) {
            wistring[istring] = wistring[istring] * 2;
        }
    }

    for (int istring = 0; istring < total_string; istring++)
    {
        phik[istring] = stringPhase(ucell,istring, nbands, npwx, psi_in, rhopw, wfcpw, kv);
        // transfer phase to complex number
        cphik[istring] = std::complex<double>(cos(phik[istring]), sin(phik[istring]));
        cave = cave + std::complex<double>(wistring[istring], 0.0) * cphik[istring];
    }

    double theta0 = atan2(cave.imag(), cave.real());
    double dtheta = 0.0;
    for (int istring = 0; istring < total_string; istring++)
    {
        cphik[istring] = cphik[istring] / cave;
        dtheta = atan2(cphik[istring].imag(), cphik[istring].real());
        phik[istring] = (theta0 + dtheta) / (2 * ModuleBase::PI);
        phik_ave = phik_ave + wistring[istring] * phik[istring];
    }

    if (PARAM.inp.nspin == 1)
    {
        pdl_elec_tot = 2 * phik_ave;
    }
    else if (PARAM.inp.nspin == 2 || PARAM.inp.nspin == 4)
    {
        pdl_elec_tot = phik_ave;
    }

    if (PARAM.inp.nspin == 1) // remap to [-1,1]
    {
        pdl_elec_tot = pdl_elec_tot - 2.0 * round(pdl_elec_tot / 2.0);
        mod_elec_tot = 2;
    }
    else if (PARAM.inp.nspin == 2 || PARAM.inp.nspin == 4) // remap to [-0.5,0.5]
    {
        pdl_elec_tot = pdl_elec_tot - 1.0 * round(pdl_elec_tot / 1.0);
        mod_elec_tot = 1;
    }

}

void berryphase::Macroscopic_polarization(const UnitCell& ucell,
                                          const int npwx,
                                          const psi::Psi<std::complex<double>>* psi_in,
                                          const ModulePW::PW_Basis* rhopw,
                                          const ModulePW::PW_Basis_K* wfcpw,
                                          const K_Vectors& kv)
{
    get_occupation_bands();

    GlobalV::ofs_running << "\n\n\n\n";
    GlobalV::ofs_running << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
    GlobalV::ofs_running << " |                                                                    |" << std::endl;
    GlobalV::ofs_running << " | POLARIZATION CALCULATION:                                          |" << std::endl;
    GlobalV::ofs_running << " |                  Modern Theory of Polarization                     |" << std::endl;
    GlobalV::ofs_running << " | calculate the Macroscopic polarization of a crystalline insulator  |" << std::endl;
    GlobalV::ofs_running << " | by using Berry Phase method.                                       |" << std::endl;
    GlobalV::ofs_running << " |                                                                    |" << std::endl;
    GlobalV::ofs_running << " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    GlobalV::ofs_running << "\n\n\n\n";

    // ion polarization
    double polarization_ion[3]; // means three lattice vector directions R1，R2，R3
    ModuleBase::GlobalFunc::ZEROS(polarization_ion, 3);
    // reciprocal lattice
    ModuleBase::Vector3<double> rcell_1(ucell.G.e11, ucell.G.e12, ucell.G.e13);
    ModuleBase::Vector3<double> rcell_2(ucell.G.e21, ucell.G.e22, ucell.G.e23);
    ModuleBase::Vector3<double> rcell_3(ucell.G.e31, ucell.G.e32, ucell.G.e33);
    // int *mod_ion = new int[ucell.nat];
    std::vector<int> mod_ion(ucell.nat);
    // double *pdl_ion_R1 = new double[ucell.nat];
    std::vector<double> pdl_ion_R1(ucell.nat);
    // double *pdl_ion_R2 = new double[ucell.nat];
    std::vector<double> pdl_ion_R2(ucell.nat);
    // double *pdl_ion_R3 = new double[ucell.nat];
    std::vector<double> pdl_ion_R3(ucell.nat);
    ModuleBase::GlobalFunc::ZEROS(mod_ion.data(), ucell.nat);
    ModuleBase::GlobalFunc::ZEROS(pdl_ion_R1.data(), ucell.nat);
    ModuleBase::GlobalFunc::ZEROS(pdl_ion_R2.data(), ucell.nat);
    ModuleBase::GlobalFunc::ZEROS(pdl_ion_R3.data(), ucell.nat);

    bool lodd = false;
    int atom_index = 0;
    for (int it = 0; it < ucell.ntype; it++)
    {
        for (int ia = 0; ia < ucell.atoms[it].na; ia++)
        {
            // should consider fractional electron number
            if (int(ucell.atoms[it].ncpp.zv) % 2 == 1)
            {
                mod_ion[atom_index] = 1;
                lodd = true;
                atom_index++;
            }
            else
            {
                mod_ion[atom_index] = 2;
                atom_index++;
            }
        }
    }

    atom_index = 0;
    for (int it = 0; it < ucell.ntype; it++)
    {
        for (int ia = 0; ia < ucell.atoms[it].na; ia++)
        {
            pdl_ion_R1[atom_index] = ucell.atoms[it].ncpp.zv * (ucell.atoms[it].tau[ia] * rcell_1);
            pdl_ion_R2[atom_index] = ucell.atoms[it].ncpp.zv * (ucell.atoms[it].tau[ia] * rcell_2);
            pdl_ion_R3[atom_index] = ucell.atoms[it].ncpp.zv * (ucell.atoms[it].tau[ia] * rcell_3);
            atom_index++;
        }
    }

    for (int i = 0; i < ucell.nat; i++)
    {
        if (mod_ion[i] == 1)
        {
            pdl_ion_R1[i] = pdl_ion_R1[i] - 1.0 * round(pdl_ion_R1[i] / 1.0);
            pdl_ion_R2[i] = pdl_ion_R2[i] - 1.0 * round(pdl_ion_R2[i] / 1.0);
            pdl_ion_R3[i] = pdl_ion_R3[i] - 1.0 * round(pdl_ion_R3[i] / 1.0);
        }
        else if (mod_ion[i] == 2)
        {
            pdl_ion_R1[i] = pdl_ion_R1[i] - 2.0 * round(pdl_ion_R1[i] / 2.0);
            pdl_ion_R2[i] = pdl_ion_R2[i] - 2.0 * round(pdl_ion_R2[i] / 2.0);
            pdl_ion_R3[i] = pdl_ion_R3[i] - 2.0 * round(pdl_ion_R3[i] / 2.0);
        }

        polarization_ion[0] = polarization_ion[0] + pdl_ion_R1[i];
        polarization_ion[1] = polarization_ion[1] + pdl_ion_R2[i];
        polarization_ion[2] = polarization_ion[2] + pdl_ion_R3[i];
    }
    if (lodd)
    {
        polarization_ion[0] = polarization_ion[0] - 1.0 * round(polarization_ion[0] / 1.0);
        polarization_ion[1] = polarization_ion[1] - 1.0 * round(polarization_ion[1] / 1.0);
        polarization_ion[2] = polarization_ion[2] - 1.0 * round(polarization_ion[2] / 1.0);
    }
    else
    {
        polarization_ion[0] = polarization_ion[0] - 2.0 * round(polarization_ion[0] / 2.0);
        polarization_ion[1] = polarization_ion[1] - 2.0 * round(polarization_ion[1] / 2.0);
        polarization_ion[2] = polarization_ion[2] - 2.0 * round(polarization_ion[2] / 2.0);
    }

    // ion polarization	end

    // calculate Macroscopic polarization modulus because berry phase
    int modulus = 0;
	if ((!lodd) && (PARAM.inp.nspin == 1)) 
	{
		modulus = 2;
	} else {
		modulus = 1;
	}

    // test by jingan
    // GlobalV::ofs_running << "ion polarization end" << std::endl;
    // test by jingan

    // berry phase calculate begin
    switch (GDIR)
    {
    case 1: {
        direction = 1;
        set_kpoints(kv, direction);
        double pdl_elec_tot = 0.0;
        int mod_elec_tot = 0;
        Berry_Phase(ucell,occ_nbands, pdl_elec_tot, mod_elec_tot, npwx, psi_in, rhopw, wfcpw, kv);

        const double rmod = ucell.a1.norm() * ucell.lat0;
        const double unit1 = rmod;
        const double unit2 = rmod / ucell.omega;
        const double unit3 = (rmod / ucell.omega) * POLARIZATION_AU_TO_SI;

        GlobalV::ofs_running << " VALUES OF POLARIZATION" << std::endl;
        GlobalV::ofs_running << std::endl;
        GlobalV::ofs_running << "  The Ionic Phase: " << std::setw(10) << std::fixed << std::setprecision(5)
                             << polarization_ion[0] << std::endl;
        GlobalV::ofs_running << " Electronic Phase: " << std::setw(10) << std::fixed << std::setprecision(5)
                             << pdl_elec_tot << std::endl;
        // GlobalV::ofs_running << " the electronic part polarization is P(ele) = " << rmod * pdl_elec_tot << "
        // (e/Omega).bohr   in R1 direction" << std::endl;

        // calculate total polarization,add electron part and ions part
        double total_polarization = pdl_elec_tot + polarization_ion[0];

        ModuleBase::Vector3<double> polarization_xyz = ucell.a1;
        polarization_xyz.normalize();
        polarization_xyz = total_polarization * polarization_xyz;

        GlobalV::ofs_running << "\n"
                             << "The calculated polarization direction is in R1 direction" << std::endl;
        GlobalV::ofs_running << "\n"
                             << " P = "
                             << outFormat(unit1 * total_polarization, unit1 * modulus, unit1 * polarization_xyz)
                             << "(e/Omega).bohr" << std::endl;
        GlobalV::ofs_running << "\n"
                             << " P = "
                             << outFormat(unit2 * total_polarization, unit2 * modulus, unit2 * polarization_xyz)
                             << "e/bohr^2" << std::endl;
        GlobalV::ofs_running << "\n"
                             << " P = "
                             << outFormat(unit3 * total_polarization, unit3 * modulus, unit3 * polarization_xyz)
                             << "C/m^2" << std::endl;
        GlobalV::ofs_running << std::endl;

        break;
    }
    case 2: {
        direction = 2;
        set_kpoints(kv, direction);
        double pdl_elec_tot = 0.0;
        int mod_elec_tot = 0;
        Berry_Phase(ucell,occ_nbands, pdl_elec_tot, mod_elec_tot, npwx, psi_in, rhopw, wfcpw, kv);

        const double rmod = ucell.a2.norm() * ucell.lat0;
        const double unit1 = rmod;
        const double unit2 = rmod / ucell.omega;
        const double unit3 = (rmod / ucell.omega) * POLARIZATION_AU_TO_SI;

        GlobalV::ofs_running << " VALUES OF POLARIZATION" << std::endl;
        GlobalV::ofs_running << std::endl;
        GlobalV::ofs_running << "  The Ionic Phase: " << std::setw(10) << std::fixed << std::setprecision(5)
                             << polarization_ion[1] << std::endl;
        GlobalV::ofs_running << " Electronic Phase: " << std::setw(10) << std::fixed << std::setprecision(5)
                             << pdl_elec_tot << std::endl;
        // GlobalV::ofs_running << " the electronic part polarization is P(ele) = " << rmod * pdl_elec_tot << "
        // (e/Omega).bohr   in R2 direction" << std::endl;

        // calculate total polarization,add electron part and ions part
        double total_polarization = pdl_elec_tot + polarization_ion[1];

        ModuleBase::Vector3<double> polarization_xyz = ucell.a2;
        polarization_xyz.normalize();
        polarization_xyz = total_polarization * polarization_xyz;

        GlobalV::ofs_running << "\n"
                             << "The calculated polarization direction is in R2 direction" << std::endl;
        GlobalV::ofs_running << "\n"
                             << " P = "
                             << outFormat(unit1 * total_polarization, unit1 * modulus, unit1 * polarization_xyz)
                             << "(e/Omega).bohr" << std::endl;
        GlobalV::ofs_running << "\n"
                             << " P = "
                             << outFormat(unit2 * total_polarization, unit2 * modulus, unit2 * polarization_xyz)
                             << "e/bohr^2" << std::endl;
        GlobalV::ofs_running << "\n"
                             << " P = "
                             << outFormat(unit3 * total_polarization, unit3 * modulus, unit3 * polarization_xyz)
                             << "C/m^2" << std::endl;
        GlobalV::ofs_running << std::endl;

        break;
    }
    case 3: {
        direction = 3;
        set_kpoints(kv, direction);
        double pdl_elec_tot = 0.0;
        int mod_elec_tot = 0;
        Berry_Phase(ucell,occ_nbands, pdl_elec_tot, mod_elec_tot, npwx, psi_in, rhopw, wfcpw, kv);

        const double rmod = ucell.a3.norm() * ucell.lat0;
        const double unit1 = rmod;
        const double unit2 = rmod / ucell.omega;
        const double unit3 = (rmod / ucell.omega) * POLARIZATION_AU_TO_SI;

        GlobalV::ofs_running << " VALUES OF POLARIZATION" << std::endl;
        GlobalV::ofs_running << std::endl;
        GlobalV::ofs_running << "  The Ionic Phase: " << std::setw(10) << std::fixed << std::setprecision(5)
                             << polarization_ion[2] << std::endl;
        GlobalV::ofs_running << " Electronic Phase: " << std::setw(10) << std::fixed << std::setprecision(5)
                             << pdl_elec_tot << std::endl;
        // GlobalV::ofs_running << " the electronic part polarization is P(ele) = " << rmod * pdl_elec_tot << "
        // (e/Omega).bohr   in R3 direction" << std::endl;

        // calculate total polarization,add electron part and ions part
        double total_polarization = pdl_elec_tot + polarization_ion[2];

        ModuleBase::Vector3<double> polarization_xyz = ucell.a3;
        polarization_xyz.normalize();
        polarization_xyz = total_polarization * polarization_xyz;

        GlobalV::ofs_running << "\n"
                             << "The calculated polarization direction is in R3 direction" << std::endl;
        GlobalV::ofs_running << "\n"
                             << " P = "
                             << outFormat(unit1 * total_polarization, unit1 * modulus, unit1 * polarization_xyz)
                             << "(e/Omega).bohr" << std::endl;
        GlobalV::ofs_running << "\n"
                             << " P = "
                             << outFormat(unit2 * total_polarization, unit2 * modulus, unit2 * polarization_xyz)
                             << "e/bohr^2" << std::endl;
        GlobalV::ofs_running << "\n"
                             << " P = "
                             << outFormat(unit3 * total_polarization, unit3 * modulus, unit3 * polarization_xyz)
                             << "C/m^2" << std::endl;
        GlobalV::ofs_running << std::endl;

        break;
    }
    }

    return;
}

std::string berryphase::outFormat(const double polarization,
                                  const double modulus,
                                  const ModuleBase::Vector3<double> project) const
{
    std::stringstream outStr;
    outStr << std::setw(12) << std::fixed << std::setprecision(7) << polarization << "  (mod ";
    outStr << std::setw(12) << std::fixed << std::setprecision(7) << modulus << ")  (";
    outStr << std::setw(12) << std::fixed << std::setprecision(7) << project.x << ",";
    outStr << std::setw(12) << std::fixed << std::setprecision(7) << project.y << ",";
    outStr << std::setw(12) << std::fixed << std::setprecision(7) << project.z << ") ";

    return outStr.str();
}
