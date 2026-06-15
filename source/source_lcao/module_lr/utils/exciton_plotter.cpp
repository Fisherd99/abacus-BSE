#include "exciton_plotter.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <set>
#include <utility>
#include "source_base/constants.h"
#include "source_base/element_name.h"
#include "source_base/ylm.h"
#include "source_base/parallel_global.h"
#include "source_cell/atom_spec.h"
#include "source_base/matrix3.h"

namespace
{
using Vec3 = ModuleBase::Vector3<double>;

Vec3 atom_position_bohr(const UnitCell& ucell, const int iat)
{
    return ucell.get_tau(iat) * ucell.lat0;
}

int atomic_number_from_label(std::string element)
{
    element.erase(std::remove_if(element.begin(),
                                 element.end(),
                                 [](const char c) { return c >= '0' && c <= '9'; }),
                  element.end());
    for (int i = 0; i < static_cast<int>(ModuleBase::element_name.size()); ++i)
    {
        if (element == ModuleBase::element_name[i])
        {
            return i + 1;
        }
    }
    return 0;
}

std::array<int, 3> bvk_mesh_from_kpoints(const K_Vectors& kv, const int nk)
{
    if (kv.nmp[0] > 0 && kv.nmp[1] > 0 && kv.nmp[2] > 0)
    {
        return {kv.nmp[0], kv.nmp[1], kv.nmp[2]};
    }

    std::set<long long> kx, ky, kz;
    const int n = std::min<int>(nk, kv.kvec_d.size());
    constexpr double scale = 1.0e10;
    for (int ik = 0; ik < n; ++ik)
    {
        kx.insert(static_cast<long long>(std::llround(kv.kvec_d[ik].x * scale)));
        ky.insert(static_cast<long long>(std::llround(kv.kvec_d[ik].y * scale)));
        kz.insert(static_cast<long long>(std::llround(kv.kvec_d[ik].z * scale)));
    }
    return {std::max(1, static_cast<int>(kx.size())),
            std::max(1, static_cast<int>(ky.size())),
            std::max(1, static_cast<int>(kz.size()))};
}

struct SliceGeometry
{
    std::string plane;
    double slice_pos = 0.0;
    Vec3 a_bohr, b_bohr, c_bohr;
    Vec3 u_vec, v_vec, perp_offset;
    int u_axis = 0, v_axis = 1;
    int nk_u = 1, nk_v = 1;
    int u_start_cells = 0, u_end_cells = 1;
    int v_start_cells = 0, v_end_cells = 1;
    double u_start = 0.0, u_end = 1.0;
    double v_start = 0.0, v_end = 1.0;
    int res = 1, nu = 2, nv = 2;
};

SliceGeometry make_slice_geometry(const UnitCell& ucell,
                                  const K_Vectors& kv,
                                  const int nk,
                                  const std::string& plane,
                                  const double slice_pos,
                                  const int npoints,
                                  const double scale)
{
    SliceGeometry geom;
    geom.plane = plane;
    geom.slice_pos = slice_pos;
    geom.a_bohr = ucell.a1 * ucell.lat0;
    geom.b_bohr = ucell.a2 * ucell.lat0;
    geom.c_bohr = ucell.a3 * ucell.lat0;

    const auto mesh = bvk_mesh_from_kpoints(kv, nk);
    if (plane == "ab")
    {
        geom.u_vec = geom.a_bohr;
        geom.v_vec = geom.b_bohr;
        geom.perp_offset = geom.c_bohr * (slice_pos / geom.c_bohr.norm());
        geom.u_axis = 0;
        geom.v_axis = 1;
        geom.nk_u = mesh[0];
        geom.nk_v = mesh[1];
    }
    else if (plane == "bc")
    {
        geom.u_vec = geom.b_bohr;
        geom.v_vec = geom.c_bohr;
        geom.perp_offset = geom.a_bohr * (slice_pos / geom.a_bohr.norm());
        geom.u_axis = 1;
        geom.v_axis = 2;
        geom.nk_u = mesh[1];
        geom.nk_v = mesh[2];
    }
    else if (plane == "ca")
    {
        geom.u_vec = geom.c_bohr;
        geom.v_vec = geom.a_bohr;
        geom.perp_offset = geom.b_bohr * (slice_pos / geom.b_bohr.norm());
        geom.u_axis = 2;
        geom.v_axis = 0;
        geom.nk_u = mesh[2];
        geom.nk_v = mesh[0];
    }
    else
    {
        ModuleBase::WARNING_QUIT("ExcitonPlotter",
            "Unknown slice plane: " + plane + ". Use ab, bc, or ca.");
    }

    const double safe_scale = std::max(1.0, scale);
    const int pad_u = std::max(0, static_cast<int>(std::ceil((safe_scale - 1.0) * 0.5 * geom.nk_u)));
    const int pad_v = std::max(0, static_cast<int>(std::ceil((safe_scale - 1.0) * 0.5 * geom.nk_v)));
    const int total_u = geom.nk_u + 2 * pad_u;
    const int total_v = geom.nk_v + 2 * pad_v;

    geom.u_start_cells = static_cast<int>(std::floor(0.5 - total_u * 0.5));
    geom.u_end_cells = geom.u_start_cells + total_u;
    geom.v_start_cells = static_cast<int>(std::floor(0.5 - total_v * 0.5));
    geom.v_end_cells = geom.v_start_cells + total_v;
    geom.u_start = static_cast<double>(geom.u_start_cells);
    geom.u_end = static_cast<double>(geom.u_end_cells);
    geom.v_start = static_cast<double>(geom.v_start_cells);
    geom.v_end = static_cast<double>(geom.v_end_cells);

    const int max_cells = std::max(total_u, total_v);
    geom.res = std::max(1, npoints / std::max(1, max_cells));
    geom.nu = total_u * geom.res + 1;
    geom.nv = total_v * geom.res + 1;
    return geom;
}

double bloch_phase_arg(const Vec3& kvec_d, const int n1, const int n2, const int n3)
{
    return ModuleBase::TWO_PI * (kvec_d.x * n1 + kvec_d.y * n2 + kvec_d.z * n3);
}

double bloch_phase_arg_axis(const Vec3& kvec_d, const int axis, const int ncell)
{
    return ModuleBase::TWO_PI * kvec_d[axis] * ncell;
}

std::complex<double> to_complex(const double value)
{
    return std::complex<double>(value, 0.0);
}

std::complex<double> to_complex(const std::complex<double>& value)
{
    return value;
}

template <typename TK>
double density_from_dmk(const TK* dmk,
                        const std::vector<std::complex<double>>& phi,
                        const int naos)
{
    std::complex<double> rho(0.0, 0.0);
    for (int mu = 0; mu < naos; ++mu)
    {
        std::complex<double> row_sum(0.0, 0.0);
        for (int nu = 0; nu < naos; ++nu)
        {
            row_sum += to_complex(dmk[mu + nu * naos]) * std::conj(phi[nu]);
        }
        rho += phi[mu] * row_sum;
    }
    return std::max(0.0, rho.real());
}

void write_slice_data(const UnitCell& ucell,
                      const SliceGeometry& geom,
                      const std::vector<double>& density,
                      const std::string& filename,
                      const int istate,
                      const double energy_ry,
                      const std::string& density_kind,
                      const bool has_hole,
                      const std::array<double, 3>& r_h_fix)
{
    std::ofstream ofs(filename);
    ofs.precision(12);
    ofs << "# state " << istate << "\n";
    ofs << "# energy_Ry " << energy_ry << "\n";
    ofs << "# density_kind " << density_kind << "\n";
    ofs << "# has_hole " << (has_hole ? 1 : 0) << "\n";
    ofs << "# fixed_particle "
        << (density_kind == "conditional_hole" ? "electron"
            : (density_kind == "conditional_elec" ? "hole" : "none")) << "\n";
    ofs << "# hole_fix_x " << r_h_fix[0] << "\n";
    ofs << "# hole_fix_y " << r_h_fix[1] << "\n";
    ofs << "# hole_fix_z " << r_h_fix[2] << "\n";
    ofs << "# plane " << geom.plane << "\n";
    ofs << "# slice_pos " << geom.slice_pos << "\n";
    ofs << "# u_min " << geom.u_start << "\n";
    ofs << "# u_max " << geom.u_end << "\n";
    ofs << "# v_min " << geom.v_start << "\n";
    ofs << "# v_max " << geom.v_end << "\n";
    ofs << "# u_vec_x " << geom.u_vec.x << "\n";
    ofs << "# u_vec_y " << geom.u_vec.y << "\n";
    ofs << "# u_vec_z " << geom.u_vec.z << "\n";
    ofs << "# v_vec_x " << geom.v_vec.x << "\n";
    ofs << "# v_vec_y " << geom.v_vec.y << "\n";
    ofs << "# v_vec_z " << geom.v_vec.z << "\n";
    ofs << "# grid_nu " << geom.nu << "\n";
    ofs << "# grid_nv " << geom.nv << "\n";
    ofs << "# BvK_u " << geom.nk_u << "\n";
    ofs << "# BvK_v " << geom.nk_v << "\n";
    ofs << "# cell_a_x " << geom.a_bohr.x << "\n";
    ofs << "# cell_a_y " << geom.a_bohr.y << "\n";
    ofs << "# cell_a_z " << geom.a_bohr.z << "\n";
    ofs << "# cell_b_x " << geom.b_bohr.x << "\n";
    ofs << "# cell_b_y " << geom.b_bohr.y << "\n";
    ofs << "# cell_b_z " << geom.b_bohr.z << "\n";
    ofs << "# cell_c_x " << geom.c_bohr.x << "\n";
    ofs << "# cell_c_y " << geom.c_bohr.y << "\n";
    ofs << "# cell_c_z " << geom.c_bohr.z << "\n";
    ofs << "# n_atoms " << ucell.nat << "\n";
    for (int iat = 0; iat < ucell.nat; ++iat)
    {
        const int it = ucell.iat2it[iat];
        const Vec3 tau = atom_position_bohr(ucell, iat);
        ofs << "# atom_label_" << iat << " " << ucell.atoms[it].label << "\n";
        ofs << "# atom_x_" << iat << " " << tau.x << "\n";
        ofs << "# atom_y_" << iat << " " << tau.y << "\n";
        ofs << "# atom_z_" << iat << " " << tau.z << "\n";
    }
    for (int iu = 0; iu < geom.nu; ++iu)
    {
        for (int iv = 0; iv < geom.nv; ++iv)
        {
            ofs << density[iu * geom.nv + iv] << (iv + 1 < geom.nv ? " " : "");
        }
        ofs << "\n";
    }
}
} // namespace

// ============================================================================
// OrbitalEvaluator non-template method implementations
// ============================================================================

void OrbitalEvaluator::init(const LCAO_Orbitals& orb, const UnitCell& ucell)
{
    atypes_.resize(ucell.ntype);
    for (int it = 0; it < ucell.ntype; ++it)
    {
        const Numerical_Orbital& no = orb.Phi[it];
        const Atom& atom = ucell.atoms[it];
        auto& td = atypes_[it];

        td.nw = atom.nw;
        td.nwl = atom.nwl;
        td.rcut = no.getRcut();
        td.dr_uniform = no.PhiLN(0, 0).dr_uniform;
        td.iw2_ylm.resize(atom.nw);
        td.iw2_new.resize(atom.nw);
        td.psi_ptrs.resize(atom.nw, nullptr);
        td.dpsi_ptrs.resize(atom.nw, nullptr);

        for (int iw = 0; iw < atom.nw; ++iw)
        {
            td.iw2_ylm[iw] = atom.iw2_ylm[iw];
            td.iw2_new[iw] = atom.iw2_new[iw];
            if (atom.iw2_new[iw])
            {
                int l = atom.iw2l[iw];
                int n = atom.iw2n[iw];
                const auto& lm = no.PhiLN(l, n);
                td.psi_ptrs[iw] = lm.psi_uniform.data();
                td.dpsi_ptrs[iw] = lm.dpsi_uniform.data();
            }
        }
    }
}

/// @brief Replicates the cubic Hermite interpolation from GintAtom::set_phi()
void OrbitalEvaluator::eval_phi(const int it, const ModuleBase::Vector3<double>& dr, double* phi_out) const
{
    const auto& td = atypes_[it];
    const double dist = dr.norm() < 1e-12 ? 1e-12 : dr.norm();
    if (dist > td.rcut)
    {
        ModuleBase::GlobalFunc::ZEROS(phi_out, td.nw);
        return;
    }

    std::vector<double> ylma;
    ModuleBase::Ylm::sph_harm(td.nwl, dr.x / dist, dr.y / dist, dr.z / dist, ylma);

    const double position = dist / td.dr_uniform;
    const int ip = static_cast<int>(position);
    const double dx = position - ip;
    const double dx2 = dx * dx;
    const double dx3 = dx2 * dx;
    const double c3 = 3.0 * dx2 - 2.0 * dx3;
    const double c1 = 1.0 - c3;
    const double c2 = (dx - 2.0 * dx2 + dx3) * td.dr_uniform;
    const double c4 = (dx3 - dx2) * td.dr_uniform;

    for (int iw = 0; iw < td.nw; ++iw)
    {
        double psi_radial = 0.0;
        if (td.iw2_new[iw])
        {
            auto* p = td.psi_ptrs[iw];
            auto* dp = td.dpsi_ptrs[iw];
            psi_radial = c1 * p[ip] + c2 * dp[ip] + c3 * p[ip + 1] + c4 * dp[ip + 1];
        }
        phi_out[iw] = psi_radial * ylma[td.iw2_ylm[iw]];
    }
}

// =============================================================================
// Evaluate the Bloch-summed numerical atomic orbital at position r:
//   phi^B_mu(r, k) = Sum_R exp(i * 2pi * kvec_d · (n1,n2,n3)) * phi_mu(r - tau - R)
// where kvec_d is the direct-coordinate k-vector. R runs over all lattice vectors
// within the orbital cutoff radius R = n1*a + n2*b + n3*c.
// =============================================================================
void OrbitalEvaluator::eval_phi_all_bloch(
    const ModuleBase::Vector3<double>& r,
    const UnitCell& ucell,
    std::complex<double>* phi_bloch,
    const ModuleBase::Vector3<double>& kvec_d) const
{
    // Cell vectors in Bohr for real-space distance checks
    const double lat0 = ucell.lat0;
    ModuleBase::Vector3<double> a = ucell.a1 * lat0;
    ModuleBase::Vector3<double> b = ucell.a2 * lat0;
    ModuleBase::Vector3<double> c = ucell.a3 * lat0;

    // Determine how many lattice translations to check in each direction.
    // nR = ceil(rcut / min(|a|,|b|,|c|)) + 1 ensures all overlapping orbitals are captured.
    double max_rcut = 0.0;
    for (const auto& td : atypes_) max_rcut = std::max(max_rcut, td.rcut);
    const int nR = static_cast<int>(std::ceil(max_rcut / std::min({a.norm(), b.norm(), c.norm()}))) + 1;

    std::vector<double> phi_tmp;
    int iw_global = 0;

    // For each atom, iterate over neighboring cell images
    for (int iat = 0; iat < ucell.nat; ++iat)
    {
        const int it = ucell.iat2it[iat];
        const int ia = ucell.iat2ia[iat];
        const ModuleBase::Vector3<double> tau = ucell.atoms[it].tau[ia] * lat0;
        const int nw = atypes_[it].nw;
        const double rcut = atypes_[it].rcut;
        phi_tmp.resize(nw);

        // Triple loop over lattice vectors R = n1*a + n2*b + n3*c
        for (int n1 = -nR; n1 <= nR; ++n1) {
        for (int n2 = -nR; n2 <= nR; ++n2) {
        for (int n3 = -nR; n3 <= nR; ++n3) {
            ModuleBase::Vector3<double> R = a * (double)n1 + b * (double)n2 + c * (double)n3;
            ModuleBase::Vector3<double> dr = r - tau - R;
            if (dr.norm() > rcut) continue;  // skip if orbital does not reach

            double kdotR = bloch_phase_arg(kvec_d, n1, n2, n3);
            std::complex<double> exp_ikR(std::cos(kdotR), std::sin(kdotR));

            // Evaluate real-space orbital and accumulate with Bloch phase
            eval_phi(it, dr, phi_tmp.data());
            for (int iw = 0; iw < nw; ++iw)
                phi_bloch[iw_global + iw] += exp_ikR * phi_tmp[iw];
        }}}
        iw_global += nw;
    }
}

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
        LR_Util::get_DMR_real_imag_part(DM_trans, DM_trans_real_imag, ucell.nat, 'I');
        ModuleGint::cal_gint_rho({ DM_trans_real_imag.get_DMR_vector().at(is) }, 1, rho_trans_imag, false);
        for (int ixx=0; ixx < this->rho_basis.nrxx; ++ixx)
        {
            rho_trans[is][ixx] = std::pow(rho_trans_real[0][ixx],2) + std::pow(rho_trans_imag[0][ixx],2);
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

template <>
std::pair<std::vector<container::Tensor>, double>
ExcitonPlotter<double>::cal_effective_dmk_hole(const int istate)
{
    ModuleBase::TITLE("ExcitonPlotter", "cal_effective_dmk_hole<double>");
    assert(this->nspin_x == 1);
    const int offset_b = istate * this->ldim;
    const int nks = this->nk;
    const int naos = this->naos;
    const int nocc = this->nocc[0];
    const int nvirt = this->nvirt[0];

    std::vector<container::Tensor> dm_hole(nks,
        container::Tensor(DAT::DT_DOUBLE, DEV::CpuDevice, {naos, naos}));
    for (auto& dm : dm_hole) ModuleBase::GlobalFunc::ZEROS(dm.data<double>(), naos * naos);

    const double alpha = 1.0, beta = 0.0;
    double total_weight = 0.0;
    char transa, transb;

    for (int ik = 0; ik < nks; ++ik)
    {
        this->psi_ks_vec[0].fix_k(ik);
        const int x_start = ik * nocc * nvirt;

        // M_k = X_k^T * X_k → nocc x nocc symmetric
        container::Tensor M_k(DAT::DT_DOUBLE, DEV::CpuDevice, {nocc, nocc});
        ModuleBase::GlobalFunc::ZEROS(M_k.data<double>(), nocc * nocc);
        transa = 'T'; transb = 'N';
        dgemm_(&transa, &transb, &nocc, &nocc, &nvirt,
               &alpha,
               this->X + offset_b + x_start, &nvirt,
               this->X + offset_b + x_start, &nvirt,
               &beta, M_k.data<double>(), &nocc);

        for (int v = 0; v < nocc; ++v)
            total_weight += M_k.data<double>()[v * nocc + v];

        // temp = C_occ * M_k → naos x nocc
        container::Tensor temp(DAT::DT_DOUBLE, DEV::CpuDevice, {naos, nocc});
        transa = 'N'; transb = 'N';
        dgemm_(&transa, &transb, &naos, &nocc, &nocc,
               &alpha,
               this->psi_ks_vec[0].get_pointer(0), &naos,
               M_k.data<double>(), &nocc,
               &beta, temp.data<double>(), &naos);

        // dm_hole = temp * C_occ^T → naos x naos
        transa = 'N'; transb = 'T';
        dgemm_(&transa, &transb, &naos, &naos, &nocc,
               &alpha,
               temp.data<double>(), &naos,
               this->psi_ks_vec[0].get_pointer(0), &naos,
               &beta, dm_hole[ik].data<double>(), &naos);
    }
    return {dm_hole, total_weight};
}

template <>
std::pair<std::vector<container::Tensor>, double>
ExcitonPlotter<double>::cal_effective_dmk_elec(const int istate)
{
    ModuleBase::TITLE("ExcitonPlotter", "cal_effective_dmk_elec<double>");
    assert(this->nspin_x == 1);
    const int offset_b = istate * this->ldim;
    const int nks = this->nk;
    const int naos = this->naos;
    const int nocc = this->nocc[0];
    const int nvirt = this->nvirt[0];

    std::vector<container::Tensor> dm_elec(nks,
        container::Tensor(DAT::DT_DOUBLE, DEV::CpuDevice, {naos, naos}));
    for (auto& dm : dm_elec) ModuleBase::GlobalFunc::ZEROS(dm.data<double>(), naos * naos);

    const double alpha = 1.0, beta = 0.0;
    double total_weight = 0.0;
    char transa, transb;

    for (int ik = 0; ik < nks; ++ik)
    {
        this->psi_ks_vec[0].fix_k(ik);
        const int x_start = ik * nocc * nvirt;

        // N_k = X_k * X_k^T → nvirt x nvirt symmetric
        container::Tensor N_k(DAT::DT_DOUBLE, DEV::CpuDevice, {nvirt, nvirt});
        ModuleBase::GlobalFunc::ZEROS(N_k.data<double>(), nvirt * nvirt);
        transa = 'N'; transb = 'T';
        dgemm_(&transa, &transb, &nvirt, &nvirt, &nocc,
               &alpha,
               this->X + offset_b + x_start, &nvirt,
               this->X + offset_b + x_start, &nvirt,
               &beta, N_k.data<double>(), &nvirt);

        for (int c = 0; c < nvirt; ++c)
            total_weight += N_k.data<double>()[c * nvirt + c];

        // temp = C_virt * N_k → naos x nvirt
        container::Tensor temp(DAT::DT_DOUBLE, DEV::CpuDevice, {naos, nvirt});
        transa = 'N'; transb = 'N';
        dgemm_(&transa, &transb, &naos, &nvirt, &nvirt,
               &alpha,
               this->psi_ks_vec[0].get_pointer(nocc), &naos,
               N_k.data<double>(), &nvirt,
               &beta, temp.data<double>(), &naos);

        // dm_elec = temp * C_virt^T → naos x naos
        transa = 'N'; transb = 'T';
        dgemm_(&transa, &transb, &naos, &naos, &nvirt,
               &alpha,
               temp.data<double>(), &naos,
               this->psi_ks_vec[0].get_pointer(nocc), &naos,
               &beta, dm_elec[ik].data<double>(), &naos);
    }
    return {dm_elec, total_weight};
}

template <>
void ExcitonPlotter<double>::plot_average_density(const int istate, const std::string& type)
{
    ModuleBase::TITLE("ExcitonPlotter", "plot_average_density<double>");

    auto dmk_weight = (type == "hole")
        ? cal_effective_dmk_hole(istate)
        : cal_effective_dmk_elec(istate);
    const auto& dmk = dmk_weight.first;
    double total_weight = dmk_weight.second;

    elecstate::DensityMatrix<double, double> DM(
        &this->pmat, this->nspin_x, this->kv.kvec_d, this->nk);

    for (int ik = 0; ik < this->nk; ++ik) {
        DM.set_DMK_pointer(ik, dmk[ik].data<double>());
    }

    LR_Util::initialize_DMR(DM, this->pmat, this->ucell, this->gd_, this->orb_cutoff_);
    DM.cal_DMR();

    double** rho_result = nullptr;
    LR_Util::_allocate_2order_nested_ptr(rho_result, this->nspin_x, this->rho_basis.nrxx);
    for (int is = 0; is < this->nspin_x; ++is) {
        ModuleBase::GlobalFunc::ZEROS(rho_result[is], this->rho_basis.nrxx);
    }

    ModuleGint::cal_gint_rho(DM.get_DMR_vector(), this->nspin_x, rho_result, false);

    if (total_weight > 0.0) {
        for (int is = 0; is < this->nspin_x; ++is) {
            for (int ixx = 0; ixx < this->rho_basis.nrxx; ++ixx) {
                rho_result[is][ixx] /= total_weight;
            }
        }
    }

    for (int is = 0; is < this->nspin_x; ++is) {
        std::string fn = PARAM.globalv.global_out_dir + "/Exciton_avg_"
                       + type + "_state" + std::to_string(istate)
                       + "_spin" + std::to_string(is) + ".cube";
        ModuleIO::write_vdata_palgrid(this->Pgrid,
            rho_result[is], is, this->nspin_x,
            0, fn,
            this->eig[istate],
            &this->ucell);
    }

    LR_Util::_deallocate_2order_nested_ptr(rho_result, this->nspin_x);
    std::cout << "Average " << type << " density written for state " << istate
              << ", total weight = " << total_weight << std::endl;
}

// ======================== complex<double> specializations ========================

template <>
std::pair<std::vector<container::Tensor>, double>
ExcitonPlotter<std::complex<double>>::cal_effective_dmk_hole(const int istate)
{
    ModuleBase::TITLE("ExcitonPlotter", "cal_effective_dmk_hole");
    assert(this->nspin_x == 1);
    const int offset_b = istate * this->ldim;
    const int nks = this->nk;
    const int naos = this->naos;
    const int nocc = this->nocc[0];
    const int nvirt = this->nvirt[0];

    std::vector<container::Tensor> dm_hole(nks,
        container::Tensor(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, {naos, naos}));
    for (auto& dm : dm_hole) ModuleBase::GlobalFunc::ZEROS(dm.data<std::complex<double>>(), naos * naos);

    const std::complex<double> alpha(1.0, 0.0);
    const std::complex<double> beta(0.0, 0.0);
    double total_weight = 0.0;

    for (int ik = 0; ik < nks; ++ik)
    {
        this->psi_ks_vec[0].fix_k(ik);
        const int x_start = ik * nocc * nvirt;

        // Step 1: M_k = X_k^H * X_k  →  nocc x nocc Hermitian
        container::Tensor M_k(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, {nocc, nocc});
        ModuleBase::GlobalFunc::ZEROS(M_k.data<std::complex<double>>(), nocc * nocc);
        char transa = 'C', transb = 'N';
        zgemm_(&transa, &transb, &nocc, &nocc, &nvirt,
               &alpha,
               this->X + offset_b + x_start, &nvirt,   // X_k: [nvirt x nocc] col-major
               this->X + offset_b + x_start, &nvirt,
               &beta, M_k.data<std::complex<double>>(), &nocc);

        // accumulate trace = Sum_{v,c} |A_{kvc}|^2
        for (int v = 0; v < nocc; ++v)
            total_weight += std::real(M_k.data<std::complex<double>>()[v * nocc + v]);

        // Step 2: temp = C_occ * M_k  →  naos x nocc
        container::Tensor temp(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, {naos, nocc});
        transa = 'N'; transb = 'N';
        zgemm_(&transa, &transb, &naos, &nocc, &nocc,
               &alpha,
               this->psi_ks_vec[0].get_pointer(0), &naos,  // C_occ: [naos x nocc]
               M_k.data<std::complex<double>>(), &nocc,
               &beta, temp.data<std::complex<double>>(), &naos);

        // Step 3: dm_hole(ik) = temp * C_occ^H  →  naos x naos
        transa = 'N'; transb = 'C';
        zgemm_(&transa, &transb, &naos, &naos, &nocc,
               &alpha,
               temp.data<std::complex<double>>(), &naos,
               this->psi_ks_vec[0].get_pointer(0), &naos,  // C_occ^H: [nocc x naos]
               &beta, dm_hole[ik].data<std::complex<double>>(), &naos);
    }

#ifdef __MPI
    Parallel_Reduce::reduce_all(total_weight);
#endif
    return {dm_hole, total_weight};
}

template <>
std::pair<std::vector<container::Tensor>, double>
ExcitonPlotter<std::complex<double>>::cal_effective_dmk_elec(const int istate)
{
    ModuleBase::TITLE("ExcitonPlotter", "cal_effective_dmk_elec");
    assert(this->nspin_x == 1);
    const int offset_b = istate * this->ldim;
    const int nks = this->nk;
    const int naos = this->naos;
    const int nocc = this->nocc[0];
    const int nvirt = this->nvirt[0];

    std::vector<container::Tensor> dm_elec(nks,
        container::Tensor(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, {naos, naos}));
    for (auto& dm : dm_elec) ModuleBase::GlobalFunc::ZEROS(dm.data<std::complex<double>>(), naos * naos);

    const std::complex<double> alpha(1.0, 0.0);
    const std::complex<double> beta(0.0, 0.0);
    double total_weight = 0.0;

    for (int ik = 0; ik < nks; ++ik)
    {
        this->psi_ks_vec[0].fix_k(ik);
        const int x_start = ik * nocc * nvirt;

        // Step 1: N_k = X_k * X_k^H  →  nvirt x nvirt Hermitian
        container::Tensor N_k(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, {nvirt, nvirt});
        ModuleBase::GlobalFunc::ZEROS(N_k.data<std::complex<double>>(), nvirt * nvirt);
        char transa = 'N', transb = 'C';
        zgemm_(&transa, &transb, &nvirt, &nvirt, &nocc,
               &alpha,
               this->X + offset_b + x_start, &nvirt,
               this->X + offset_b + x_start, &nvirt,
               &beta, N_k.data<std::complex<double>>(), &nvirt);

        // accumulate trace = Sum_{v,c} |A_{kvc}|^2
        for (int c = 0; c < nvirt; ++c)
            total_weight += std::real(N_k.data<std::complex<double>>()[c * nvirt + c]);

        // Step 2: temp = C_virt * N_k  →  naos x nvirt
        container::Tensor temp(DAT::DT_COMPLEX_DOUBLE, DEV::CpuDevice, {naos, nvirt});
        transa = 'N'; transb = 'N';
        zgemm_(&transa, &transb, &naos, &nvirt, &nvirt,
               &alpha,
               this->psi_ks_vec[0].get_pointer(nocc), &naos,  // C_virt: [naos x nvirt]
               N_k.data<std::complex<double>>(), &nvirt,
               &beta, temp.data<std::complex<double>>(), &naos);

        // Step 3: dm_elec(ik) = temp * C_virt^H  →  naos x naos
        transa = 'N'; transb = 'C';
        zgemm_(&transa, &transb, &naos, &naos, &nvirt,
               &alpha,
               temp.data<std::complex<double>>(), &naos,
               this->psi_ks_vec[0].get_pointer(nocc), &naos,
               &beta, dm_elec[ik].data<std::complex<double>>(), &naos);
    }

#ifdef __MPI
    Parallel_Reduce::reduce_all(total_weight);
#endif
    return {dm_elec, total_weight};
}

template <>
void ExcitonPlotter<std::complex<double>>::plot_average_density(const int istate, const std::string& type)
{
    ModuleBase::TITLE("ExcitonPlotter", "plot_average_density");

    auto dmk_weight = (type == "hole")
        ? cal_effective_dmk_hole(istate)
        : cal_effective_dmk_elec(istate);
    const auto& dmk = dmk_weight.first;
    double total_weight = dmk_weight.second;

    // Build DensityMatrix<complex<double>, double> — TR=double means cal_DMR()
    // will compute Re[DMR] = Sum_k [cos(kR)*Re(DMK) - sin(kR)*Im(DMK)],
    // which is exactly what we need for the real-space density.
    elecstate::DensityMatrix<std::complex<double>, double> DM(
        &this->pmat, this->nspin_x, this->kv.kvec_d, this->nk);

    // Copy effective DMK into the DensityMatrix for each k-point
    // set_DMK_pointer COPIES the data, so dmk tensors can be freed afterwards
    for (int ik = 0; ik < this->nk; ++ik) {
        DM.set_DMK_pointer(ik, dmk[ik].data<std::complex<double>>());
    }

    // Initialize DMR structure (atom-pair/R-vector layout) and compute DMR
    LR_Util::initialize_DMR(DM, this->pmat, this->ucell, this->gd_, this->orb_cutoff_);
    DM.cal_DMR();

    // Allocate and compute real-space density via gint
    double** rho_result = nullptr;
    LR_Util::_allocate_2order_nested_ptr(rho_result, this->nspin_x, this->rho_basis.nrxx);
    for (int is = 0; is < this->nspin_x; ++is) {
        ModuleBase::GlobalFunc::ZEROS(rho_result[is], this->rho_basis.nrxx);
    }

    // cal_gint_rho computes rho(r) = Sum_{mu,nu,R} DMR_re(mu,nu,R) * phi_mu * phi_nu
    // For Hermitian DMR, this IS the correct density — no squaring needed.
    ModuleGint::cal_gint_rho(DM.get_DMR_vector(), this->nspin_x, rho_result, false);

    // Normalize by total weight if requested
    if (total_weight > 0.0) {
        for (int is = 0; is < this->nspin_x; ++is) {
            for (int ixx = 0; ixx < this->rho_basis.nrxx; ++ixx) {
                rho_result[is][ixx] /= total_weight;
            }
        }
    }

    // Write cube files
    for (int is = 0; is < this->nspin_x; ++is) {
        std::string fn = PARAM.globalv.global_out_dir + "/Exciton_avg_"
                       + type + "_state" + std::to_string(istate)
                       + "_spin" + std::to_string(is) + ".cube";
        ModuleIO::write_vdata_palgrid(this->Pgrid,
            rho_result[is], is, this->nspin_x,
            0, fn,
            this->eig[istate],
            &this->ucell);
    }

    LR_Util::_deallocate_2order_nested_ptr(rho_result, this->nspin_x);
    std::cout << "Average " << type << " density written for state " << istate
              << ", total weight = " << total_weight << std::endl;
}

template <>
void ExcitonPlotter<double>::plot_average_slice(
    const int istate, const std::string& type,
    const std::string& plane, double slice_pos, int npoints, double scale)
{
    ModuleBase::TITLE("ExcitonPlotter", "plot_average_slice<double>");
    assert(this->nspin_x == 1);
    if (!orb_)
    {
        ModuleBase::WARNING_QUIT("ExcitonPlotter",
            "plot_average_slice requires LCAO_Orbitals; pass orb to constructor.");
    }
    if (type != "hole" && type != "elec")
    {
        ModuleBase::WARNING_QUIT("ExcitonPlotter",
            "Unknown average density type: " + type + ". Use hole or elec.");
    }

    auto dmk_weight = (type == "hole")
        ? cal_effective_dmk_hole(istate)
        : cal_effective_dmk_elec(istate);
    const auto& dmk = dmk_weight.first;
    const double total_weight = dmk_weight.second;

    const SliceGeometry geom = make_slice_geometry(
        this->ucell, this->kv, this->nk, plane, slice_pos, npoints, scale);
    const double du = (geom.u_end - geom.u_start) / (geom.nu - 1);
    const double dv = (geom.v_end - geom.v_start) / (geom.nv - 1);

    std::vector<double> density(geom.nu * geom.nv, 0.0);
    std::vector<std::complex<double>> phi_bloch(this->naos);
    const int cell_res = geom.res;
    std::vector<double> rho_cache(cell_res * cell_res, 0.0);
    std::vector<bool> cache_valid(cell_res * cell_res, false);

    for (int iu = 0; iu < geom.nu; ++iu)
    {
        const double u_frac = geom.u_start + du * iu;
        const double u_cell = u_frac - std::floor(u_frac);
        int uc_idx = static_cast<int>(std::round(u_cell * cell_res));
        if (uc_idx >= cell_res) uc_idx -= cell_res;
        for (int iv = 0; iv < geom.nv; ++iv)
        {
            const double v_frac = geom.v_start + dv * iv;
            const double v_cell = v_frac - std::floor(v_frac);
            int vc_idx = static_cast<int>(std::round(v_cell * cell_res));
            if (vc_idx >= cell_res) vc_idx -= cell_res;

            const int cache_idx = uc_idx * cell_res + vc_idx;
            if (!cache_valid[cache_idx])
            {
                const Vec3 r = geom.perp_offset + geom.u_vec * u_cell + geom.v_vec * v_cell;

                double rho = 0.0;
                for (int ik = 0; ik < this->nk; ++ik)
                {
                    std::fill(phi_bloch.begin(), phi_bloch.end(), std::complex<double>(0.0, 0.0));
                    const Vec3 kvec_d = ik < static_cast<int>(this->kv.kvec_d.size())
                        ? this->kv.kvec_d[ik]
                        : Vec3(0.0, 0.0, 0.0);
                    orb_eval_.eval_phi_all_bloch(r, this->ucell, phi_bloch.data(), kvec_d);
                    rho += density_from_dmk(dmk[ik].data<double>(), phi_bloch, this->naos);
                }
                rho_cache[cache_idx] = total_weight > 0.0 ? rho / total_weight : rho;
                cache_valid[cache_idx] = true;
            }
            density[iu * geom.nv + iv] = rho_cache[cache_idx];
        }
    }

    const std::string fn = PARAM.globalv.global_out_dir + "/Exciton_avg_"
                         + type + "_slice_state" + std::to_string(istate) + ".dat";
    write_slice_data(this->ucell, geom, density, fn, istate, this->eig[istate],
                     "average_" + type, false, {0.0, 0.0, 0.0});
    std::cout << "Average " << type << " density slice written to " << fn
              << " (" << geom.nu << "x" << geom.nv << ", "
              << geom.res << " pts/cell, total weight = "
              << total_weight << ")" << std::endl;
}

template <>
void ExcitonPlotter<std::complex<double>>::plot_average_slice(
    const int istate, const std::string& type,
    const std::string& plane, double slice_pos, int npoints, double scale)
{
    ModuleBase::TITLE("ExcitonPlotter", "plot_average_slice");
    assert(this->nspin_x == 1);
    if (!orb_)
    {
        ModuleBase::WARNING_QUIT("ExcitonPlotter",
            "plot_average_slice requires LCAO_Orbitals; pass orb to constructor.");
    }
    if (type != "hole" && type != "elec")
    {
        ModuleBase::WARNING_QUIT("ExcitonPlotter",
            "Unknown average density type: " + type + ". Use hole or elec.");
    }

    auto dmk_weight = (type == "hole")
        ? cal_effective_dmk_hole(istate)
        : cal_effective_dmk_elec(istate);
    const auto& dmk = dmk_weight.first;
    const double total_weight = dmk_weight.second;

    const SliceGeometry geom = make_slice_geometry(
        this->ucell, this->kv, this->nk, plane, slice_pos, npoints, scale);
    const double du = (geom.u_end - geom.u_start) / (geom.nu - 1);
    const double dv = (geom.v_end - geom.v_start) / (geom.nv - 1);

    std::vector<double> density(geom.nu * geom.nv, 0.0);
    std::vector<std::complex<double>> phi_bloch(this->naos);
    const int cell_res = geom.res;
    std::vector<double> rho_cache(cell_res * cell_res, 0.0);
    std::vector<bool> cache_valid(cell_res * cell_res, false);

    for (int iu = 0; iu < geom.nu; ++iu)
    {
        const double u_frac = geom.u_start + du * iu;
        const double u_cell = u_frac - std::floor(u_frac);
        int uc_idx = static_cast<int>(std::round(u_cell * cell_res));
        if (uc_idx >= cell_res) uc_idx -= cell_res;
        for (int iv = 0; iv < geom.nv; ++iv)
        {
            const double v_frac = geom.v_start + dv * iv;
            const double v_cell = v_frac - std::floor(v_frac);
            int vc_idx = static_cast<int>(std::round(v_cell * cell_res));
            if (vc_idx >= cell_res) vc_idx -= cell_res;

            const int cache_idx = uc_idx * cell_res + vc_idx;
            if (!cache_valid[cache_idx])
            {
                const Vec3 r = geom.perp_offset + geom.u_vec * u_cell + geom.v_vec * v_cell;

                double rho = 0.0;
                for (int ik = 0; ik < this->nk; ++ik)
                {
                    std::fill(phi_bloch.begin(), phi_bloch.end(), std::complex<double>(0.0, 0.0));
                    orb_eval_.eval_phi_all_bloch(r, this->ucell, phi_bloch.data(), this->kv.kvec_d[ik]);
                    rho += density_from_dmk(dmk[ik].data<std::complex<double>>(), phi_bloch, this->naos);
                }
                rho_cache[cache_idx] = total_weight > 0.0 ? rho / total_weight : rho;
                cache_valid[cache_idx] = true;
            }
            density[iu * geom.nv + iv] = rho_cache[cache_idx];
        }
    }

    const std::string fn = PARAM.globalv.global_out_dir + "/Exciton_avg_"
                         + type + "_slice_state" + std::to_string(istate) + ".dat";
    write_slice_data(this->ucell, geom, density, fn, istate, this->eig[istate],
                     "average_" + type, false, {0.0, 0.0, 0.0});
    std::cout << "Average " << type << " density slice written to " << fn
              << " (" << geom.nu << "x" << geom.nv << ", "
              << geom.res << " pts/cell, total weight = "
              << total_weight << ")" << std::endl;
}

// ============================================================================
// Conditional density — coherent real-space evaluation (no DMR pipeline)
// ============================================================================
// Unlike average density, conditional density requires coherent sum over k-points:
//   psi_cond(r) = Sum_{k,c} b_{k,c} * psi_{ck}(r)
//   rho_cond(r) = |psi_cond(r)|^2
// The DMR pipeline CANNOT be used because it forces unit-cell periodicity.
// ============================================================================

template <>
void ExcitonPlotter<double>::plot_conditional_density(
    const int istate, const std::array<double, 3>& r_fix, const std::string& type)
{
    ModuleBase::TITLE("ExcitonPlotter", "plot_conditional_density<double>");
    ModuleBase::WARNING_QUIT("ExcitonPlotter",
        "conditional density requires complex wavefunctions (not double/gamma-only).");
}

template <>
void ExcitonPlotter<std::complex<double>>::plot_conditional_density(
    const int istate, const std::array<double, 3>& r_fix_in, const std::string& type)
{
    using T = std::complex<double>;
    ModuleBase::TITLE("ExcitonPlotter", "plot_conditional_density");
    assert(this->nspin_x == 1);
    if (!orb_) {
        ModuleBase::WARNING_QUIT("ExcitonPlotter",
            "plot_conditional_density requires LCAO_Orbitals; pass orb to constructor.");
    }

    const bool plot_elec = (type == "elec");
    if (!plot_elec && type != "hole") {
        ModuleBase::WARNING_QUIT("ExcitonPlotter",
            "Unknown conditional density type: " + type + ". Use elec or hole.");
    }
    const ModuleBase::Vector3<double> r_fix(r_fix_in[0], r_fix_in[1], r_fix_in[2]);
    const int offset_b = istate * this->ldim;
    const int nks = this->nk, naos = this->naos;
    const int nocc = this->nocc[0], nvirt = this->nvirt[0];
    const int nx = this->rho_basis.nx, ny = this->rho_basis.ny, nz = this->rho_basis.nz;
    const int nrxx = this->rho_basis.nrxx; // nx * ny * nz

    // For conditional electron density, mix conduction states with psi_v^*(r_h).
    // For conditional hole density, mix valence states with psi_c(r_e), then
    // evaluate the complex-conjugated valence wavefunction.
    const int nmix = plot_elec ? nvirt : nocc;
    std::vector<std::vector<T>> mix_all(nks, std::vector<T>(nmix, T(0)));
    for (int ik = 0; ik < nks; ++ik)
    {
        const int x_start = ik * nocc * nvirt;
        if (plot_elec)
        {
            for (int v = 0; v < nocc; ++v)
            {
                const T psi_hole = std::conj(
                    orb_eval_.eval_wfc_bloch<T>(r_fix, ik, v, this->psi_ks_vec[0],
                                                this->ucell, this->kv.kvec_d[ik]));
                for (int c = 0; c < nvirt; ++c)
                    mix_all[ik][c] += this->X[offset_b + x_start + c + v * nvirt] * psi_hole;
            }
        }
        else
        {
            for (int c = 0; c < nvirt; ++c)
            {
                const T psi_elec = orb_eval_.eval_wfc_bloch<T>(
                    r_fix, ik, nocc + c, this->psi_ks_vec[0],
                    this->ucell, this->kv.kvec_d[ik]);
                for (int v = 0; v < nocc; ++v)
                    mix_all[ik][v] += this->X[offset_b + x_start + c + v * nvirt] * psi_elec;
            }
        }
    }

    // Step 2: form one effective AO coefficient vector for each k-point.
    std::vector<std::vector<T>> eff_C_all(nks, std::vector<T>(naos, T(0)));
    const T alpha(1.0, 0.0), beta(0.0, 0.0);
    int inc_one = 1;
    char trans = 'N';

    for (int ik = 0; ik < nks; ++ik)
    {
        this->psi_ks_vec[0].fix_k(ik);
        if (!plot_elec)
            for (T& value : mix_all[ik]) value = std::conj(value);
        const int band_start = plot_elec ? nocc : 0;
        zgemv_(&trans, &naos, &nmix, &alpha,
               this->psi_ks_vec[0].get_pointer(band_start), &naos,
               mix_all[ik].data(), &inc_one,
               &beta, eff_C_all[ik].data(), &inc_one);
    }

    // Step 3: evaluate psi_k(r) on real-space grid, coherent sum, then square
    //   psi_cond(r) = Sum_k psi_k(r)
    //   rho_cond(r) = |psi_cond(r)|^2
    //
    // Grid positions: r(i,j,k) = i*dx + j*dy + k*dz (i=0..nx-1, j=0..ny-1, k=0..nz-1)
    ModuleBase::Vector3<double> a = this->ucell.a1 * this->ucell.lat0;
    ModuleBase::Vector3<double> b = this->ucell.a2 * this->ucell.lat0;
    ModuleBase::Vector3<double> c = this->ucell.a3 * this->ucell.lat0;
    const double inv_nx = 1.0 / nx, inv_ny = 1.0 / ny, inv_nz = 1.0 / nz;
    ModuleBase::Vector3<double> dx = a * inv_nx;
    ModuleBase::Vector3<double> dy = b * inv_ny;
    ModuleBase::Vector3<double> dz = c * inv_nz;

    const auto mesh = bvk_mesh_from_kpoints(this->kv, this->nk);
    const int nk1 = mesh[0], nk2 = mesh[1], nk3 = mesh[2];
    const bool write_supercell = (nk1 > 1 || nk2 > 1 || nk3 > 1);
    const int snx = nx * nk1, sny = ny * nk2, snz = nz * nk3;

    double** rho_result = nullptr;
    LR_Util::_allocate_2order_nested_ptr(rho_result, this->nspin_x, nrxx);
    ModuleBase::GlobalFunc::ZEROS(rho_result[0], nrxx);
    std::cout << "Conditional density: coherent sum over " << nks << " k-points"
              << ", BvK supercell " << nk1 << "x" << nk2 << "x" << nk3
              << ", grid " << snx << "x" << sny << "x" << snz << std::endl;

    // Allocate supercell cube data (if writing supercell)
    std::vector<double> supercell_data;
    if (write_supercell)
        supercell_data.resize(snx * sny * snz, 0.0);

    std::vector<std::vector<std::complex<double>>> psi_k_cache;
    if (write_supercell)
        psi_k_cache.resize(nks, std::vector<std::complex<double>>(nrxx));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int ix = 0; ix < nx; ++ix)
    {
        std::vector<std::complex<double>> phi_bloch(naos);
        ModuleBase::Vector3<double> pos;
        for (int iy = 0; iy < ny; ++iy)
        {
            for (int iz = 0; iz < nz; ++iz)
            {
                const int idx = ix * ny * nz + iy * nz + iz;
                pos = dx * (double)ix + dy * (double)iy + dz * (double)iz;

                std::complex<double> psi_cond(0.0, 0.0);
                for (int ik = 0; ik < nks; ++ik)
                {
                    std::fill(phi_bloch.begin(), phi_bloch.end(), std::complex<double>(0.0, 0.0));
                    orb_eval_.eval_phi_all_bloch(pos, this->ucell, phi_bloch.data(),
                                                 this->kv.kvec_d[ik]);
                    const auto* coeff = eff_C_all[ik].data();
                    std::complex<double> psi_k(0.0, 0.0);
                    for (int mu = 0; mu < naos; ++mu)
                        psi_k += coeff[mu] * phi_bloch[mu];
                    if (!plot_elec) psi_k = std::conj(psi_k);
                    if (write_supercell)
                        psi_k_cache[ik][idx] = psi_k;
                    psi_cond += psi_k;
                }
                rho_result[0][idx] = std::norm(psi_cond);
            }
        }
    }
    std::cout << "Home cell coherent evaluation complete." << std::endl;

    // Step 4: write cube file(s)
    ModuleIO::write_vdata_palgrid(this->Pgrid,
        rho_result[0], 0, this->nspin_x,
        0, PARAM.globalv.global_out_dir + "/Exciton_cond_" + type + "_state"
           + std::to_string(istate) + "_spin0_home.cube",
        this->eig[istate], &this->ucell);

    // Step 5: write BvK supercell cube if multi-k
    if (write_supercell)
    {
        std::cout << "Writing BvK supercell cube (" << snx << "x" << sny << "x" << snz << ")..." << std::endl;

        // Supercell lattice vectors
        ModuleBase::Vector3<double> sa = a * (double)nk1;
        ModuleBase::Vector3<double> sb = b * (double)nk2;
        ModuleBase::Vector3<double> sc = c * (double)nk3;

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(dynamic)
#endif
        for (int ni = 0; ni < nk1; ++ni)
        {
            for (int nj = 0; nj < nk2; ++nj)
            {
                for (int nk = 0; nk < nk3; ++nk)
                {
                    for (int ix = 0; ix < nx; ++ix)
                    {
                        for (int iy = 0; iy < ny; ++iy)
                        {
                            for (int iz = 0; iz < nz; ++iz)
                            {
                                const int h_idx = ix * ny * nz + iy * nz + iz;
                                std::complex<double> psi(0.0, 0.0);
                                for (int ik = 0; ik < nks; ++ik)
                                {
                                    // Bloch phase for integer cell translation ni*a + nj*b + nk*c.
                                    double kdotR = bloch_phase_arg(this->kv.kvec_d[ik], ni, nj, nk);
                                    if (!plot_elec) kdotR = -kdotR;
                                    std::complex<double> phase(std::cos(kdotR), std::sin(kdotR));
                                    psi += phase * psi_k_cache[ik][h_idx];
                                }
                                const int s_idx = (ni * nx + ix) * sny * snz
                                                + (nj * ny + iy) * snz
                                                + (nk * nz + iz);
                                supercell_data[s_idx] = std::norm(psi);
                            }
                        }
                    }
                }
            }
        }

        // Write supercell cube file (custom writer for non-standard grid size)
        std::string sfn = PARAM.globalv.global_out_dir + "/Exciton_cond_" + type + "_state"
                        + std::to_string(istate) + "_spin0_supercell.cube";
        std::ofstream ofs(sfn);
        ofs << "BvK supercell conditional " << type << " density, state " << istate
            << "\nExciton energy (Ry) = " << this->eig[istate] << "\n";
        ofs << this->ucell.nat * nk1 * nk2 * nk3 << " 0.0 0.0 0.0\n";
        ofs << snx << " " << sa.x/snx << " " << sa.y/snx << " " << sa.z/snx << "\n";
        ofs << sny << " " << sb.x/sny << " " << sb.y/sny << " " << sb.z/sny << "\n";
        ofs << snz << " " << sc.x/snz << " " << sc.y/snz << " " << sc.z/snz << "\n";

        // Write atoms (replicated over supercell)
        for (int ia = 0; ia < this->ucell.nat; ++ia)
        {
            const int it = this->ucell.iat2it[ia];
            const ModuleBase::Vector3<double> tau = atom_position_bohr(this->ucell, ia);
            for (int ni = 0; ni < nk1; ++ni)
                for (int nj = 0; nj < nk2; ++nj)
                    for (int nk = 0; nk < nk3; ++nk)
                    {
                        double x = tau.x + ni * a.x + nj * b.x + nk * c.x;
                        double y = tau.y + ni * a.y + nj * b.y + nk * c.y;
                        double z = tau.z + ni * a.z + nj * b.z + nk * c.z;
                        ofs << atomic_number_from_label(this->ucell.atoms[it].label) << " "
                            << this->ucell.atoms[it].ncpp.zv << " "
                            << x << " " << y << " " << z << "\n";
                    }
        }

        // Write data (inner loop z, then y, then x)
        ofs << std::scientific << std::setprecision(12);
        for (int ix = 0; ix < snx; ++ix)
            for (int iy = 0; iy < sny; ++iy)
                for (int iz = 0; iz < snz; ++iz)
                    ofs << supercell_data[ix * sny * snz + iy * snz + iz] << "\n";
        ofs.close();
        std::cout << "Supercell cube written to " << sfn << std::endl;
    }

    LR_Util::_deallocate_2order_nested_ptr(rho_result, this->nspin_x);
    std::cout << "Conditional " << type << " density for state " << istate
              << ", " << (plot_elec ? "hole" : "electron") << " fixed at ("
              << r_fix_in[0] << ", " << r_fix_in[1] << ", " << r_fix_in[2]
              << ") Bohr" << std::endl;
}

// =============================================================================
// 2D cross-section conditional density — verifies BvK supercell periodicity.
//
// Algorithm:
//   1. Compute mixing coefficients b_k[c] = Sum_v A_{kvc} * psi*_{vk}(r_h_fix)
//      by evaluating the hole wavefunction at the fixed point r_h_fix.
//   2. Form effective conduction-band coefficients: eff_C_k = C_virt(k) * b_k.
//   3. For each unique home-cell fractional position (u_cell, v_cell), evaluate
//      psi_k(r_home) using the FULL Bloch-summed orbital evaluator (R-sum over
//      neighboring cells). Cache these values (typically res×res = 18×18 entries).
//   4. For each supercell grid point (u, v), apply the Bloch phase shift
//      psi_k(r) = exp(i * 2pi * k_d · R_int) * psi_k(r_home), then
//      sum coherently over k-points: psi_cond = Sum_k psi_k, rho = |psi_cond|^2.
//
// The DMR pipeline (cal_gint_rho) is deliberately AVOIDED because it forces
// unit-cell periodicity by summing over all lattice vectors in the density
// evaluation. Instead, the coherent k-point sum preserves the phase information
// that breaks unit-cell periodicity while exactly preserving BvK periodicity.
//
// Output: a data file with the 2D density grid and metadata headers suitable
// for the companion Python plotting script (tools/plot-tools/plot_cond_slice.py).
// =============================================================================

template <>
void ExcitonPlotter<double>::plot_cond_slice(
    const int, const std::array<double,3>&, const std::string&, double, int, double,
    const std::string&)
{
    ModuleBase::WARNING_QUIT("ExcitonPlotter",
        "cond_slice requires complex wavefunctions (not double/gamma-only).");
}

template <>
void ExcitonPlotter<std::complex<double>>::plot_cond_slice(
    const int istate, const std::array<double, 3>& r_fix_in,
    const std::string& plane, double slice_pos, int npoints, double scale,
    const std::string& type)
{
    using T = std::complex<double>;
    ModuleBase::TITLE("ExcitonPlotter", "plot_cond_slice");
    assert(this->nspin_x == 1);
    if (!orb_) {
        ModuleBase::WARNING_QUIT("ExcitonPlotter",
            "plot_cond_slice requires LCAO_Orbitals; pass orb to constructor.");
    }

    const bool plot_elec = (type == "elec");
    if (!plot_elec && type != "hole") {
        ModuleBase::WARNING_QUIT("ExcitonPlotter",
            "Unknown conditional slice type: " + type + ". Use elec or hole.");
    }
    const ModuleBase::Vector3<double> r_fix(r_fix_in[0], r_fix_in[1], r_fix_in[2]);
    const int nks = this->nk, naos = this->naos;
    const int nocc = this->nocc[0], nvirt = this->nvirt[0];
    const int offset_b = istate * this->ldim;

    const SliceGeometry geom = make_slice_geometry(
        this->ucell, this->kv, this->nk, plane, slice_pos, npoints, scale);

    const int nmix = plot_elec ? nvirt : nocc;
    std::vector<std::vector<T>> mix_all(nks, std::vector<T>(nmix, T(0)));
    for (int ik = 0; ik < nks; ++ik)
    {
        const int x_start = ik * nocc * nvirt;
        if (plot_elec)
        {
            for (int v = 0; v < nocc; ++v)
            {
                const T psi_hole = std::conj(
                    orb_eval_.eval_wfc_bloch<T>(r_fix, ik, v, this->psi_ks_vec[0],
                                                this->ucell, this->kv.kvec_d[ik]));
                for (int c = 0; c < nvirt; ++c)
                    mix_all[ik][c] += this->X[offset_b + x_start + c + v * nvirt] * psi_hole;
            }
        }
        else
        {
            for (int c = 0; c < nvirt; ++c)
            {
                const T psi_elec = orb_eval_.eval_wfc_bloch<T>(
                    r_fix, ik, nocc + c, this->psi_ks_vec[0],
                    this->ucell, this->kv.kvec_d[ik]);
                for (int v = 0; v < nocc; ++v)
                    mix_all[ik][v] += this->X[offset_b + x_start + c + v * nvirt] * psi_elec;
            }
        }
    }

    // eff_C_k = C_virt * b_k
    std::vector<std::vector<T>> eff_C_all(nks, std::vector<T>(naos, T(0)));
    const T alpha(1.0, 0.0), beta(0.0, 0.0);
    int inc_one = 1;
    char trans = 'N';
    for (int ik = 0; ik < nks; ++ik) {
        this->psi_ks_vec[0].fix_k(ik);
        if (!plot_elec)
            for (T& value : mix_all[ik]) value = std::conj(value);
        const int band_start = plot_elec ? nocc : 0;
        zgemv_(&trans, &naos, &nmix, &alpha,
               this->psi_ks_vec[0].get_pointer(band_start), &naos,
               mix_all[ik].data(), &inc_one, &beta, eff_C_all[ik].data(), &inc_one);
    }

    // --- Step 2: evaluate on 2D cross-section grid ---
    //
    // For each grid point (iu, iv) in [0, npoints-1]:
    //   u_frac = u_start + (u_end - u_start) * iu / (npoints - 1)
    //   v_frac = v_start + (v_end - v_start) * iv / (npoints - 1)
    //   r = perp_vec + u_frac * u_vec + v_frac * v_vec
    //
    //   Decompose into home-cell position and supercell offset:
    //     u_cell = u_frac mod 1,  R_iu = floor(u_frac)
    //     v_cell = v_frac mod 1,  R_iv = floor(v_frac)
    //     r_home = perp_vec + u_cell * u_vec + v_cell * v_vec
    //     R_int = R_iu * e_u + R_iv * e_v, with e_u/e_v determined by plane.
    //   psi_k(r) = exp(i * 2pi * k_d · R_int) * psi_k(r_home)
    //
    // Optimization: cache phi at unique home-cell (u_cell, v_cell) points
    // since only npoints unique values of each, repeated for different R offsets.

    const double du = (geom.u_end - geom.u_start) / (geom.nu - 1);
    const double dv = (geom.v_end - geom.v_start) / (geom.nv - 1);

    // Pre-compute phase advances for integer translations along the slice vectors.
    std::vector<double> kdotu(nks), kdotv(nks);
    for (int ik = 0; ik < nks; ++ik) {
        kdotu[ik] = bloch_phase_arg_axis(this->kv.kvec_d[ik], geom.u_axis, 1);
        kdotv[ik] = bloch_phase_arg_axis(this->kv.kvec_d[ik], geom.v_axis, 1);
    }

    std::vector<double> density(geom.nu * geom.nv, 0.0);
    std::vector<T> phi_bloch(naos);
    // Cache psi_k at unique home-cell fractional positions (u_cell, v_cell)
    const int cell_res = geom.res; // grid points per cell
    std::vector<std::vector<T>> psi_cache(cell_res * cell_res, std::vector<T>(nks, T(0)));
    std::vector<bool> cache_valid(cell_res * cell_res, false);

    for (int iu = 0; iu < geom.nu; ++iu)
    {
        double u_frac = geom.u_start + du * iu;
        double u_cell = u_frac - std::floor(u_frac);
        int R_iu = (int)std::floor(u_frac);
        int uc_idx = (int)std::round(u_cell * cell_res);
        if (uc_idx >= cell_res) uc_idx -= cell_res;

        for (int iv = 0; iv < geom.nv; ++iv)
        {
            double v_frac = geom.v_start + dv * iv;
            double v_cell = v_frac - std::floor(v_frac);
            int R_iv = (int)std::floor(v_frac);
            int vc_idx = (int)std::round(v_cell * cell_res);
            if (vc_idx >= cell_res) vc_idx -= cell_res;

            int cache_idx = uc_idx * cell_res + vc_idx;

            // Evaluate psi_k at home cell using full Bloch sum, cache by (uc,vc)
            if (!cache_valid[cache_idx]) {
                ModuleBase::Vector3<double> r_home = geom.perp_offset
                    + geom.u_vec * u_cell + geom.v_vec * v_cell;
                for (int ik = 0; ik < nks; ++ik) {
                    std::fill(phi_bloch.begin(), phi_bloch.end(), T(0));
                    orb_eval_.eval_phi_all_bloch(r_home, this->ucell, phi_bloch.data(),
                                                 this->kv.kvec_d[ik]);
                    T psi_k(0);
                    auto* ck = eff_C_all[ik].data();
                    for (int mu = 0; mu < naos; ++mu)
                        psi_k += ck[mu] * phi_bloch[mu];
                    if (!plot_elec) psi_k = std::conj(psi_k);
                    psi_cache[cache_idx][ik] = psi_k;
                }
                cache_valid[cache_idx] = true;
            }

            // Shift to supercell position via Bloch theorem
            T psi_cond(0);
            for (int ik = 0; ik < nks; ++ik) {
                double phase_arg = kdotu[ik] * R_iu + kdotv[ik] * R_iv;
                if (!plot_elec) phase_arg = -phase_arg;
                T phase(std::cos(phase_arg), std::sin(phase_arg));
                psi_cond += phase * psi_cache[cache_idx][ik];
            }
            density[iu * geom.nv + iv] = std::norm(psi_cond);
        }
    }
    std::cout << "Bloch-sum evaluation complete (cached " << cell_res << "x" << cell_res << " home-cell positions)." << std::endl;

    std::string fn = PARAM.globalv.global_out_dir + "/Exciton_cond_" + type + "_slice_state"
                   + std::to_string(istate) + ".dat";
    write_slice_data(this->ucell, geom, density, fn, istate, this->eig[istate],
                     "conditional_" + type, true, r_fix_in);

    std::cout << "Conditional density slice written to " << fn
              << " (" << geom.nu << "x" << geom.nv
              << ", range " << scale << "x BvK, "
              << geom.res << " pts/cell)" << std::endl;
}

} //namespace LR_Util
