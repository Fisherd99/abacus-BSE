#pragma once
#include "source_base/tool_title.h"
#include "source_cell/klist.h"
#include "source_estate/module_dm/density_matrix.h"
#include "source_io/module_parameter/parameter.h"
#include "source_io/cube_io.h"
#include "source_lcao/module_lr/dm_trans/dm_trans.h"
#include "source_lcao/module_lr/utils/lr_util.h"
#include "source_lcao/module_lr/utils/lr_util_hcontainer.h"
#include "source_lcao/module_gint/temp_gint/gint_interface.h"
#include "source_basis/module_ao/ORB_read.h"
#include "source_base/ylm.h"
#include "source_cell/atom_spec.h"
#include <string>
#include <array>
#include <complex>
#include <vector>

// ============================================================================
// OrbitalEvaluator: single-point NAO wavefunction evaluation
// Replicates the cubic Hermite interpolation from GintAtom::set_phi()
// ============================================================================
struct OrbitalEvaluator
{
    struct AtomTypeData
    {
        int nw = 0, nwl = 0;               ///< number of NAOs, max L for this atom type
        double rcut = 0.0, dr_uniform = 0.0; ///< orbital cutoff radius, uniform grid spacing
        std::vector<int> iw2_ylm;           ///< spherical-harmonic index per orbital
        std::vector<bool> iw2_new;          ///< true for m==0 orbitals (radial interp done once)
        std::vector<const double*> psi_ptrs; ///< pointers to uniform radial function data
        std::vector<const double*> dpsi_ptrs;///< pointers to uniform radial derivative data
    };
    std::vector<AtomTypeData> atypes_;

    /// @brief Initialize evaluator by extracting radial function pointers from LCAO_Orbitals
    /// @param orb numerical atomic orbital data (one Numerical_Orbital per atom type)
    void init(const LCAO_Orbitals& orb, const UnitCell& ucell);

    /// @brief Evaluate all NAO values for one atom type at relative position dr (in Bohr)
    /// @param it atom type index
    /// @param dr Cartesian vector from atom center to evaluation point (Bohr)
    /// @param phi_out array of length nw[it], filled with phi values
    void eval_phi(int it, const ModuleBase::Vector3<double>& dr, double* phi_out) const;

    /// @brief Evaluate LCAO wavefunction psi_{iband, ik}(r) at a single real-space point
    /// @param r_fix evaluation point in Bohr
    /// @return complex wavefunction value psi_{iband, ik}(r_fix)
    template <typename T>
    T eval_wfc(const ModuleBase::Vector3<double>& r_fix, int ik, int iband,
               const psi::Psi<T>& psi_ks, const UnitCell& ucell) const
    {
        psi_ks.fix_k(ik);
        std::vector<double> phi_atom;
        T result(0);
        int iw_global = 0;
        for (int iat = 0; iat < ucell.nat; ++iat)
        {
            const int it = ucell.iat2it[iat];
            const ModuleBase::Vector3<double> tau = ucell.get_tau(iat) * ucell.lat0;
            const int nw_atom = atypes_[it].nw;
            phi_atom.resize(nw_atom);
            eval_phi(it, r_fix - tau, phi_atom.data());
            for (int iw = 0; iw < nw_atom; ++iw, ++iw_global)
                result += psi_ks(iband, iw_global) * static_cast<T>(phi_atom[iw]);
        }
        return result;
    }

    /// @brief Evaluate LCAO wavefunction at a point using custom coefficients
    /// psi(r) = Sum_mu coeff[mu] * phi_mu(r).
    /// Unlike eval_wfc, this uses an explicit coefficient vector instead of
    /// extracting coefficients from psi_ks for a specific band.
    /// @tparam T scalar type (double or std::complex<double>)
    /// @param r_fix evaluation point in Bohr
    /// @param coeff custom coefficient vector, length = total naos
    /// @param ucell unit cell
    /// @return complex wavefunction value
    template <typename T>
    T eval_wfc_vec(const ModuleBase::Vector3<double>& r_fix, const T* coeff,
                   const UnitCell& ucell) const
    {
        std::vector<double> phi_atom;
        T result(0);
        int iw_global = 0;
        for (int iat = 0; iat < ucell.nat; ++iat)
        {
            const int it = ucell.iat2it[iat];
            const ModuleBase::Vector3<double> tau = ucell.get_tau(iat) * ucell.lat0;
            const int nw_atom = atypes_[it].nw;
            phi_atom.resize(nw_atom);
            eval_phi(it, r_fix - tau, phi_atom.data());
            for (int iw = 0; iw < nw_atom; ++iw, ++iw_global)
                result += coeff[iw_global] * static_cast<T>(phi_atom[iw]);
        }
        return result;
    }

    /// @brief Evaluate all NAO values (all atoms, all orbitals) at a single point
    /// Writes concatenated real-valued phi for every orbital in the system.
    /// R=0 only — does NOT include Bloch sum over neighboring cells.
    /// @param r_fix evaluation point in Bohr
    /// @param ucell unit cell
    /// @param phi_all output array of length total_naos, filled with phi values
    void eval_phi_all(const ModuleBase::Vector3<double>& r_fix, const UnitCell& ucell,
                      double* phi_all) const
    {
        int iw_global = 0;
        for (int iat = 0; iat < ucell.nat; ++iat)
        {
            const int it = ucell.iat2it[iat];
            const ModuleBase::Vector3<double> tau = ucell.get_tau(iat) * ucell.lat0;
            const int nw_atom = atypes_[it].nw;
            eval_phi(it, r_fix - tau, phi_all + iw_global);
            iw_global += nw_atom;
        }
    }

    /// @brief Evaluate Bloch-summed NAO values including neighboring cell contributions
    /// phi^B_mu(r, k) = Sum_R exp(i * 2pi * k_d·R_int) * phi_mu(r - tau - R).
    /// The R-sum runs over all lattice vectors within the orbital cutoff radius.
    /// Output is complex because the Bloch phase contributes a complex factor.
    /// The caller must zero-initialize phi_bloch before calling (this function accumulates).
    /// @param r evaluation point in Bohr
    /// @param ucell unit cell
    /// @param phi_bloch output array of length total_naos (complex), values are added
    /// @param kvec_d direct-coordinate k-vector
    void eval_phi_all_bloch(const ModuleBase::Vector3<double>& r,
                            const UnitCell& ucell,
                            std::complex<double>* phi_bloch,
                            const ModuleBase::Vector3<double>& kvec_d) const;

    /// @brief Evaluate a Bloch-summed LCAO wavefunction at one point.
    /// Uses the same neighboring-cell AO sum as eval_phi_all_bloch().
    template <typename T>
    std::complex<double> eval_wfc_bloch(const ModuleBase::Vector3<double>& r_fix,
                                        int ik,
                                        int iband,
                                        const psi::Psi<T>& psi_ks,
                                        const UnitCell& ucell,
                                        const ModuleBase::Vector3<double>& kvec_d) const
    {
        psi_ks.fix_k(ik);
        int total_naos = 0;
        for (int iat = 0; iat < ucell.nat; ++iat)
        {
            total_naos += atypes_[ucell.iat2it[iat]].nw;
        }

        std::vector<std::complex<double>> phi_bloch(total_naos, std::complex<double>(0.0, 0.0));
        eval_phi_all_bloch(r_fix, ucell, phi_bloch.data(), kvec_d);

        std::complex<double> result(0.0, 0.0);
        for (int iw = 0; iw < total_naos; ++iw)
        {
            result += std::complex<double>(psi_ks(iband, iw)) * phi_bloch[iw];
        }
        return result;
    }
};

namespace LR_Util
{
template <typename T>
class ExcitonPlotter
{
    /// @brief Construct an exciton density plotter for TDA excitons.
    /// Supports average density (integrating out one particle coordinate) and
    /// conditional density (fixing the hole position and evaluating the coherent
    /// electron wavefunction).
    /// @param nspin_global global number of spin channels
    /// @param naos number of atomic-orbital basis functions
    /// @param nocc number of occupied bands per spin
    /// @param nvirt number of virtual bands per spin
    /// @param psi_ks_in Kohn-Sham wavefunction coefficients
    /// @param ucell_in unit cell
    /// @param kv_in k-point list
    /// @param gd_in grid driver for neighbor-list search
    /// @param orb_cutoff_in orbital cutoff radii per atom type
    /// @param Pgrid_in parallel grid for real-space decomposition
    /// @param rho_basis_in plane-wave basis for the real-space grid
    /// @param pX_in 2D block-cyclic distribution for X matrix
    /// @param pc_in 2D block-cyclic distribution for KS orbitals
    /// @param pmat_in parallel distribution for AO matrices
    /// @param eig excitation energies in Ry
    /// @param X BSE eigenvector amplitudes (TDA or full)
    /// @param openshell true for spin-unrestricted (nspin_x=2)
    /// @param orb optional LCAO_Orbitals pointer (required for conditional density)
public:
    ExcitonPlotter(const int nspin_global, const int naos, const std::vector<int>& nocc, const std::vector<int>& nvirt,
                   const psi::Psi<T>& psi_ks_in,
                   const UnitCell& ucell_in,
                   const K_Vectors& kv_in,
                   const Grid_Driver& gd_in, const std::vector<double>& orb_cutoff_in, const Parallel_Grid& Pgrid_in,
                   const ModulePW::PW_Basis& rho_basis_in,
                   const std::vector<Parallel_2D>& pX_in, const Parallel_2D& pc_in, const Parallel_Orbitals& pmat_in,
                   const double* eig, const T* X,
                   const bool openshell,
                   const LCAO_Orbitals* orb = nullptr)
    : nspin_x(openshell ? 2 : 1), naos(naos), nocc(nocc), nvirt(nvirt), ucell(ucell_in), kv(kv_in),
      gd_(gd_in), orb_cutoff_(orb_cutoff_in), Pgrid(Pgrid_in), rho_basis(rho_basis_in),
      pX(pX_in), pc(pc_in), pmat(pmat_in), eig(eig), X(X),
      nk(nspin_global == 2 ? kv_in.get_nks() / 2 : kv_in.get_nks()),
      ldim(nk* (nspin_x == 2 ? pX_in[0].get_local_size() + pX_in[1].get_local_size() : pX_in[0].get_local_size())),
      gdim(nk* std::inner_product(nocc.begin(), nocc.end(), nvirt.begin(), 0)),
      orb_(orb)
    {
        for (int is = 0;is < nspin_global;++is) { psi_ks_vec.emplace_back(LR_Util::get_psi_spin(psi_ks_in, is, nk)); }
        if (orb) { orb_eval_.init(*orb, ucell_in); }
    };
    /// @brief Set the de-excitation (Y) amplitude pointer for full BSE
    /// @param Y_in pointer to Y amplitudes
    void set_Y(T* Y_in) { this->Y = Y_in; };
    /// @brief Enable full BSE mode (X+Y instead of TDA X-only)
    /// @param tag true for full BSE, false for TDA
    void set_full(bool tag) { this->is_full = tag; };

    /// @brief Compute the transition density matrix D_trans = C_virt * X * C_occ^H
    /// for a given BSE state, in the AO basis at each k-point. Converts DMK to DMR.
    /// @param istate BSE state index
    /// @return DensityMatrix with DMR populated
    elecstate::DensityMatrix<T, T> cal_transition_density_matrix(const int istate);

    /// @brief [DEPRECATED] Plot exciton using transition density (physically incorrect).
    /// Computes |rho_trans(r)|^2 which is neither a proper exciton density nor a
    /// transition density. Use plot_average_density() or plot_cond_slice() instead.
    /// @param istate BSE state index
    /// @param type spin type label (e.g., "singlet")
    void plot_exciton(const int istate, const std::string& type);

    // --- Correct exciton density methods following ExciView formulas ---------
    // These replace the incorrect plot_exciton() which squared a transition
    // density, producing a physically meaningless quantity. The average density
    // integrates out one particle coordinate; the conditional density fixes
    // the hole position and evaluates the coherent electron wavefunction.
    //
    // Both use Hermitian effective DMK → DMR → cal_gint_rho (average) or
    // direct coherent real-space summation (conditional) to produce proper
    // non-negative densities.

    /// @brief Compute effective DMK for average hole density
    /// D_hole(k) = C_occ(k) * (X_k^H * X_k) * C_occ(k)^H.
    /// Marginalizes over conduction bands: M_k[v,v'] = Sum_c A_{kvc} * conj(A_{kv'c}).
    /// @param istate BSE state index
    /// @return {dmk_per_kpoint, total_weight} where total_weight = Sum_{k,v,c} |A_{kvc}|^2
    std::pair<std::vector<container::Tensor>, double> cal_effective_dmk_hole(const int istate);

    /// @brief Compute effective DMK for average electron density
    /// D_elec(k) = C_virt(k) * (X_k * X_k^H) * C_virt(k)^H.
    /// Marginalizes over valence bands: N_k[c,c'] = Sum_v A_{kvc} * conj(A_{kvc'}).
    /// @param istate BSE state index
    /// @return {dmk_per_kpoint, total_weight} where total_weight = Sum_{k,v,c} |A_{kvc}|^2
    std::pair<std::vector<container::Tensor>, double> cal_effective_dmk_elec(const int istate);

    /// @brief Plot average hole or electron density as a .cube file
    /// Integrates out the other particle coordinate. The effective DMK is Hermitian,
    /// so DMR's real part suffices for cal_gint_rho — no squaring needed.
    /// @param istate BSE state index
    /// @param type "hole" for average hole density, "elec" for average electron density
    void plot_average_density(const int istate, const std::string& type);

    /// @brief Plot average hole or electron density on a 2D cross-section
    /// Evaluates the Hermitian effective density matrix directly on the requested
    /// plane and writes a data file readable by plot_cond_slice.py.
    /// @param istate BSE state index
    /// @param type "hole" for average hole density, "elec" for average electron density
    /// @param plane cross-section plane: "ab", "bc", or "ca"
    /// @param slice_pos offset along the perpendicular direction (Bohr)
    /// @param npoints desired grid resolution
    /// @param scale view range relative to the plotted periodic cell
    void plot_average_slice(const int istate, const std::string& type,
                            const std::string& plane, double slice_pos, int npoints, double scale);

    /// @brief Plot conditional electron or hole density as home-cell and BvK .cube files
    /// Fixes the opposite particle at r_fix and evaluates the coherent conditional
    /// wavefunction. Uses direct real-space evaluation (NOT the DMR pipeline) to
    /// preserve k-point coherence.
    /// @param istate BSE state index
    /// @param r_fix fixed opposite-particle position {x, y, z} in Bohr
    /// @param type "elec" or "hole"
    void plot_conditional_density(const int istate, const std::array<double, 3>& r_fix,
                                  const std::string& type = "elec");

    /// @brief Compute conditional electron or hole density on a 2D BvK cross-section
    /// Evaluates psi_cond(r) on a regular grid in the chosen plane (ab, bc, or ca),
    /// spanning scale × BvK supercell extent centered on the home cell. Uses full
    /// Bloch-summed orbital evaluation with home-cell caching for efficiency.
    /// Writes a data file with grid + metadata readable by plot_cond_slice.py.
    /// @param istate BSE state index
    /// @param r_fix fixed opposite-particle position {x, y, z} in Bohr
    /// @param plane cross-section plane: "ab", "bc", or "ca"
    /// @param slice_pos offset along the perpendicular direction (Bohr)
    /// @param npoints desired grid resolution (pts per cell ≈ npoints / total_cells)
    /// @param scale view range relative to BvK supercell (default 1.3)
    /// @param type "elec" or "hole"
    void plot_cond_slice(const int istate, const std::array<double, 3>& r_fix,
                         const std::string& plane, double slice_pos, int npoints, double scale,
                         const std::string& type = "elec");
private:
    const int nspin_x = 1;      ///< 1 for singlet/triplet, 2 for up/down (open-shell)
    const int naos;              ///< number of atomic-orbital basis functions
    const std::vector<int>& nocc; ///< number of occupied bands per spin
    const std::vector<int>& nvirt;///< number of virtual bands per spin
    const int nk;                ///< number of k-points
    const int ldim;              ///< local leading dimension of X (data size per state)
    const int gdim;              ///< global leading dimension of X
    const double* eig;           ///< excitation energies (Ry)
    const T* X;                  ///< BSE eigenvector amplitudes
    T* Y = nullptr;              ///< de-excitation amplitudes (full BSE only)
    bool is_full = false;        ///< true if using full BSE (X+Y), false for TDA
    const K_Vectors& kv;         ///< k-point list
    std::vector<psi::Psi<T>> psi_ks_vec; ///< KS wavefunction per spin
    const UnitCell& ucell;       ///< unit cell
    const std::vector<double>& orb_cutoff_; ///< orbital cutoff radii
    const Grid_Driver& gd_;      ///< grid driver for neighbor search
    const Parallel_Grid& Pgrid;  ///< parallel real-space grid
    const ModulePW::PW_Basis& rho_basis; ///< plane-wave basis for density grid

    const std::vector<Parallel_2D>& pX; ///< 2D distribution for X matrix
    const Parallel_2D& pc;       ///< 2D distribution for KS orbitals
    const Parallel_Orbitals& pmat; ///< parallel distribution for AO matrices
    const LCAO_Orbitals* orb_ = nullptr; ///< orbital data for single-point evaluation
    OrbitalEvaluator orb_eval_;  ///< single-point NAO evaluator (initialized if orb_ != nullptr)

};


} // namespace LR_Util
