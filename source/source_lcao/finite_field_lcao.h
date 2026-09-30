#ifndef FINITE_FIELD_LCAO_H
#define FINITE_FIELD_LCAO_H

#include <array>
#include <complex>
#include <iosfwd>
#include <memory>
#include <string>

class Grid_Driver;
class K_Vectors;
class LCAO_Orbitals;
class Parallel_Orbitals;
class RadialCollection;
class UnitCell;
namespace ModuleBase { class matrix; }

namespace psi
{
template <typename T, typename Device> class Psi;
}
namespace base_device { struct DEVICE_CPU; }

namespace hamilt
{
template <typename TK, typename TR> class HamiltLCAO;

struct FiniteFieldLCAOConfig
{
    int cartesian_axis = 0;
    double amplitude = 0.0;
    int berry_cycles = 1;
    std::array<int, 3> mesh{{0, 0, 0}};
    int occupied_bands = 0;
    double cell_volume_bohr3 = 0.0;
    std::string branch_io;
    std::string read_directory;
    std::string output_directory;
    bool root_rank = false;
    bool calculate_forces = false;
    std::string overlap_backend = "rayleigh_expansion";
    int rayleigh_lmax = 6;
};

struct FiniteFieldLCAODirectionRole
{
    bool report_polarization = true;
    bool couple_field = false;
    bool calculate_force_derivatives = false;
};

inline FiniteFieldLCAODirectionRole finite_field_lcao_direction_role(
    const double cartesian_projection, const bool calculate_forces)
{
    FiniteFieldLCAODirectionRole role;
    role.couple_field = cartesian_projection != 0.0;
    role.calculate_force_derivatives = role.couple_field && calculate_forces;
    return role;
}

/** Finite-field controller for insulating complex LCAO runs.
 * Dense Berry matrices are replicated across the diagonalization communicator,
 * while wavefunctions, overlap matrices, and the injected Hamiltonian retain
 * the normal two-dimensional block-cyclic distribution. */
class FiniteFieldLCAOController
{
  public:
    using Wavefunctions
        = psi::Psi<std::complex<double>, base_device::DEVICE_CPU>;

    FiniteFieldLCAOController();
    ~FiniteFieldLCAOController();
    FiniteFieldLCAOController(const FiniteFieldLCAOController&) = delete;
    FiniteFieldLCAOController& operator=(const FiniteFieldLCAOController&) = delete;

    void configure(const UnitCell& ucell,
                   const Grid_Driver& grid,
                   const K_Vectors& kpoints,
                   const LCAO_Orbitals& orbitals,
                   const RadialCollection& radial_orbitals,
                   const Parallel_Orbitals& parallel_orbitals,
                   HamiltLCAO<std::complex<double>, double>& hamiltonian,
                   const FiniteFieldLCAOConfig& config,
                   std::ostream& log);
    void prepare_cycle(const Wavefunctions& wavefunctions);
    double evaluate(const Wavefunctions& wavefunctions, std::ostream& log);
    void add_force(const Wavefunctions& wavefunctions,
                   ModuleBase::matrix& force,
                   std::ostream& log);
    int berry_cycles() const;
    bool needs_bootstrap() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace hamilt
#endif
