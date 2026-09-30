#ifndef FINITE_FIELD_CONTROL_H
#define FINITE_FIELD_CONTROL_H

#include <array>
#include <complex>
#include <iosfwd>
#include <memory>
#include <string>

class UnitCell;

namespace ModulePW
{
class PW_Basis_K;
}

namespace psi
{
template <typename T, typename Device>
class Psi;
}

namespace base_device
{
struct DEVICE_CPU;
}

namespace hamilt
{

class FiniteFieldOperatorPW;

struct FiniteFieldPWConfig
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
};

/**
 * Owns the finite-field state that spans PW diagonalization cycles.
 *
 * The controller keeps Cartesian geometry, polarization branches, and the
 * frozen Hamiltonian snapshots together. MPI collectives remain in the PW
 * preparation and operator layers; phase files are touched only by rank zero.
 */
class FiniteFieldPWController
{
  public:
    using Wavefunctions
        = psi::Psi<std::complex<double>, base_device::DEVICE_CPU>;

    FiniteFieldPWController();
    ~FiniteFieldPWController();

    FiniteFieldPWController(const FiniteFieldPWController&) = delete;
    FiniteFieldPWController& operator=(const FiniteFieldPWController&) = delete;

    void configure(const UnitCell& ucell,
                   const ModulePW::PW_Basis_K& basis,
                   FiniteFieldOperatorPW& finite_field_operator,
                   const FiniteFieldPWConfig& config,
                   std::ostream& log);

    void prepare_cycle(const Wavefunctions& wavefunctions);

    double evaluate(const Wavefunctions& wavefunctions, std::ostream& log);

    int berry_cycles() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace hamilt

#endif
