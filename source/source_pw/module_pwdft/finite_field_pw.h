#ifndef FINITE_FIELD_PW_H
#define FINITE_FIELD_PW_H

#include "source_cell/kpoint_strings.h"

#include <array>
#include <complex>
#include <functional>
#include <vector>

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

using FiniteFieldGLabel = std::array<int, 3>;
using FiniteFieldReducer = std::function<void(std::complex<double>*, int)>;

struct FiniteFieldPWState
{
    std::vector<FiniteFieldGLabel> labels;
    std::vector<std::complex<double>> occupied;
};

struct FiniteFieldPreparedKPoint
{
    std::vector<std::complex<double>> occupied;
    std::vector<std::complex<double>> dual_difference;
    std::complex<double> forward_overlap_determinant
        = std::complex<double>(0.0, 0.0);
};

void prepare_finite_field_pw_data(
    const ModuleCell::KPointStrings& strings,
    const std::vector<FiniteFieldPWState>& states,
    int occupied_bands,
    int state_stride,
    const std::array<int, 3>& reciprocal_grid,
    const FiniteFieldReducer& reducer,
    std::vector<FiniteFieldPreparedKPoint>& prepared);

void prepare_finite_field_pw_data_from_abacus(
    const ModuleCell::KPointStrings& strings,
    const ModulePW::PW_Basis_K& basis,
    const psi::Psi<std::complex<double>, base_device::DEVICE_CPU>& wavefunctions,
    int occupied_bands,
    std::vector<FiniteFieldPreparedKPoint>& prepared);

} // namespace hamilt

#endif
