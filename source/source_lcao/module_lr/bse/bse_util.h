#pragma once

#include <ATen/core/tensor.h>
#include "source_psi/psi.h"
#include "source_base/tool_title.h"
#include "source_base/module_external/blas_connector.h"
#include "source_base/module_external/scalapack_connector.h"
#include "source_base/parallel_2d.h"
#include "source_basis/module_ao/parallel_orbitals.h"

namespace BSE_Util
{
using DAT = container::DataType;
using DEV = container::DeviceType;
template <typename T>
using DAT_ENUM = container::DataTypeToEnum<T>;

std::vector<std::complex<double>>
to_complex(const std::vector<double>& vin);

#ifdef __MPI
    /// @brief calculate the 2d-block transition density matrix in AO basis
    /// \f[ \tilde{\rho}_{\mu\mu}=c_{j,\mu}c^*_{b,\nu} \f]
    template<typename T>
    container::Tensor cal_dm_trans_onebase_pblas(
        const psi::Psi<T>& c,
        const Parallel_2D& pc,
        const int& ik,
        const int& naos,
        const int& imo1, // imo1 → imo2, for excitation imo1[1,nocc], imo2[nocc+1,nocc+nvirt]
        const int& imo2,
        const Parallel_Orbitals& pmat,
        const T& factor = (T)1.0);
#endif
    
    /// @brief calculate the transition density matrix in AO basis
    template<typename T>
    container::Tensor cal_dm_trans_onebase_blas(
        const psi::Psi<T>& c,
        const int& ik,
        const int& naos,
        const int& imo1, // imo1 → imo2, for excitation imo1[1,nocc], imo2[nocc+1,nocc+nvirt]
        const int& imo2,
        const T& factor = (T)1.0);

}
