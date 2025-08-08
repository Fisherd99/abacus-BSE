#pragma once
#include <ATen/core/tensor.h>
#include "source_psi/psi.h"
#include <vector>
#include <cassert>
#ifdef __MPI
#include "source_base/parallel_2d.h"
#endif
namespace LR
{
#ifndef MO_TYPE_H
#define MO_TYPE_H
    enum MO_TYPE { OO, VO, VV, ALL };
#endif
/*
MO_TYPE: OO   VO    VV    ALL
nmo1    nocc nocc  nvirt nocc+nvirt
nmo2    nocc nvirt nvirt nocc+nvirt
imo1    0    0     nocc  0
imo2    0    nocc  nocc  0
*/
    inline void set_dim(const MO_TYPE type, const int& nocc, const int& nvirt,
        int& nmo1, int& nmo2, int& imo1, int& imo2)
    {
        if (type == MO_TYPE::ALL)
        {
            nmo1 = nocc + nvirt;
            nmo2 = nocc + nvirt;
            imo1 = 0;
            imo2 = 0;
        }
        else{
            assert(type == MO_TYPE::OO || type == MO_TYPE::VO || type == MO_TYPE::VV);
            nmo1 = type == MO_TYPE::VV ? nvirt : nocc;
            nmo2 = type == MO_TYPE::OO ? nocc : nvirt;
            imo1 = type == MO_TYPE::VV ? nocc : 0;
            imo2 = type == MO_TYPE::OO ? 0 : nocc;
        }
    }
    template<typename T>
    void ao_to_mo_forloop_serial(
        const std::vector<container::Tensor>& mat_ao,
        const psi::Psi<T>& coeff,
        const int& nocc,
        const int& nvirt,
        T* const mat_mo,
        const MO_TYPE type = VO);
    template<typename T>
    void ao_to_mo_blas(
        const std::vector<container::Tensor>& mat_ao,
        const psi::Psi<T>& coeff,
        const int& nocc,
        const int& nvirt,
        T* const mat_mo,
        const bool add_on = true,
        const MO_TYPE type = VO);
#ifdef __MPI
    template<typename T>
    void ao_to_mo_pblas(
        const std::vector<container::Tensor>& mat_ao,
        const Parallel_2D& pmat_ao,
        const psi::Psi<T>& coeff,
        const Parallel_2D& pcoeff,
        const int& naos,
        const int& nocc,
        const int& nvirt,
        const Parallel_2D& pmat_mo,
        T* const mat_mo,
        const bool add_on = true,
        const MO_TYPE type = VO);
#endif
}