#pragma once
#define HAVE_SKEWSYMMETRIC // for elpa_skew_eigenvectors

#include <elpa/elpa.h>
#include <omp.h>
#include "source_base/parallel_2d.h"
#include "source_base/parallel_reduce.h"
#include "source_base/module_external/blacs_connector.h"
#include "source_base/module_external/scalapack_connector.h"
#include "source_lcao/module_lr/utils/lr_util.h"

namespace BSE
{
template <typename T>
void printM(const std::vector<T>& A, int m, int n, std::string file, std::string name);

/// @brief result={{A,B},{-A*,-B*}}
void arrayFlatten1(int nA /*A_part dim*/,
                   const std::vector<std::complex<double>>& A,
                   const std::vector<std::complex<double>>& B,
                   std::vector<std::complex<double>>& result);

/// @brief result={{Re(A+B), Im(A-B)},{-Im(A+B), Re(A-B)}}
void arrayFlatten2(int nA /*A_part dim*/,
                   const std::vector<std::complex<double>>& A,
                   const std::vector<std::complex<double>>& B,
                   std::vector<double>& result);

/// @brief solve full BSE through matrix M = {{Re(A+B), Im(A-B)},{-Im(A+B), Re(A-B)}}
void solve_full(const int& my_rank,
                const std::vector<std::complex<double>>& A_part,
                const std::vector<std::complex<double>>& B_part,
                const int& nA /*part_dim*/,
                std::vector<double>& ev,
                std::vector<std::complex<double>>& global_v);
/// @brief template solve TDA BSE, implemented in hamilt_bse_solver.hpp
template <typename T>
void solve_tda(const int& my_rank,
                const std::vector<T>& A_part,
                const Parallel_2D& pA,
                const int& nA /*part_dim*/,
                std::vector<double>& ev,
                std::vector<T>& global_v);
}

#include "hamilt_bse_solver.hpp"