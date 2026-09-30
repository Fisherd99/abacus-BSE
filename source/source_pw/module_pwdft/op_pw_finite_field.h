#ifndef OP_PW_FINITE_FIELD_H
#define OP_PW_FINITE_FIELD_H

#include "op_pw.h"

#include <complex>
#include <functional>
#include <vector>

namespace hamilt
{

class FiniteFieldOperatorPW : public OperatorPW<std::complex<double>, base_device::DEVICE_CPU>
{
  public:
    using ProjectionReducer = std::function<void(std::complex<double>*, int)>;

    FiniteFieldOperatorPW(int kpoint_count, int occupied_bands, int state_stride);
    FiniteFieldOperatorPW(int kpoint_count,
                          int occupied_bands,
                          int state_stride,
                          const ProjectionReducer& projection_reducer);

    void set_kpoint_data(int ik,
                         const std::complex<double>& factor,
                         const std::vector<std::complex<double>>& occupied,
                         const std::vector<std::complex<double>>& dual_difference);

    void add_kpoint_data(int ik,
                         const std::complex<double>& factor,
                         const std::vector<std::complex<double>>& occupied,
                         const std::vector<std::complex<double>>& dual_difference);

    void clear_kpoint_data();

    bool has_kpoint_data(int ik) const;

    void act(int nbands,
             int nbasis,
             int npol,
             const std::complex<double>* wavefunctions,
             std::complex<double>* output,
             int active_basis_size = 0,
             bool is_first_node = false) const override;

  private:
    struct KPointTerm
    {
        std::complex<double> factor = std::complex<double>(0.0, 0.0);
        std::vector<std::complex<double>> occupied;
        std::vector<std::complex<double>> dual_difference;
    };

    int occupied_bands = 0;
    int state_stride = 0;
    ProjectionReducer projection_reducer;
    std::vector<std::vector<KPointTerm>> kpoint_data;
};

} // namespace hamilt

#endif
