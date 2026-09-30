#ifdef __LCAO
#include "fr_overlap.h"
#include "source_basis/module_ao/orb_read.h"
#include "source_base/timer.h"
#include "source_base/tool_title.h"
#include "source_base/math_integral.h"
#include "source_base/module_external/blas_connector.h"
#include "source_hamilt/module_finite_field/finite_field_matrix.h"

#include <type_traits>

namespace
{
template <typename T>
T generalized_overlap_cast(const std::complex<double>& value);

template <>
double generalized_overlap_cast<double>(const std::complex<double>& value)
{
    return value.real();
}

template <>
std::complex<double> generalized_overlap_cast<std::complex<double>>(
    const std::complex<double>& value)
{
    return value;
}

template <typename T>
T phase_translation_derivative(const T&, const double)
{
    return T(0);
}

template <>
std::complex<double> phase_translation_derivative(
    const std::complex<double>& value, const double momentum)
{
    return -ModuleBase::IMAG_UNIT * momentum * value;
}
}

template <typename T>
FiniteFieldFROverlap<T>::FiniteFieldFROverlap()
{

}

template <typename T>
void FiniteFieldFROverlap<T>::set_parameters(fr_ptr fr_in,
                                   const UnitCell* ucell_in,
                                   const LCAO_Orbitals* ptr_orb,
                                   const Grid_Driver* GridD_in,
                                   const Parallel_Orbitals* paraV,
                                   int radial_grid_num,
                                   int degree,
                                   bool calculate_center_gradients,
                                   bool analytic_center_gradients,
                                   ModuleBase::Vector3<double> momentum_transfer)
{
    this->fr = fr_in;
    this->ucell = ucell_in;
    this->ptr_orb_ = ptr_orb;
    this->FR_container = new hamilt::HContainer<T>(paraV);
    this->radial_grid_num = radial_grid_num;
    this->calculate_center_gradients = calculate_center_gradients;
    this->analytic_center_gradients = analytic_center_gradients;
    this->momentum_transfer = momentum_transfer;
    this->Leb_grid = new ModuleBase::Lebedev_laikov_grid(degree);
    this->Leb_grid->generate_grid_points();
    this->initialize_FR(GridD_in, paraV);
}

template <typename T>
void FiniteFieldFROverlap<T>::set_first_order_parameters(
    const UnitCell* ucell_in, const LCAO_Orbitals* ptr_orb,
    const Grid_Driver* GridD_in, const Parallel_Orbitals* paraV,
    const ModuleBase::Vector3<double> momentum_transfer,
    const bool calculate_center_gradients)
{
    this->ucell = ucell_in;
    this->ptr_orb_ = ptr_orb;
    this->calculate_center_gradients = calculate_center_gradients;
    this->analytic_center_gradients = true;
    this->momentum_transfer = momentum_transfer;
    this->first_order_integrator.reset(new FirstOrderOverlapIntegrator(
        *ptr_orb, momentum_transfer, calculate_center_gradients));
    this->FR_container = new hamilt::HContainer<T>(paraV);
    this->initialize_FR(GridD_in, paraV);
}

template <typename T>
FiniteFieldFROverlap<T>::FiniteFieldFROverlap(
    const FiniteFieldFROverlap<T>& FR_in)
{
    this->fr = FR_in.fr;
    this->ucell = FR_in.ucell;
    this->ptr_orb_ = FR_in.ptr_orb_;
    this->FR_container = new hamilt::HContainer<T>(*(FR_in.FR_container));
    this->radial_grid_num = FR_in.radial_grid_num;
    this->calculate_center_gradients = false;
    this->analytic_center_gradients = FR_in.analytic_center_gradients;
    this->momentum_transfer = FR_in.momentum_transfer;
    if (FR_in.Leb_grid)
    {
        this->Leb_grid
            = new ModuleBase::Lebedev_laikov_grid(FR_in.Leb_grid->degree);
        this->Leb_grid->generate_grid_points();
    }
    if (FR_in.first_order_integrator)
    {
        this->first_order_integrator.reset(new FirstOrderOverlapIntegrator(
            *this->ptr_orb_, this->momentum_transfer, false));
    }
}

template <typename T>
FiniteFieldFROverlap<T>::FiniteFieldFROverlap(FiniteFieldFROverlap<T>&& FR_in)
{
    this->fr = std::move(FR_in.fr);
    this->ucell = FR_in.ucell;
    this->ptr_orb_ = FR_in.ptr_orb_;
    this->FR_container = FR_in.FR_container;
    FR_in.FR_container = nullptr;
    this->radial_grid_num = FR_in.radial_grid_num;
    this->calculate_center_gradients = FR_in.calculate_center_gradients;
    this->analytic_center_gradients = FR_in.analytic_center_gradients;
    this->momentum_transfer = FR_in.momentum_transfer;
    this->dFR_datom = std::move(FR_in.dFR_datom);
    this->active_gradient_atom = FR_in.active_gradient_atom;
    FR_in.active_gradient_atom = -1;
    this->first_order_integrator = std::move(FR_in.first_order_integrator);
    this->Leb_grid = FR_in.Leb_grid;
    FR_in.Leb_grid = nullptr;
}

template <typename T>
const hamilt::HContainer<T>* FiniteFieldFROverlap<T>::get_dFR_pointer(
    const int atom, const int component) const
{
    if (!this->calculate_center_gradients || atom != this->active_gradient_atom
        || component < 0 || component >= 3)
    {
        return nullptr;
    }
    return this->dFR_datom[component].get();
}

template <typename T>
FiniteFieldFROverlap<T>::~FiniteFieldFROverlap()
{
    if (this->Leb_grid)
    {
        delete this->Leb_grid;
    }

    if (this->FR_container)
    {
        delete this->FR_container;
    }
}

template <typename T>
void FiniteFieldFROverlap<T>::initialize_FR(
    const Grid_Driver* GridD, const Parallel_Orbitals* paraV)
{
    ModuleBase::TITLE("FR_overlap", "initialize_FR");
    ModuleBase::timer::start("FR_overlap", "initialize_FR");
    for (int iat1 = 0; iat1 < ucell->nat; iat1++)
    {
        auto tau1 = ucell->get_tau(iat1);
        int T1, I1;
        ucell->iat2iait(iat1, &I1, &T1);
        AdjacentAtomInfo adjs;
        GridD->Find_atom(*ucell, tau1, T1, I1, &adjs);
        for (int ad = 0; ad < adjs.adj_num + 1; ++ad)
        {
            const int T2 = adjs.ntype[ad];
            const int I2 = adjs.natom[ad];
            int iat2 = ucell->itia2iat(T2, I2);
            if (paraV->is_invalid_atom_pair(iat1, iat2))
            {
                continue;
            }
            const ModuleBase::Vector3<int>& R_index = adjs.box[ad];
            // choose the real adjacent atoms
            // Note: the distance of atoms should less than the cutoff radius,
            // When equal, the theoretical value of matrix element is zero,
            // but the calculated value is not zero due to the numerical error, which would lead to result changes.
            if (this->ucell->cal_dtau(iat1, iat2, R_index).norm() * this->ucell->lat0
                >= ptr_orb_->Phi[T1].getRcut() + ptr_orb_->Phi[T2].getRcut())
            {
                continue;
            }
            hamilt::AtomPair<T> tmp(iat1, iat2, R_index.x, R_index.y, R_index.z, paraV);
            FR_container->insert_pair(tmp);
        }
    }
    // allocate the memory of BaseMatrix in FR_container, and set the new values to zero
    FR_container->allocate(nullptr, true);
    ModuleBase::timer::end("FR_overlap", "initialize_FR");
}

template <typename T>
void FiniteFieldFROverlap<T>::initialize_center_gradients(const int atom)
{
    this->dFR_datom.clear();
    this->dFR_datom.reserve(3);
    const Parallel_Orbitals* paraV = this->FR_container->get_paraV();
    for (int component = 0; component < 3; ++component)
    {
        std::unique_ptr<hamilt::HContainer<T>> derivative(
            new hamilt::HContainer<T>(paraV));
        for (int iap = 0; iap < this->FR_container->size_atom_pairs(); ++iap)
        {
            const hamilt::AtomPair<T>& pair
                = this->FR_container->get_atom_pair(iap);
            if (pair.get_atom_i() == atom || pair.get_atom_j() == atom)
            {
                for (int iR = 0; iR < pair.get_R_size(); ++iR)
                {
                    const ModuleBase::Vector3<int> R = pair.get_R_index(iR);
                    hamilt::AtomPair<T> empty_pair(pair.get_atom_i(),
                                                   pair.get_atom_j(),
                                                   R.x, R.y, R.z, paraV);
                    derivative->insert_pair(empty_pair);
                }
            }
        }
        derivative->allocate(nullptr, true);
        this->dFR_datom.emplace_back(std::move(derivative));
    }
    this->active_gradient_atom = atom;
}

template <typename T>
void FiniteFieldFROverlap<T>::calculate_FR()
{
    ModuleBase::TITLE("FR_overlap", "calculate_FR");
    ModuleBase::timer::start("FR_overlap", "calculate_FR");

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int iap = 0; iap < this->FR_container->size_atom_pairs(); ++iap)
    {
        hamilt::AtomPair<T>& tmp = this->FR_container->get_atom_pair(iap);
        int iat1 = tmp.get_atom_i();
        int iat2 = tmp.get_atom_j();
        const Parallel_Orbitals* paraV = tmp.get_paraV();

        for (int iR = 0; iR < tmp.get_R_size(); ++iR)
        {
            const ModuleBase::Vector3<int> R_index = tmp.get_R_index(iR);
            ModuleBase::Vector3<int> R_vector(R_index);
            auto dtau = ucell->cal_dtau(iat1, iat2, R_vector) * ucell->lat0;
            T* data_pointer = tmp.get_pointer(iR);
            std::array<T*, 3> bra_derivative{{nullptr, nullptr, nullptr}};
            std::array<T*, 3> ket_derivative{{nullptr, nullptr, nullptr}};
            if (this->first_order_integrator)
            {
                this->cal_FR_IJR_two_center(iat1, iat2, paraV, dtau,
                                            data_pointer, bra_derivative,
                                            ket_derivative);
            }
            else
            {
                this->cal_FR_IJR(iat1, iat2, paraV, dtau, data_pointer,
                                 bra_derivative, ket_derivative);
            }
        }
    }

    ModuleBase::timer::end("FR_overlap", "calculate_FR");
}

template <typename T>
void FiniteFieldFROverlap<T>::calculate_center_gradient(const int atom)
{
    if (!this->calculate_center_gradients || atom < 0 || atom >= this->ucell->nat)
    {
        return;
    }
    this->initialize_center_gradients(atom);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int iap = 0; iap < this->FR_container->size_atom_pairs(); ++iap)
    {
        hamilt::AtomPair<T>& pair = this->FR_container->get_atom_pair(iap);
        const int iat1 = pair.get_atom_i();
        const int iat2 = pair.get_atom_j();
        if (iat1 != atom && iat2 != atom) continue;
        const Parallel_Orbitals* paraV = pair.get_paraV();
        for (int iR = 0; iR < pair.get_R_size(); ++iR)
        {
            const ModuleBase::Vector3<int> R_index = pair.get_R_index(iR);
            const ModuleBase::Vector3<double> dtau
                = this->ucell->cal_dtau(iat1, iat2, R_index) * this->ucell->lat0;
            std::array<T*, 3> bra{{nullptr, nullptr, nullptr}};
            std::array<T*, 3> ket{{nullptr, nullptr, nullptr}};
            for (int component = 0; component < 3; ++component)
            {
                T* pointer = this->dFR_datom[component]
                                 ->find_matrix(iat1, iat2, R_index)->get_pointer();
                if (iat1 == atom) bra[component] = pointer;
                if (iat2 == atom) ket[component] = pointer;
            }
            if (this->first_order_integrator)
            {
                this->cal_FR_IJR_two_center(iat1, iat2, paraV, dtau,
                                            pair.get_pointer(iR), bra, ket);
            }
            else
            {
                this->cal_FR_IJR(iat1, iat2, paraV, dtau, pair.get_pointer(iR),
                                 bra, ket);
            }
        }
    }
}

template <typename T>
void FiniteFieldFROverlap<T>::cal_FR_IJR_two_center(
    const int& iat1, const int& iat2, const Parallel_Orbitals* paraV,
    const ModuleBase::Vector3<double>& dtau, T* data_pointer,
    const std::array<T*, 3>& bra_derivative,
    const std::array<T*, 3>& ket_derivative)
{
    int I1, T1, I2, T2;
    this->ucell->iat2iait(iat1, &I1, &T1);
    this->ucell->iat2iait(iat2, &I2, &T2);
    const Atom& atom1 = this->ucell->atoms[T1];
    const Atom& atom2 = this->ucell->atoms[T2];
    const int npol = this->ucell->get_npol();
    const std::vector<int> row_indexes = paraV->get_indexes_row(iat1);
    const std::vector<int> col_indexes = paraV->get_indexes_col(iat2);
    const ModuleBase::Vector3<double> center_bra
        = this->ucell->get_tau(iat1) * this->ucell->lat0;
    const ModuleBase::Vector3<double> center_ket = center_bra + dtau;

    int irow = -1;
    for (std::size_t iw1l = 0; iw1l < row_indexes.size(); iw1l += npol)
    {
        ++irow;
        const int iw1 = row_indexes[iw1l] / npol;
        const int l1 = atom1.iw2l[iw1];
        const int n1 = atom1.iw2n[iw1];
        const int encoded_m1 = atom1.iw2m[iw1];
        const int m1 = encoded_m1 % 2 == 0 ? -encoded_m1 / 2
                                            : (encoded_m1 + 1) / 2;
        int icol = -1;
        for (std::size_t iw2l = 0; iw2l < col_indexes.size(); iw2l += npol)
        {
            ++icol;
            const int iw2 = col_indexes[iw2l] / npol;
            const int l2 = atom2.iw2l[iw2];
            const int n2 = atom2.iw2n[iw2];
            const int encoded_m2 = atom2.iw2m[iw2];
            const int m2 = encoded_m2 % 2 == 0 ? -encoded_m2 / 2
                                                : (encoded_m2 + 1) / 2;
            std::complex<double> value;
            std::complex<double> grad_bra[3];
            std::complex<double> grad_ket[3];
            this->first_order_integrator->calculate(
                T1, l1, n1, encoded_m1, T2, l2, n2, encoded_m2,
                center_bra, center_ket, &value,
                this->calculate_center_gradients ? grad_bra : nullptr,
                this->calculate_center_gradients ? grad_ket : nullptr);
            for (int ipol = 0; ipol < npol; ++ipol)
            {
                const int index = (npol * irow + ipol)
                                      * static_cast<int>(col_indexes.size())
                                  + npol * icol + ipol;
                data_pointer[index] = generalized_overlap_cast<T>(value);
                if (this->calculate_center_gradients)
                {
                    for (int alpha = 0; alpha < 3; ++alpha)
                    {
                        if (bra_derivative[alpha])
                        {
                            bra_derivative[alpha][index]
                                += generalized_overlap_cast<T>(grad_bra[alpha]);
                        }
                        if (ket_derivative[alpha])
                        {
                            ket_derivative[alpha][index]
                                += generalized_overlap_cast<T>(grad_ket[alpha]);
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void FiniteFieldFROverlap<T>::cal_FR_IJR(const int& iat1, const int& iat2, const Parallel_Orbitals* paraV, const ModuleBase::Vector3<double>& dtau, T* data_pointer, const std::array<T*, 3>& bra_derivative, const std::array<T*, 3>& ket_derivative)
{
    // ---------------------------------------------
    // get info of orbitals of atom1 and atom2 from ucell
    // ---------------------------------------------
    int T1, I1;
    this->ucell->iat2iait(iat1, &I1, &T1);
    int T2, I2;
    this->ucell->iat2iait(iat2, &I2, &T2);
    Atom& atom1 = this->ucell->atoms[T1];
    Atom& atom2 = this->ucell->atoms[T2];

    ModuleBase::Vector3<double> tau_1 = this->ucell->get_tau(iat1) * this->ucell->lat0;

    double Rcut1 = ptr_orb_->Phi[T1].getRcut();
    double Rcut2 = ptr_orb_->Phi[T2].getRcut();

    // npol is the number of polarizations,
    // 1 for non-magnetic (one Hamiltonian matrix only has spin-up or spin-down),
    // 2 for magnetic (one Hamiltonian matrix has both spin-up and spin-down)
    const int npol = this->ucell->get_npol();

    const int* iw2l1 = atom1.iw2l.data();
    const int* iw2n1 = atom1.iw2n.data();
    const int* iw2m1 = atom1.iw2m.data();
    const int* iw2l2 = atom2.iw2l.data();
    const int* iw2n2 = atom2.iw2n.data();
    const int* iw2m2 = atom2.iw2m.data();

    const int maxL1 = atom1.nwl;
    const int maxL2 = atom2.nwl;

    // ---------------------------------------------
    // calculate the overlap matrix for each pair of orbitals
    // ---------------------------------------------
    auto row_indexes = paraV->get_indexes_row(iat1);
    auto col_indexes = paraV->get_indexes_col(iat2);

    std::set<std::pair<int, int>> LN_pair1;
    std::set<std::pair<int, int>> LN_pair2;
    for (int iw1l = 0; iw1l < row_indexes.size(); iw1l += npol)
    {
        const int iw1 = row_indexes[iw1l] / npol;
        const int L1 = iw2l1[iw1];
        const int N1 = iw2n1[iw1];

        LN_pair1.insert(std::make_pair(L1, N1));
    }

    for (int iw2l = 0; iw2l < col_indexes.size(); iw2l += npol)
    {
        const int iw2 = col_indexes[iw2l] / npol;
        const int L2 = iw2l2[iw2];
        const int N2 = iw2n2[iw2];

        LN_pair2.insert(std::make_pair(L2, N2));
    }

    int angular_grid_num = Leb_grid->degree;
    int grid_num = radial_grid_num * angular_grid_num;
    int row_num = static_cast<int>(row_indexes.size() / npol);
    int col_num = static_cast<int>(col_indexes.size() / npol);

    T *grid_1 = new T[row_num*grid_num]; // matrix [row_num, grid_num]
    T *grid_2 = new T[grid_num*col_num]; // matrix [grid_num, col_num]
    std::array<T*, 3> grid_2_derivative{{nullptr, nullptr, nullptr}};
    if (this->calculate_center_gradients)
    {
        for (int component = 0; component < 3; ++component)
        {
            grid_2_derivative[component] = new T[grid_num * col_num];
            ModuleBase::GlobalFunc::ZEROS(grid_2_derivative[component], grid_num * col_num);
        }
    }
    ModuleBase::GlobalFunc::ZEROS(grid_1, row_num*grid_num);
    ModuleBase::GlobalFunc::ZEROS(grid_2, grid_num*col_num);

    double xmin = 0.0;
    double xmax = Rcut1;
    double *r_radial = new double[radial_grid_num];
    double *weights_radial = new double[radial_grid_num];
    ModuleBase::Integral::Gauss_Legendre_grid_and_weight(xmin, xmax, radial_grid_num, r_radial, weights_radial);

    int count = -1;
    for (int ir = 0; ir < radial_grid_num; ir++)
    {
        std::map<std::pair<int, int>, double> psi_value1 = psi_inter(T1, LN_pair1, r_radial[ir]);

        for (int ian = 0; ian < angular_grid_num; ian++)
        {
            count++;

            ModuleBase::Vector3<double> r_angular_tmp = Leb_grid->get_grid_coor()[ian];
            ModuleBase::Vector3<double> r_coor = r_radial[ir] * r_angular_tmp;
            ModuleBase::Vector3<double> tmp_r_coor = r_coor - dtau;
            double tmp_r_coor_norm = tmp_r_coor.norm();
            ModuleBase::Vector3<double> tmp_r_unit;
            if (tmp_r_coor_norm > 1e-10)
            {
                tmp_r_unit = tmp_r_coor / tmp_r_coor_norm;
            }

            if (tmp_r_coor_norm > Rcut2) { continue;
}

            std::map<std::pair<int, int>, double> psi_value2 = psi_inter(T2, LN_pair2, tmp_r_coor_norm);

            std::vector<double> rly1;
            ModuleBase::Ylm::rl_sph_harm (maxL1, r_angular_tmp.x, r_angular_tmp.y, r_angular_tmp.z, rly1);

            std::vector<double> rly2;
            ModuleBase::Ylm::rl_sph_harm (maxL2, tmp_r_unit.x, tmp_r_unit.y, tmp_r_unit.z, rly2);
            std::vector<double> grad_rly2;
            std::array<std::map<std::pair<int, int>, double>, 6> shifted_psi2;
            std::array<std::vector<double>, 6> shifted_rly2;
            constexpr double derivative_step = 1.0e-4;
            if (this->calculate_center_gradients && this->analytic_center_gradients)
            {
                grad_rly2.resize(rly2.size() * 3);
                ModuleBase::Ylm::grad_rl_sph_harm(
                    maxL2, tmp_r_unit.x, tmp_r_unit.y, tmp_r_unit.z,
                    rly2.data(), grad_rly2.data());
            }
            else if (this->calculate_center_gradients)
            {
                for (int component = 0; component < 3; ++component)
                {
                    for (int sign_index = 0; sign_index < 2; ++sign_index)
                    {
                        ModuleBase::Vector3<double> shifted = tmp_r_coor;
                        shifted[component] += sign_index == 0
                                                  ? -derivative_step
                                                  : derivative_step;
                        const double radius = shifted.norm();
                        ModuleBase::Vector3<double> unit;
                        if (radius > 1.0e-10) unit = shifted / radius;
                        const int slot = 2 * component + sign_index;
                        shifted_psi2[slot] = psi_inter(T2, LN_pair2, radius);
                        ModuleBase::Ylm::rl_sph_harm(
                            maxL2, unit.x, unit.y, unit.z, shifted_rly2[slot]);
                    }
                }
            }

            double weights_angular = Leb_grid->get_weight()[ian];

            int irow1 = -1;
            for (int iw1l = 0; iw1l < row_indexes.size(); iw1l += npol)
            {
                irow1++;
                const int iw1 = row_indexes[iw1l] / npol;
                const int L1 = iw2l1[iw1];
                const int N1 = iw2n1[iw1];
                const int m1 = iw2m1[iw1];

                grid_1[irow1*grid_num+count] = psi_value1[std::make_pair(L1, N1)] * rly1[L1*L1+m1] * r_radial[ir] * r_radial[ir] * weights_radial[ir];

                int icol2 = -1;
                for (int iw2l = 0; iw2l < col_indexes.size(); iw2l += npol)
                {
                    icol2++;
                    const int iw2 = col_indexes[iw2l] / npol;
                    const int L2 = iw2l2[iw2];
                    const int N2 = iw2n2[iw2];
                    const int m2 = iw2m2[iw2];

                    grid_2[count*col_num+icol2] = psi_value2[std::make_pair(L2, N2)] * rly2[L2*L2+m2] * weights_angular * fr(r_coor+tau_1);
                    if (this->calculate_center_gradients)
                    {
                        const Numerical_Orbital_Lm& orbital
                            = ptr_orb_->Phi[T2].PhiLN(L2, N2);
                        for (int component = 0; component < 3; ++component)
                        {
                            const int lm = L2 * L2 + m2;
                            double orbital_gradient = 0.0;
                            if (this->analytic_center_gradients)
                            {
                                const double radial_derivative
                                    = Polynomial_Interpolation_Derivative(
                                        orbital.getPsi(), orbital.getNr(),
                                        orbital.getRab(0), tmp_r_coor_norm);
                                double angular_derivative = 0.0;
                                if (tmp_r_coor_norm > 1.0e-10)
                                {
                                    angular_derivative
                                        = (grad_rly2[3 * lm + component]
                                           - L2 * rly2[lm] * tmp_r_unit[component])
                                          / tmp_r_coor_norm;
                                }
                                orbital_gradient
                                    = radial_derivative * rly2[lm]
                                          * tmp_r_unit[component]
                                      + psi_value2[std::make_pair(L2, N2)]
                                          * angular_derivative;
                            }
                            else
                            {
                                const int minus_slot = 2 * component;
                                const int plus_slot = minus_slot + 1;
                                const double orbital_minus
                                    = shifted_psi2[minus_slot]
                                          [std::make_pair(L2, N2)]
                                      * shifted_rly2[minus_slot][lm];
                                const double orbital_plus
                                    = shifted_psi2[plus_slot]
                                          [std::make_pair(L2, N2)]
                                      * shifted_rly2[plus_slot][lm];
                                orbital_gradient
                                    = (orbital_plus - orbital_minus)
                                      / (2.0 * derivative_step);
                            }
                            // d phi(r-R2) / d R2 = -grad_{r-R2} phi.
                            grid_2_derivative[component][count * col_num + icol2]
                                = -orbital_gradient
                                  * weights_angular * fr(r_coor + tau_1);
                        }
                    }
                }

            }

        }
    }

    T *matrix_mul = new T[row_num*col_num]; // matrix [row_num, col_num]
    ModuleBase::GlobalFunc::ZEROS(matrix_mul, row_num*col_num);

    BlasConnector::gemm('N', 'N', row_num, col_num, grid_num,
        1, grid_1, grid_num, grid_2, col_num,
        0, matrix_mul, col_num);

    for(int ir = 0; ir < row_num; ir++)
    {
        for(int ic = 0; ic < col_num; ic++)
        {
            for (int ipol = 0; ipol < npol; ipol++)
            {
                int index = (npol*ir+ipol)*col_num*npol + npol*ic + ipol;
                data_pointer[index] = matrix_mul[ir*col_num+ic];
            }
        }
    }

    if (this->calculate_center_gradients)
    {
        std::vector<T> ket_matrix(row_num * col_num);
        for (int component = 0; component < 3; ++component)
        {
            BlasConnector::gemm('N', 'N', row_num, col_num, grid_num,
                1, grid_1, grid_num, grid_2_derivative[component], col_num,
                0, ket_matrix.data(), col_num);
            for (int ir = 0; ir < row_num; ++ir)
            {
                for (int ic = 0; ic < col_num; ++ic)
                {
                    for (int ipol = 0; ipol < npol; ++ipol)
                    {
                        const int index = (npol * ir + ipol) * col_num * npol
                                          + npol * ic + ipol;
                        const T ket = ket_matrix[ir * col_num + ic];
                        if (ket_derivative[component])
                        {
                            ket_derivative[component][index] += ket;
                        }
                        // Translating both centers differentiates only the
                        // plane-wave phase: d1 T + d2 T = -i q T.
                        if (bra_derivative[component])
                        {
                            bra_derivative[component][index]
                                += phase_translation_derivative(
                                       data_pointer[index],
                                       this->momentum_transfer[component])
                                   - ket;
                        }
                    }
                }
            }
            delete[] grid_2_derivative[component];
        }
    }

    delete[] r_radial;
    delete[] weights_radial;
    delete[] grid_1;
    delete[] grid_2;
    delete[] matrix_mul;
}

template <typename T>
std::map<std::pair<int, int>, double> FiniteFieldFROverlap<T>::psi_inter(const int &T1, const std::set<std::pair<int, int>> &LN_pair1, const double &r_norm)
{
    std::map<std::pair<int, int>, double> psi_value;

    for (auto i : LN_pair1)
    {
        int L1 = i.first;
        int N1 = i.second;

        const double *psi_r1 = ptr_orb_->Phi[T1].PhiLN(L1, N1).getPsi();
        int mesh_r1 = ptr_orb_->Phi[T1].PhiLN(L1, N1).getNr();
        double dr1 = ptr_orb_->Phi[T1].PhiLN(L1, N1).getRab(0);

        psi_value[i] = Polynomial_Interpolation(psi_r1, mesh_r1, dr1, r_norm);
    }

    return psi_value;
}

template <typename T>
double FiniteFieldFROverlap<T>::Polynomial_Interpolation(
    const double *psi_r,
    const int &mesh_r,
    const double &dr,
    const double &x
)
{
    return hamilt::finite_field_cubic_interpolate(psi_r, mesh_r, dr, x);
}

template <typename T>
double FiniteFieldFROverlap<T>::Polynomial_Interpolation_Derivative(
    const double* psi_r, const int& mesh_r, const double& dr, const double& x)
{
    return hamilt::finite_field_cubic_interpolate_derivative(
        psi_r, mesh_r, dr, x);
}

// T of FR_overlap can be double or std::complex<double>
template class FiniteFieldFROverlap<double>;
template class FiniteFieldFROverlap<std::complex<double>>;

#endif
