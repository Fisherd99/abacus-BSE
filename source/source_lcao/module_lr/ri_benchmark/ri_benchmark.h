#ifdef __EXX
#pragma once
#include "source_cell/unitcell.h"
#include "source_psi/psi.h"
#include "source_lcao/module_ri/RI_Util.h" // for get_Born_von_Karmen_cells
#include "source_basis/module_ao/parallel_orbitals.h"
#include <RI/global/Tensor.h>
namespace RI_Benchmark
{
    using TA = int;
    using TC = std::array<int, 3>;
    using TAC = std::pair<int, TC>;

    template <typename T>
    using TLRI = std::map<int, std::map<TAC, RI::Tensor<T>>>;
    template<typename T>
    using TLRIX = std::map<int, std::map<TAC, std::vector<T>>>;

    class RI_kRlist{
    public:
        std::unique_ptr<K_Vectors> klist;
        std::vector<TC> Rlist;

        RI_kRlist(const std::string& file, const UnitCell& ucell){
            this->klist = std::make_unique<K_Vectors>();
            read_kpts(file, ucell, this->klist);
            const TC period = RI_Util::get_Born_vonKarmen_period(*klist);
            this->Rlist = RI_Util::get_Born_von_Karmen_cells(period);
            std::cout << "Rlist:" << std::endl;
            for (const auto& iR: Rlist)
            {
                std::cout << "iR:" << iR[0] << " " << iR[1] << " " << iR[2] << std::endl;
            }
        };
        ~RI_kRlist(){};

        void read_kpts(const std::string& file, const UnitCell& ucell, std::unique_ptr<K_Vectors> & klist)
        {
            std::ifstream ifs;
            ifs.open(file);
            if (!ifs) throw std::runtime_error(file + "not found");
            std::string tmp;
            for (int i = 0;i < 7;++i) { std::getline(ifs, tmp); } // skip the first 7 lines(include 7th atom coord line)
            ifs >> klist->nmp[0] >> klist->nmp[1] >> klist->nmp[2];
            int nk = klist->nmp[0] * klist->nmp[1] * klist->nmp[2];
            klist->set_nks(nk);
            klist->kvec_c.resize(nk);
            klist->kvec_d.resize(nk);
            klist->wk.resize(nk);
            for (int ik = 0;ik < nk;++ik)
            {
                ifs >> klist->kvec_c[ik].x >> klist->kvec_c[ik].y >> klist->kvec_c[ik].z;
                klist->kvec_d[ik] = klist->kvec_c[ik] * ucell.latvec / ModuleBase::TWO_PI;
            }
            std::cout << "FISH_output: klist:" << std::endl;
            for (int ik = 0;ik < nk;++ik)
            {
                std::cout << "ik=" << ik <<": " << klist->kvec_c[ik].x << " " << klist->kvec_c[ik].y << " " << klist->kvec_c[ik].z 
                << " | " << klist->kvec_d[ik].x << " " << klist->kvec_d[ik].y << " " << klist->kvec_d[ik].z << std::endl;
            }
        } 
    };

    template <typename TK, typename TR>
    void benchmark_driver_A(std::string& file_Cs, std::string& file_Vs, std::string& file_kswfc, const int nocc, const int nvirt);
    template <typename TK, typename TR>
    void benchmark_driver_AX(std::string& file_Cs, std::string& file_Vs, std::string& file_kswfc, const int nocc, const int nvirt);

    // 1. full-matrix version

    /// calculate $Cs_{Ua\alpha,Vi}^{mo} = \sum_{\mu\nu} C_{U\mu\alpha,V\nu}^{ao} c^*_{\mu a} c_{\nu i}$
    /// if occ_first, calculate $Cs_{Ui\alpha,Va}^{mo} = \sum_{\mu\nu} C_{U\mu\alpha,V\nu}^{ao} c^*_{\mu i} c_{\nu a}$
    template <typename TK, typename TR>
    TLRI<TK> cal_Cs_mo(const UnitCell& ucell,
        const TLRI<TR>& Cs_ao,
        const psi::Psi<TK>& wfc_ks,
        const int& nocc,
        const int& nvirt,
        const int& occ_first=false,
        const bool& read_from_aims=false,
        const std::vector<int>& aims_nbasis={});

    /// A=CVC, sum over atom quads
    template <typename TK, typename TR>
    std::vector<TK> cal_Amat_full(const TLRI<TK>& Cs_a,
        const TLRI<TK>& Cs_b,
        const TLRI<TR>& Vs);

    // 2. AX version
    template <typename TK>
    TLRIX<TK> cal_CsX(const TLRI<TK>& Cs_mo, TK* X);

    template <typename TK, typename TR>
    TLRI<TK> cal_CV(const TLRI<TK>& Cs_a,
        const TLRI<TR>& Vs);

    /// AX=CV(CX), sum over atom quads
    template <typename TK, typename TR>
    void cal_AX(const TLRI<TK>& Cs_a,
        const TLRIX<TK>& Cs_bX,
        const TLRI<TR>& Vs,
        TK* AX,
        const double& scale = 2.0);
    /// AX=（CV)(CX), sum over atom quads
    template <typename TK>
    void cal_AX(const TLRI<TK>& CV,
        const TLRIX<TK>& Cs_bX,
        TK* AX,
        const double& scale = 2.0);

    // 3. read/write tools    
    template<typename FPTYPE>
    std::vector<FPTYPE> read_aims_ebands(const std::string& file, const int nocc, const int nvirt, int& ncore);

    /// read the number of bands from the file `band_out`
    inline void read_nbands_file(const std::string& file, int& nbands_file)
    {
        std::ifstream ifs;
        ifs.open(file);
        for (int i = 0;i < 3;++i) { ifs >> nbands_file; }
    }

    inline void read_one_data(std::ifstream& ifs, double& data){
		std::string temp;
		ifs >> data >> temp;
	}    
    inline void read_one_data(std::ifstream& ifs, std::complex<double>& data){
		double real, imag;
		ifs >> real >> imag;
		data = std::complex<double>(real, imag);
	}

    template <typename TK>
    void read_aims_eigenvectors(psi::Psi<TK>& wfc_ks, const std::string& file, const int ncore, const int nbands, const int nbasis);

    template <typename TK>
    void read_librpa_eigenvectors(psi::Psi<TK>& wfc_ks, const std::string& file, const int ncore, const int nbands_file, Parallel_Orbitals& pmat);

    /// only for blocking by atom pairs (abacus type)
    template <typename TCs, typename TR>
    TLRI<TR> read_coulomb_mat(const std::string& file, const TLRI<TCs>& Cs, const RI_kRlist& kRlist);
    /// for any way of blocking (aims type)
    template <typename TCs, typename TR>
    TLRI<TR> read_coulomb_mat_general(const std::string& file, const TLRI<TCs>& Cs, const RI_kRlist& kRlist);
    template <typename TR>
    bool compare_Vs(const TLRI<TR>& Vs1, const TLRI<TR>& Vs2, const double thr = 1e-4);
    template <typename TR>
    std::vector<TLRI<TR>> split_Ds(const std::vector<std::vector<TR>>& Ds, const std::vector<int>& aims_nbasis, const UnitCell& ucell);
}
#include "ri_benchmark.hpp"
#endif