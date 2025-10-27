#include "bse_io.h"
#include "bse_util.h"
#include <dirent.h>
namespace BSE_IO{

RI_kRlist::RI_kRlist(const std::string& file, const UnitCell& ucell, K_Vectors* const pkv)
: klist(pkv)
{
    read_kpts(file, ucell, this->klist);
    const TC period = RI_Util::get_Born_vonKarmen_period(*klist);
    this->Rlist = RI_Util::get_Born_von_Karmen_cells(period);
    std::cout << "Rlist:" << std::endl;
    for (const auto& iR: Rlist)
    {
        std::cout << "iR:" << iR[0] << " " << iR[1] << " " << iR[2] << std::endl;
    }
};

void RI_kRlist::read_kpts(const std::string& file, const UnitCell& ucell, K_Vectors* const klist)
{
    std::ifstream ifs;
    ifs.open(file);
    if (!ifs) throw std::runtime_error(file + " not found");
    std::string tmp;
    for (int i = 0; i < 7; ++i) { std::getline(ifs, tmp); } // get the 7th line(number of atoms)
    int nat = std::stoi(tmp);
    for (int i = 0; i != nat; ++i) { std::getline(ifs, tmp); }
    int nk_original = klist->get_nks();
    std::cout << "Origianl klist(Cartesian|Direct)" << std::endl;
    for (int ik = 0;ik < nk_original;++ik)
    {
        std::cout << "ik=" << ik <<": " << klist->kvec_c[ik].x << " " << klist->kvec_c[ik].y << " " << klist->kvec_c[ik].z 
        << " | " << klist->kvec_d[ik].x << " " << klist->kvec_d[ik].y << " " << klist->kvec_d[ik].z << std::endl;
    }

    ifs >> klist->nmp[0] >> klist->nmp[1] >> klist->nmp[2];
    int nk = klist->nmp[0] * klist->nmp[1] * klist->nmp[2];
    klist->set_nks(nk);
    klist->kvec_c.resize(nk);
    klist->kvec_d.resize(nk);
    klist->wk.resize(nk);
    for (int ik = 0;ik < nk;++ik)
    {
        ifs >> klist->kvec_c[ik].x >> klist->kvec_c[ik].y >> klist->kvec_c[ik].z;
        klist->kvec_c[ik] /= ModuleBase::TWO_PI * ModuleBase::BOHR_TO_A; // in unit of 2pi/angstrom
        klist->kvec_d[ik] = klist->kvec_c[ik] * ucell.latvec;
    }
    std::cout << "After read_kpts: klist(Cartesian|Direct)" << std::endl;
    for (int ik = 0;ik < nk;++ik)
    {
        std::cout << "ik=" << ik <<": " << klist->kvec_c[ik].x << " " << klist->kvec_c[ik].y << " " << klist->kvec_c[ik].z 
        << " | " << klist->kvec_d[ik].x << " " << klist->kvec_d[ik].y << " " << klist->kvec_d[ik].z << std::endl;
    }
}

std::vector<double> read_energy_qp(const std::string& file,
                                   const int nocc,
                                   const int nvirt,
                                   int& ncore,
                                   const int nk,
                                   const int nspin_tmp,
                                   const int nspin_file)
{
    std::cout << "in read_energy_qp" << std::endl;
    std::cout << "FISH_OUTPUT: nbands(nocc+nvir): " << (nocc+nvirt) << std::endl;
    std::vector<double> eig_info( 3 * nk * nspin_tmp * (nocc + nvirt)); // occ, eig_ks, eig_gw
    std::ifstream file_gw (file);
    if (!file_gw) throw std::runtime_error(file + " not found");
    std::string temp;
    int read_ik;
    double occ, eig_ks, eig_gw;

    for (int is =0; is < nspin_file; ++is){
        for (int ik = 0; ik < nk; ++ik){
            for (int i = 0;i < 2;++i) { std::getline(file_gw, temp); } // skip the first 2 lines
            file_gw >> temp >> read_ik ;
            std::cout<<"read_ik: " << read_ik <<" is:" << is << std::endl;
            assert(ik == (read_ik-1));
            int ivirt = 0;
            std::getline(file_gw, temp); // skip the interval line
            std::getline(file_gw, temp); // skip the interval line
            std::vector<double> ks_temps;
            std::vector<double> gw_temps;
            std::vector<double> occ_temps;
            while (file_gw.peek() != '-')
            {
                std::getline(file_gw, temp);
                std::istringstream iss(temp);
                iss >> temp >> occ >> eig_ks >> eig_gw;
                ks_temps.push_back(eig_ks * 2); // Ha to Ry
                gw_temps.push_back(eig_gw * 2); // Ha to Ry
                occ_temps.push_back(occ);
                if (occ < 0.1) { ivirt++;}
                if (ivirt == nvirt) { break; }
            }
            ncore = gw_temps.size() - nocc - nvirt;
            for (int ib = 0;ib < nocc + nvirt;++ib)
            {   
                int ikstep = (ik + is * nk) * (nocc + nvirt);
                eig_info[(ikstep + ib)*3] = occ_temps[ncore + ib];
                eig_info[(ikstep + ib)*3 + 1] = ks_temps[ncore + ib];
                eig_info[(ikstep + ib)*3 + 2] = gw_temps[ncore + ib];
                std::cout <<"FISH_OUTPUT: ik=" << ik << "\t" << ib << "\t"
                            << eig_info[(ikstep + ib)*3] << "\t" 
                            << eig_info[(ikstep + ib)*3 + 1] << "\t" 
                            << eig_info[(ikstep + ib)*3 + 2] << std::endl; //check
            }
            while (file_gw.peek() != '-' && file_gw.peek() != EOF)
            {
                std::getline(file_gw, temp); // skip the virtual bands to next k-point
            }            
        }
    }
    if (nspin_file == 1 && nspin_tmp == 2) {
        std::cout << "duplicate the spin channel since the gw file only has one spin channel" << std::endl;
        int spin_block = nk * (nocc + nvirt) * 3;
        assert(eig_info.size() == 2 * spin_block);
        std::copy_n(eig_info.data(), spin_block, eig_info.data() + spin_block);
    }        
    file_gw.close();
    std::cout << "FISH_OUTPUT: Finish read gw, ncore=" << ncore << std::endl;
    return eig_info;
}

/// @brief  read the eigenvectors from librpa
template <typename TK>
void read_librpa_eigenvectors(psi::Psi<TK>& wfc_ks,
                              psi::Psi<TK>& wfc_ks_global,
                              const std::string& path,
                              const int ncore,
                              const int nbands_file,
                              const int nspin_tmp,
                              const int nspin_file,
                              Parallel_Orbitals& pmat)
{
    int nbands = pmat.get_wfc_global_nbands();// nbands = nocc + nvirt
    int nbasis = pmat.get_wfc_global_nbasis();
    assert(nbands == wfc_ks_global.get_nbands());
    assert(nbasis == wfc_ks_global.get_nbasis());
    const size_t nk = PARAM.inp.nspin == 2 ? wfc_ks.get_nk() / 2 : wfc_ks.get_nk();

    if (GlobalV::MY_RANK == 0)
    {
        struct dirent *ptr;
        DIR *dir;
        dir = opendir(path.c_str());
        std::vector<bool> readen_k(nk, false);

        while ((ptr = readdir(dir)) != NULL){// read all the files in the directory
            std::string fm(ptr->d_name);
            if (fm.find("KS_eigenvector") == 0)// find file KS_eigenvectorXXX
            {
                std::cout << "found librpa_eigenvector file:" << fm << std::endl;
                std::ifstream file_librpa_ks(path + fm);
                std::string tmp;
                while (file_librpa_ks.peek() != EOF)
                {
                    int ik;
                    file_librpa_ks >> ik;
                    ik = ik - 1;
                    assert(readen_k[ik] == false);
                    for (int iw = 0; iw < nbasis; ++iw) {
                        for (int ib = 0; ib < nbands_file; ++ib) {
                            for (int is = 0; is < nspin_file; ++is) {
                                if (ib >= ncore && ib< (ncore+nbands)) {
                                    BSE_IO::read_one_data(file_librpa_ks, wfc_ks_global(ik+is*nk, ib-ncore, iw));
                                    file_librpa_ks >> std::ws; // skip the blank if there is
                                }
                                else {
                                    std::getline(file_librpa_ks, tmp); //skip the useless bands
                                }
                            }
                        }
                    }
                    if (nspin_tmp == 2 && nspin_file == 1) {
                        std::copy_n(&wfc_ks_global(ik,0,0), nbands*nbasis, &wfc_ks_global(ik + nk,0,0));
                    }
                    readen_k[ik] = true;
                }
            }
        }
        closedir(dir);
        for(int ik = 0; ik < nk; ++ik) {
            if (!readen_k[ik])
                throw std::runtime_error("librpa_eigenvector file not found for k-point " + std::to_string(ik+1));
        }
        
    }// end of if (GlobalV::MY_RANK == 0); next MPI_comm to other ranks
    for (int iks = 0; iks < wfc_ks.get_nk(); ++iks){
        if (GlobalV::MY_RANK == 0) {
            // test: output wfc            
            std::cout << "wfc_gs_read_from_librpa for iks:" << iks << std::endl;
            for (int ib = 0; ib < nbands; ++ib)
            {
                std::cout << "band " << ib << ": ";
                for (int iw = 0; iw < nbasis; ++iw)
                {
                    std::cout << wfc_ks_global(iks, ib, iw) << "  ";
                }
                std::cout << std::endl;
            }

            // change wfc_ks_global phase to make arg(<psi(k)|psi(k'=0)>) = 0
            if (iks != 0)
            {
                for(int ib = 0; ib < nbands; ++ib)
                {
                    TK phase = BSE_Util::inner_product(&wfc_ks_global(iks,ib,0), &wfc_ks_global(0,ib,0), nbasis);
                    phase = phase / std::abs(phase);
                    for (int iw = 0; iw < nbasis; ++iw)
                    {
                        wfc_ks_global(iks, ib, iw) *= phase;
                    }
                    TK test_phase = BSE_Util::inner_product(&wfc_ks_global(iks,ib,0), &wfc_ks_global(0,ib,0), nbasis);
                    std::cout << "After phase correction, iks, ib, phase: " << iks << " " << ib << " " << test_phase << std::endl;
                }
            }
        }

        wfc_ks_global.fix_k(iks);
        wfc_ks.fix_k(iks);
#ifdef __MPI
        MPI_Bcast(wfc_ks_global.get_pointer(), nbands * nbasis, BSE_Util::MPIType<TK>::value, 0, MPI_COMM_WORLD);
        Parallel_2D pv_glb;
        pv_glb.set(nbasis, nbands, std::max(nbasis, nbands), pmat.blacs_ctxt);
        Cpxgemr2d(nbasis, nbands, wfc_ks_global.get_pointer(), 1, 1, pv_glb.desc,
                    wfc_ks.get_pointer(), 1, 1, const_cast<int*>(pmat.desc_wfc)/*nbasis×nbands*/,
                    pv_glb.blacs_ctxt);
#else
        BlasConnector::copy(nbands*nlocal, wfc_ks_global.get_pointer(), 1, wfc_ks.get_pointer(), 1);
#endif
    }
}

template<typename Tdata, typename TR>
auto read_Ws(const TLRI<TR>& Vs, const std::vector<TC>& Rlist)
-> std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>
{
    ModuleBase::TITLE("BSE", "read_Ws");
    std::map<TA,std::map<TAC,RI::Tensor<Tdata>>> Ws;
    
    const int nat = Vs.size();
    std::string temp;
    int nk, istart, iend, jstart, jend, ik;
    size_t nabfmu, nabfnu, non_zero, mu, nu; //I.nab, J.nab
    size_t nR = Rlist.size();
    for(int iat = 0; iat != nat; ++iat)//loop atom I
    {
        for(int jat = 0; jat != nat; ++jat)//loop atom J
        {
            for(int iR = 0; iR < nR; ++iR)
            {
                std::ifstream infileW;
                std::string filename = "Wc_Mu_"+std::to_string(iat)+"_Nu_"+std::to_string(jat)+"_iR_"+std::to_string(iR)+"_ifreq_0.mtx";
                infileW.open("librpa.d/" + filename);
                if(!infileW) throw std::runtime_error( filename + " not found!");
                else std::cout << "reading Wc file: " << filename << std::endl;
                int nabf1 = Vs.at(iat).at({jat,{0,0,0}}).shape[0];
                int nabf2 = Vs.at(iat).at({jat,{0,0,0}}).shape[1];
                while(infileW.peek() == '%') infileW.ignore(2048, '\n');	//skip comments

                infileW >> nabfmu >> nabfnu >> non_zero;
                assert(nabfmu == nabf1);
                assert(nabfnu == nabf2);
                RI::Tensor<Tdata> tensor_W({ nabfmu, nabfnu });
                for (int index = 0; index < non_zero; ++index)
                {
                    infileW >> mu >> nu ;
                    BSE_IO::read_one_data(infileW, tensor_W(mu-1, nu-1));
                }
                infileW.close();
                for(int i = 0; i != nabf1; ++i)
                    for(int j = 0; j != nabf2; ++j)
                    {
                        tensor_W(i, j) += Vs.at(iat).at({jat, Rlist[iR]})(i,j);
                        //std::cout << "FISH_OUTPUT: Wxc: " << i << " " << j << " " << tensor_W(i,j) << std::endl; //check
                    }
                Ws[iat][{jat, Rlist[iR]}] = tensor_W;
                std::cout << "FISH_OUTPUT: Finish read W for iat, jat, iR: " << iat << " " << jat << " " << iR << std::endl;
            }
        }
    }
    return Ws;
}

template void read_librpa_eigenvectors<double>(
    psi::Psi<double>& wfc_ks, psi::Psi<double>& wfc_ks_global,
    const std::string& path, const int ncore, const int nbands_file,
    const int nspin_tmp, const int nspin_file, Parallel_Orbitals& pmat);

template void read_librpa_eigenvectors<std::complex<double>>(
    psi::Psi<std::complex<double>>& wfc_ks, psi::Psi<std::complex<double>>& wfc_ks_global,
    const std::string& path, const int ncore, const int nbands_file,
    const int nspin_tmp, const int nspin_file, Parallel_Orbitals& pmat);

template std::map<TA, std::map<TAC, RI::Tensor<double>>> 
read_Ws<double, double>(const TLRI<double>& Vs, const std::vector<TC>& Rlist);

template std::map<TA, std::map<TAC, RI::Tensor<std::complex<double>>>>
read_Ws<std::complex<double>, double>(const TLRI<double>& Vs, const std::vector<TC>& Rlist);

}// end of namespace BSE_IO
