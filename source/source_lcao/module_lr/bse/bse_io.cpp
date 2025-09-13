#include "bse_io.h"

namespace BSE{

    RI_kRlist::RI_kRlist(const std::string& file, const UnitCell& ucell){
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

    void RI_kRlist::read_kpts(const std::string& file, const UnitCell& ucell, std::unique_ptr<K_Vectors> & klist)
    {
        std::ifstream ifs;
        ifs.open(file);
        if (!ifs) throw std::runtime_error(file + "not found");
        std::string tmp;
        for (int i = 0; i < 7; ++i) { std::getline(ifs, tmp); } // get the 7th line(number of atoms)
        std::cout << "FISH_output: nat:" << tmp << std::endl;
        int nat = std::stoi(tmp);
        for (int i = 0; i != nat; ++i) { std::getline(ifs, tmp); }
        ifs >> klist->nmp[0] >> klist->nmp[1] >> klist->nmp[2];
        int nk = klist->nmp[0] * klist->nmp[1] * klist->nmp[2];
        std::cout << "FISH_output: nmp: " << klist->nmp[0] << " " << klist->nmp[1] << " " << klist->nmp[2] << std::endl;
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
        std::cout << "FISH_output: klist(Cartesian|Direct)" << std::endl;
        for (int ik = 0;ik < nk;++ik)
        {
            std::cout << "ik=" << ik <<": " << klist->kvec_c[ik].x << " " << klist->kvec_c[ik].y << " " << klist->kvec_c[ik].z 
            << " | " << klist->kvec_d[ik].x << " " << klist->kvec_d[ik].y << " " << klist->kvec_d[ik].z << std::endl;
        }
    }

    std::vector<std::vector<std::pair<double,double>>> read_energy_qp(const std::string& file,
    const int nocc, const int nvirt, int& ncore, const int nk, const int nspin_tmp, const int nspin_file)
    {
        std::cout << "in read_energy_qp" << std::endl;
        std::cout << "FISH_OUTPUT: nbands(nocc+nvir): " << (nocc+nvirt) << std::endl;
        std::vector<std::vector<std::pair<double, double>>> eig_gw(nk*nspin_tmp, std::vector<std::pair<double, double>>(nocc + nvirt));
        std::ifstream file_gw (file);
        if (!file_gw) throw std::runtime_error(file + "not found");
        std::string temp;
        int read_ik;
        double occ, gw_temp;
        // while(file_gw.peek() == '%') file_gw.ignore(2048, '\n');	//skip comments

        for (int is =0; is < nspin_file; ++is){
            for (int ik = 0; ik < nk; ++ik){
                for (int i = 0;i < 2;++i) { std::getline(file_gw, temp); } // skip the first 2 lines
                file_gw >> temp >> read_ik ;
                std::cout<<"read_ik: " << read_ik <<" is:" << is << std::endl;
                assert(ik == (read_ik-1));
                int ivirt = 0;
                std::getline(file_gw, temp); // skip the interval line
                std::getline(file_gw, temp); // skip the interval line
                std::vector<double> gw_temps;
                std::vector<double> occ_temps;
                while (file_gw.peek() != '-')
                {
                    std::getline(file_gw, temp);
                    std::istringstream iss(temp);
                    iss >> temp >> occ >> temp >> gw_temp;
                    gw_temps.push_back(gw_temp * 2); // Ha to Ry
                    occ_temps.push_back(occ);
                    if (occ < 0.1) { ivirt++;}
                    if (ivirt == nvirt) { break; }
                }
                int ncore = gw_temps.size() - nocc - nvirt;
                for (int ib = 0;ib < nocc + nvirt;++ib)
                {
                    eig_gw[ik+is*nk][ib] = std::pair<double, double>(occ_temps[ncore + ib], gw_temps[ncore + ib]);
                    std::cout <<"FISH_OUTPUT: ik=" << ik << "\t" << ib << "\t" << eig_gw[ik+is*nk][ib].first 
                        << "\t" << eig_gw[ik+is*nk][ib].second << std::endl; //check
                }
                while (file_gw.peek() != '-' && file_gw.peek() != EOF)
                {
                    std::getline(file_gw, temp); // skip the virtual bands to next k-point
                }            
            }
        }
        if (nspin_file == 1 && nspin_tmp == 2) {
            for (int ik = 0; ik < nk; ++ik) {
                eig_gw[ik + nk] = eig_gw[ik];
            }
        }        
        file_gw.close();
        std::cout << "FISH_OUTPUT: Finish read gw, ncore=" << ncore << std::endl;
        return eig_gw;
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
                        BSE::read_one_data(infileW, tensor_W(mu-1, nu-1));
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

    template std::map<TA, std::map<TAC, RI::Tensor<double>>> 
    read_Ws<double, double>(const TLRI<double>& Vs, const std::vector<TC>& Rlist);

    template std::map<TA, std::map<TAC, RI::Tensor<std::complex<double>>>>
    read_Ws<std::complex<double>, double>(const TLRI<double>& Vs, const std::vector<TC>& Rlist);
}
