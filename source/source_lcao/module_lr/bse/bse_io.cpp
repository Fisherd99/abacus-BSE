#include "bse_io.h"

namespace BSE{
    std::vector<std::vector<std::pair<double,double>>> read_energy_qp(const std::string& file,
    const int nocc, const int nvirt, int& ncore, const int nk)
    {
        std::cout << "in read_energy_qp" << std::endl;
        std::cout << "FISH_OUTPUT: nbands(nocc+nvir): " << (nocc+nvirt) << std::endl;
        std::vector<std::vector<std::pair<double, double>>> eig_gw(nk, std::vector<std::pair<double, double>>(nocc + nvirt));
        std::ifstream file_gw (file);
        if (!file_gw) throw std::runtime_error(file + "not found");
        std::string temp;
        int read_ik;
        double occ, gw_temp;
        // while(file_gw.peek() == '%') file_gw.ignore(2048, '\n');	//skip comments

        for (int i = 0;i < 2;++i) { std::getline(file_gw, temp); } // skip the first 2 lines
        for (int ik = 0; ik < nk; ++ik){
            file_gw >> temp >> read_ik ;
            assert(ik == (read_ik-1));
            int ivirt = 0;
            std::getline(file_gw, temp); // skip the interval line
            std::getline(file_gw, temp); // skip the interval line
            std::vector<double> gw_temps;
            std::vector<double> occ_temps;
            while (file_gw.peek() != '-')
            {
                file_gw >> temp >> occ >> temp >> gw_temp;
                gw_temps.push_back(gw_temp * 2); // Ha to Ry
                occ_temps.push_back(occ);
                if (occ < 0.1) { ivirt++;}
                if (ivirt == nvirt) { break; }
            }
            int ncore = gw_temps.size() - nocc - nvirt;
            for (int ib = 0;ib < nocc + nvirt;++ib)
            {
                eig_gw[ik][ib] = std::pair<double, double>(occ_temps[ncore+ib], gw_temps[ncore + ib]);
                std::cout <<"FISH_OUTPUT: ik=" << ik << "\t" << ib << " " << eig_gw[ik][ib].first << eig_gw[ik][ib].second << std::endl; //check
            }
            while (file_gw.peek() != '-' && file_gw.peek() != EOF)
            {
                std::getline(file_gw, temp); // skip the virtual bands to next k-point
            }
            std::cout << "FISH_OUTPUT: Finish read gw, ncore=" << ncore << std::endl;
        }
        file_gw.close();
        return eig_gw;    
    }

    template<typename Tdata, typename TR>
    auto read_Ws(const TLRI<TR>& Vs, const std::vector<TC>& Rlist)
    -> std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>
    {
        ModuleBase::TITLE("BSE", "read_Ws");
        std::map<TA,std::map<TAC,RI::Tensor<Tdata>>> Ws;
        
        const int nat = Vs.size();
        std::ifstream infileW;
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
                    std::string filename = "Wc_Mu_"+std::to_string(iat)+"_Nu_"+std::to_string(jat)+"_iR_"+std::to_string(iR)+"_ifreq_0.mtx";
                    infileW.open(PARAM.globalv.global_readin_dir + "librpa.d/" + filename);
                    if(!infileW) throw std::runtime_error( filename + " not found!");

                    int nabf1 = Vs.at(iat).at({jat,{0,0,0}}).shape[0];
                    int nabf2 = Vs.at(iat).at({jat,{0,0,0}}).shape[1];
                    while(infileW.peek() == '%') infileW.ignore(2048, '\n');	//skip comments

                    infileW >> nabfmu >> nabfnu >> non_zero;
                    assert(nabfmu == nabf1);
                    assert(nabfnu == nabf2);
                    RI::Tensor<Tdata> tensor_W({ nabfmu, nabfnu });
                    std::vector<Tdata> WcIJ(nabfmu * nabfnu, 0.0);
                    for (int index = 0; index < non_zero; ++index)
                    {
                        infileW >> mu >> nu ;
                        BSE::read_one_data(infileW, WcIJ[mu * nabfnu + nu]);
                    }
                    infileW.close();
                    for(int i = 0; i != nabf1; ++i)
                        for(int j = 0; j != nabf2; ++j)
                        {
                            tensor_W(i, j) = Vs.at(iat).at({jat, Rlist[iR]})(i,j) + WcIJ[i * nabf2 + j];
                            std::cout << "FISH_OUTPUT: Wxc: " << i << " " << j << " " << tensor_W(i,j) << std::endl; //check
                        }
                    Ws[iat][{jat, Rlist[iR]}] = tensor_W;
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
