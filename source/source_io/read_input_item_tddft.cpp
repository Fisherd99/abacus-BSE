#include "source_base/constants.h"
#include "source_base/tool_quit.h"
#include "read_input.h"
#include "read_input_tool.h"

namespace ModuleIO
{
void ReadInput::item_rt_tddft()
{ 
    // real time TDDFT
    {
        Input_Item item("td_dt");
        item.annotation = "time step for evolving wavefunction";
        item.reset_value = [](const Input_Item& item, Parameter& para) {
            if (para.input.td_dt == -1.0)
            {
                GlobalV::ofs_running << "td_dt don't exist, set td_dt with md_dt" << std::endl;
                para.input.td_dt = para.input.mdp.md_dt / para.input.estep_per_md;
            }
        };
        read_sync_double(input.td_dt); 
        this->add_item(item);
    }
    {
        Input_Item item("estep_per_md");
        item.annotation = "steps of force change";
        read_sync_int(input.estep_per_md);
        this->add_item(item);
    }
    {
        Input_Item item("td_vext");
        item.annotation = "add extern potential or not";
        read_sync_bool(input.td_vext);
        this->add_item(item);
    }
    // {
    //     Input_Item item("td_vext_dire");
    //     item.annotation = "extern potential direction";
    //     item.read_value = [](const Input_Item& item, Parameter& para) {
    //         para.input.td_vext_dire = longstring(item.str_values);
    //     };
    //     sync_string(input.td_vext_dire);
    //     this->add_item(item);
    // }
    {
        Input_Item item("td_vext_dire");
        item.annotation = "extern potential direction";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            parse_expression(item.str_values, para.input.td_vext_dire);
        };
        item.get_final_value = [](Input_Item& item, const Parameter& para) {
            if (item.is_read())
            {
                item.final_value.str(longstring(item.str_values));
            }
        };
        add_intvec_bcast(input.td_vext_dire, para.input.td_vext_dire.size(), 0);
        this->add_item(item);
    }
    {
        Input_Item item("init_vecpot_file");
        item.annotation = "init vector potential through file or not";
        read_sync_bool(input.init_vecpot_file);
        this->add_item(item);
    }
    {
        Input_Item item("td_print_eij");
        item.annotation = "print eij or not";
        read_sync_double(input.td_print_eij);
        this->add_item(item);
    }
    {
        Input_Item item("td_edm");
        item.annotation = "the method to calculate the energy density matrix";
        read_sync_int(input.td_edm);
        this->add_item(item);
    }
    {
        Input_Item item("td_propagator");
        item.annotation = "method of propagator";
        read_sync_int(input.propagator);
        this->add_item(item);
    }
    {
        Input_Item item("td_stype");
        item.annotation = "type of electric field in space domain";
        read_sync_int(input.td_stype);
        this->add_item(item);
    }
    {
        Input_Item item("td_ttype");
        item.annotation = "type of electric field in time domain";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_ttype = longstring(item.str_values);
        };
        sync_string(input.td_ttype);
        this->add_item(item);
    }
    {
        Input_Item item("td_tstart");
        item.annotation = " number of steps where electric field starts";
        read_sync_int(input.td_tstart);
        this->add_item(item);
    }
    {
        Input_Item item("td_tend");
        item.annotation = "number of steps where electric field ends";
        read_sync_int(input.td_tend);
        this->add_item(item);
    }
    {
        Input_Item item("td_lcut1");
        item.annotation = "cut1 of interval in length gauge";
        read_sync_double(input.td_lcut1);
        this->add_item(item);
    }
    {
        Input_Item item("td_lcut2");
        item.annotation = "cut2 of interval in length gauge";
        read_sync_double(input.td_lcut2);
        this->add_item(item);
    }
    {
        Input_Item item("td_gauss_freq");
        item.annotation = "frequency (freq) of Gauss type electric field";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_gauss_freq = longstring(item.str_values);
        };
        sync_string(input.td_gauss_freq);
        this->add_item(item);
    }
    {
        Input_Item item("td_gauss_phase");
        item.annotation = "phase of Gauss type electric field";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_gauss_phase = longstring(item.str_values);
        };
        sync_string(input.td_gauss_phase);
        this->add_item(item);
    }
    {
        Input_Item item("td_gauss_sigma");
        item.annotation = "sigma of Gauss type electric field";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_gauss_sigma = longstring(item.str_values);
        };
        sync_string(input.td_gauss_sigma);
        this->add_item(item);
    }
    {
        Input_Item item("td_gauss_t0");
        item.annotation = "step number of time center (t0) of Gauss type electric field";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_gauss_t0 = longstring(item.str_values);
        };
        sync_string(input.td_gauss_t0);
        this->add_item(item);
    }
    {
        Input_Item item("td_gauss_amp");
        item.annotation = "amplitude of Gauss type electric field";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_gauss_amp = longstring(item.str_values);
        };
        sync_string(input.td_gauss_amp);
        this->add_item(item);
    }
    {
        Input_Item item("td_trape_freq");
        item.annotation = "frequency of Trapezoid type electric field";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_trape_freq = longstring(item.str_values);
        };
        sync_string(input.td_trape_freq);
        this->add_item(item);
    }
    {
        Input_Item item("td_trape_phase");
        item.annotation = "phase of Trapezoid type electric field";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_trape_phase = longstring(item.str_values);
        };
        sync_string(input.td_trape_phase);
        this->add_item(item);
    }
    {
        Input_Item item("td_trape_t1");
        item.annotation = "t1 of Trapezoid type electric field";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_trape_t1 = longstring(item.str_values);
        };
        sync_string(input.td_trape_t1);
        this->add_item(item);
    }
    {
        Input_Item item("td_trape_t2");
        item.annotation = "t2 of Trapezoid type electric field";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_trape_t2 = longstring(item.str_values);
        };
        sync_string(input.td_trape_t2);
        this->add_item(item);
    }
    {
        Input_Item item("td_trape_t3");
        item.annotation = "t3 of Trapezoid type electric field";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_trape_t3 = longstring(item.str_values);
        };
        sync_string(input.td_trape_t3);
        this->add_item(item);
    }
    {
        Input_Item item("td_trape_amp");
        item.annotation = "amplitude of Trapezoid type electric field";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_trape_amp = longstring(item.str_values);
        };
        sync_string(input.td_trape_amp);
        this->add_item(item);
    }
    {
        Input_Item item("td_trigo_freq1");
        item.annotation = "frequency 1 of Trigonometric type electric field";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_trigo_freq1 = longstring(item.str_values);
        };
        sync_string(input.td_trigo_freq1);
        this->add_item(item);
    }
    {
        Input_Item item("td_trigo_freq2");
        item.annotation = "frequency 2 of Trigonometric type electric field";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_trigo_freq2 = longstring(item.str_values);
        };
        sync_string(input.td_trigo_freq2);
        this->add_item(item);
    }
    {
        Input_Item item("td_trigo_phase1");
        item.annotation = "phase 1 of Trigonometric type electric field";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_trigo_phase1 = longstring(item.str_values);
        };
        sync_string(input.td_trigo_phase1);
        this->add_item(item);
    }
    {
        Input_Item item("td_trigo_phase2");
        item.annotation = "phase 2 of Trigonometric type electric field";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_trigo_phase2 = longstring(item.str_values);
        };
        sync_string(input.td_trigo_phase2);
        this->add_item(item);
    }
    {
        Input_Item item("td_trigo_amp");
        item.annotation = "amplitude of Trigonometric type electric field";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_trigo_amp = longstring(item.str_values);
        };
        sync_string(input.td_trigo_amp);
        this->add_item(item);
    }
    {
        Input_Item item("td_heavi_t0");
        item.annotation = "t0 of Heaviside type electric field";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_heavi_t0 = longstring(item.str_values);
        };
        sync_string(input.td_heavi_t0);
        this->add_item(item);
    }
    {
        Input_Item item("td_heavi_amp");
        item.annotation = "amplitude of Heaviside type electric field";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.td_heavi_amp = longstring(item.str_values);
        };
        sync_string(input.td_heavi_amp);
        this->add_item(item);
    }
    {
        Input_Item item("ocp");
        item.annotation = "change occupation or not";
        read_sync_bool(input.ocp);
        this->add_item(item);
    }
    {
        Input_Item item("ocp_set");
        item.annotation = "set occupation";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            parse_expression(item.str_values, para.input.ocp_kb);
        };
        item.get_final_value = [](Input_Item& item, const Parameter& para) {
            if(item.is_read())
            {
                item.final_value.str(longstring(item.str_values));
            }
        };
        add_doublevec_bcast(input.ocp_kb, para.input.ocp_kb.size(), 0.0);
        this->add_item(item);
    }


}
void ReadInput::item_lr_tddft()
{
    // Linear Responce TDDFT
    {
        Input_Item item("lr_nstates");
        item.annotation = "the number of 2-particle states to be solved";
        read_sync_int(input.lr_nstates);
        this->add_item(item);
    }
    {
        Input_Item item("nocc");
        item.annotation = "the number of occupied orbitals to form the 2-particle basis ( <= nelec/2)";
        read_sync_int(input.nocc);
        item.reset_value = [](const Input_Item& item, Parameter& para) {
            const int nocc_default = std::max(static_cast<int>(para.input.nelec + 1) / 2, para.input.nbands);
            if (para.input.nocc <= 0 || para.input.nocc > nocc_default) { para.input.nocc = nocc_default; }
            };
        this->add_item(item);
    }
    {
        Input_Item item("nvirt");
        item.annotation = "the number of virtual orbitals to form the 2-particle basis (nocc + nvirt <= nbands)";
        read_sync_int(input.nvirt);
        this->add_item(item);
    }
    {
        Input_Item item("xc_kernel");
        item.annotation = "exchange correlation (XC) kernel for LR-TDDFT";
        read_sync_string(input.xc_kernel);
        this->add_item(item);
    }
    {
        Input_Item item("lr_init_xc_kernel");
        item.annotation = "The method to initalize the xc kernel";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            size_t count = item.get_size();
            auto& ifxc = para.input.lr_init_xc_kernel;
            for (int i = 0; i < count; i++) { ifxc.push_back(item.str_values[i]); }
            };
        item.reset_value = [](const Input_Item& item, Parameter& para) {
            if (para.input.lr_init_xc_kernel.empty()) { para.input.lr_init_xc_kernel.push_back("default"); }
            };
        sync_stringvec(input.lr_init_xc_kernel, para.input.lr_init_xc_kernel.size(), "default");
        this->add_item(item);
    }
    {
        Input_Item item("lr_solver");
        item.annotation = "the eigensolver for LR-TDDFT";
        read_sync_string(input.lr_solver);
        this->add_item(item);
    }
    {
        Input_Item item("lr_thr");
        item.annotation = "convergence threshold of the LR-TDDFT eigensolver";
        read_sync_double(input.lr_thr);
        this->add_item(item);
    }
    {
        Input_Item item("out_wfc_lr");
        item.annotation = "whether to output the eigenvectors (excitation amplitudes) in the particle-hole basis";
        read_sync_bool(input.out_wfc_lr);
        this->add_item(item);
    }
    {
        Input_Item item("lr_unrestricted");
        item.annotation = "Whether to use unrestricted construction for LR-TDDFT";
        read_sync_bool(input.lr_unrestricted);
        this->add_item(item);
    }
    {
        Input_Item item("abs_wavelen_range");
        item.annotation = "the range of wavelength(nm) to output the absorption spectrum ";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            size_t count = item.get_size();
            for (int i = 0; i < count; i++)
            {
                para.input.abs_wavelen_range.push_back(std::stod(item.str_values[i]));
            }
            };
        sync_doublevec(input.abs_wavelen_range, 2, 0.0);
        this->add_item(item);
    }
    {
        Input_Item item("abs_gauge");
        item.annotation = "whether to use length or velocity gauge to calculate the absorption spectrum in LR-TDDFT";
        read_sync_string(input.abs_gauge);
        this->add_item(item);
    }
    {
        Input_Item item("abs_broadening");
        item.annotation = "the broadening (eta) for LR-TDDFT absorption spectrum";
        read_sync_double(input.abs_broadening);
        this->add_item(item);
    }
    {
        Input_Item item("bse_tda");
        item.annotation = "whether Tamm-Dancoff Approximation is used (can be 'tda', 'full' or 'both')";
        read_sync_string(input.bse_tda);
        this->add_item(item);
    }
    {
        Input_Item item("bse_spin_types");
        item.annotation = "which spin type is calculated (can be 'singlet', 'triplet', also for test 'rpa', 'ipa')";

        item.read_value = [](const Input_Item& item, Parameter& para) {
            size_t count = item.get_size();
            auto& ist = para.input.bse_spin_types;
            ist.clear();
            for (int i = 0; i < count; i++) { ist.push_back(item.str_values[i]); }
            };
        item.reset_value = [](const Input_Item& item, Parameter& para) {
            if (para.input.bse_spin_types.empty()) { para.input.bse_spin_types={"singlet","triplet"}; }
            };
        sync_stringvec(input.bse_spin_types, para.input.bse_spin_types.size(), "singlet");
        this->add_item(item);
    }
    {
        Input_Item item("bse_mem_save");
        item.annotation = "whether to save memory by adding V and W to BSE matrix directly";
        item.reset_value = [](const Input_Item& item, Parameter& para) {
            if (para.input.bse_mem_save == true) { para.input.bse_continue=0; para.input.bse_ri_hartree=true; }
            };
        read_sync_bool(input.bse_mem_save);
        this->add_item(item);
    }
    {
        Input_Item item("bse_ri_hartree");
        item.annotation = "whether to use RI approximation for Hartree term in BSE";
        read_sync_bool(input.bse_ri_hartree);
        this->add_item(item);
    }
    {
        Input_Item item("bse_use_fine_kgrid");
        item.annotation = "whether to use a finer k-grid for BSE";
        read_sync_int(input.bse_use_fine_kgrid);
        this->add_item(item);
    }
    {
        Input_Item item("bse_q_approx_mode");
        item.annotation = "q->kpair mapping mode: 0=exact, 1=coarse q grid, 2=mixed";
        read_sync_int(input.bse_q_approx_mode);
        this->add_item(item);
    }
    {
        Input_Item item("bse_q_approx_threshold");
        item.annotation = "threshold radius (Bohr^-1) for exact q mapping in mode 2";
        read_sync_double(input.bse_q_approx_threshold);
        this->add_item(item);
    }
    {
        Input_Item item("bse_write_ab");
        item.annotation = "whether to write the AB matrix to file";
        read_sync_bool(input.bse_write_ab);
        this->add_item(item);
    }
    {
        Input_Item item("bse_continue");
        item.annotation = "which step to continue from previous BSE calculation";
        read_sync_int(input.bse_continue);
        this->add_item(item);
    }
    {
        Input_Item item("plot_istate");
        item.annotation = "which state of exciton to be ploted";
        read_sync_int(input.plot_istate);
        this->add_item(item);
    }
    {
        Input_Item item("exciton_plot_type");
        item.annotation = "exciton density plot type: average or conditional";
        read_sync_string(input.exciton_plot_type);
        this->add_item(item);
    }
    {
        Input_Item item("exciton_plot_format");
        item.annotation = "exciton plot format: auto, cube, slice, or both";
        read_sync_string(input.exciton_plot_format);
        this->add_item(item);
    }
    {
        Input_Item item("exciton_hole_fix_x");
        item.annotation = "fixed hole x position (Bohr) for conditional density";
        read_sync_double(input.exciton_hole_fix_x);
        this->add_item(item);
    }
    {
        Input_Item item("exciton_hole_fix_y");
        item.annotation = "fixed hole y position (Bohr) for conditional density";
        read_sync_double(input.exciton_hole_fix_y);
        this->add_item(item);
    }
    {
        Input_Item item("exciton_hole_fix_z");
        item.annotation = "fixed hole z position (Bohr) for conditional density";
        read_sync_double(input.exciton_hole_fix_z);
        this->add_item(item);
    }
    {
        Input_Item item("exciton_elec_fix_x");
        item.annotation = "fixed electron x position (Bohr) for conditional hole density";
        read_sync_double(input.exciton_elec_fix_x);
        this->add_item(item);
    }
    {
        Input_Item item("exciton_elec_fix_y");
        item.annotation = "fixed electron y position (Bohr) for conditional hole density";
        read_sync_double(input.exciton_elec_fix_y);
        this->add_item(item);
    }
    {
        Input_Item item("exciton_elec_fix_z");
        item.annotation = "fixed electron z position (Bohr) for conditional hole density";
        read_sync_double(input.exciton_elec_fix_z);
        this->add_item(item);
    }
    {
        Input_Item item("exciton_slice_plane");
        item.annotation = "cross-section plane: ab, bc, or ca";
        read_sync_string(input.exciton_slice_plane);
        this->add_item(item);
    }
    {
        Input_Item item("exciton_slice_pos");
        item.annotation = "offset along perpendicular direction (Bohr) for slice";
        read_sync_double(input.exciton_slice_pos);
        this->add_item(item);
    }
    {
        Input_Item item("exciton_slice_npoints");
        item.annotation = "grid points per dimension for slice";
        read_sync_int(input.exciton_slice_npoints);
        this->add_item(item);
    }
    {
        Input_Item item("exciton_slice_scale");
        item.annotation = "scale relative to BvK supercell for slice";
        read_sync_double(input.exciton_slice_scale);
        this->add_item(item);
    }
}
}
