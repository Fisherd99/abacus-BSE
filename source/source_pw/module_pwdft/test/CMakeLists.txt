abacus_disable_feature_definitions(__MLALGO)
abacus_disable_feature_definitions(__CUDA)
abacus_disable_feature_definitions(__ROCM)
abacus_disable_feature_definitions(__EXX)

AddTest(
  TARGET MODULE_PW_finite_field_pw_test
  LIBS parameter psi planewave device base
  SOURCES finite_field_pw_test.cpp ../finite_field_pw.cpp
)

AddTest(
  TARGET MODULE_PW_finite_field_pw_mpi_test
  LIBS parameter psi planewave device base MPI::MPI_CXX
  SOURCES finite_field_pw_mpi_test.cpp ../finite_field_pw.cpp
)

add_test(
  NAME MODULE_PW_finite_field_pw_test_parallel
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
    $<TARGET_FILE:MODULE_PW_finite_field_pw_mpi_test>
  WORKING_DIRECTORY $<TARGET_FILE_DIR:MODULE_PW_finite_field_pw_mpi_test>
)

AddTest(
  TARGET MODULE_PW_op_finite_field_test
  LIBS parameter psi device base
  SOURCES op_pw_finite_field_test.cpp
    ../op_pw_finite_field.cpp
    ../op_pw.cpp
    ../../../source_hamilt/operator.cpp
)

AddTest(
  TARGET MODULE_PW_pwdft_soc
  LIBS parameter
  SOURCES soc_test.cpp ../soc.cpp
  	../../../source_base/global_variable.cpp
  	../../../source_base/global_function.cpp
	../../../source_base/global_file.cpp
	../../../source_base/tool_quit.cpp
	../../../source_base/global_variable.cpp
	../../../source_base/global_file.cpp
	../../../source_base/memory_recorder.cpp
	../../../source_base/timer.cpp
	../../../source_base/module_external/blas_connector_base.cpp ../../../source_base/module_external/blas_connector_vector.cpp ../../../source_base/module_external/blas_connector_matrix.cpp
	../../../source_base/parallel_global.cpp
	../../../source_base/parallel_comm.cpp
	../../../source_base/parallel_common.cpp
	../../../source_base/parallel_reduce.cpp
)

AddTest(
	TARGET MODULE_PW_dftu_base_test
	LIBS parameter base device
	SOURCES dftu_base_test.cpp
	../dftu_base.cpp
	../dftu_base_io.cpp
	../yukawa_screening.cpp
	../../../source_estate/occ_matrix.cpp
	../../../source_estate/occ_mixer.cpp
	../../../source_cell/unitcell.cpp
	../../../source_cell/atom_spec.cpp
	../../../source_cell/atom_pseudo.cpp
	../../../source_cell/pseudo.cpp
	../../../source_cell/magnetism.cpp
	../../../source_cell/sep.cpp
	../../../source_cell/sep_cell.cpp
	../../../source_cell/read_atom_species.cpp
	../../../source_cell/read_atoms.cpp
	../../../source_cell/read_atoms_helper.cpp
	../../../source_cell/cell_tools.cpp
	../../../source_cell/print_cell.cpp
	../../../source_cell/read_orb.cpp
	../../../source_cell/read_stru.cpp
	../../../source_cell/bcast_cell.cpp
)

AddTest(
	TARGET MODULE_PW_test_dftu_pw_tools
	LIBS parameter base device
	SOURCES test_dftu_pw_tools.cpp
	../dftu_pw_tools.cpp
)

AddTest(
	TARGET MODULE_PW_structure_factor_test
	LIBS parameter base device planewave
	SOURCES structure_factor_test.cpp ../stru_fac.cpp
	../../../source_cell/unitcell.cpp

	../../../source_cell/update_cell.cpp
	../../../source_cell/bcast_cell.cpp
	../../../source_cell/print_cell.cpp
	../../../source_cell/atom_spec.cpp
	../../../source_cell/atom_pseudo.cpp
	../../../source_cell/pseudo.cpp
	../../../source_cell/read_stru.cpp
	../../../source_cell/read_atom_species.cpp
	../../../source_cell/read_atoms.cpp
	../../../source_cell/read_atoms_helper.cpp
	../../../source_cell/cell_tools.cpp
	../../../source_cell/read_pp.cpp
	../../../source_cell/read_pp_complete.cpp
	../../../source_cell/read_pp_upf100.cpp
	../../../source_cell/read_pp_upf201.cpp
	../../../source_cell/read_pp_vwr.cpp
	../../../source_cell/read_pp_blps.cpp
  ../../../source_cell/sep.cpp
  ../../../source_cell/sep_cell.cpp
	../../../source_cell/read_pp_ucell.cpp
	../../../source_cell/cal_wfc.cpp
	../../../source_cell/cal_nelec_nband.cpp
	../../../source_cell/read_orb.cpp
)
