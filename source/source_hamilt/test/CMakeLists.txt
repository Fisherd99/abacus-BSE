AddTest(
  TARGET MODULE_HAMILT_finite_field_polarization_test
  SOURCES finite_field_polarization_test.cpp
    ../module_finite_field/finite_field_polarization.cpp
)

AddTest(
  TARGET MODULE_HAMILT_finite_field_matrix_test
  SOURCES finite_field_matrix_test.cpp
    ../module_finite_field/finite_field_matrix.cpp
)

AddTest(
  TARGET MODULE_HAMILT_ewald_dnrm2
  SOURCES dnrm2_test.cpp  ../module_ewald/dnrm2.cpp
)

AddTest(
  TARGET MODULE_HAMILT_ewald_rgen
  LIBS parameter base device
  SOURCES rgen_test.cpp ../module_ewald/h_ewald_pw.cpp ../module_ewald/dnrm2.cpp
)
