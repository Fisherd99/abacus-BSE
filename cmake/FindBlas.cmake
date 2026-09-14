if(DEFINED BLAS_DIR)
    string(APPEND CMAKE_PREFIX_PATH ";${BLAS_DIR}")
endif()
if(DEFINED BLAS_LIBRARY)
    set(BLAS_LIBRARIES ${BLAS_LIBRARY})
endif()

find_package(BLAS REQUIRED)

# CMake's FindBLAS does not expose a normalized vendor name. Detect OpenBLAS
# from either its vendor-specific cache entry or the resolved library path so
# that ABACUS can use the optional OpenBLAS thread-control API.
if(BLAS_openblas_LIBRARY OR "${BLAS_LIBRARIES}" MATCHES "[Oo]pen[Bb][Ll][Aa][Ss]")
    add_compile_definitions(__OPENBLAS)
endif()

if(NOT TARGET BLAS::BLAS)
    add_library(BLAS::BLAS UNKNOWN IMPORTED)
    set_target_properties(BLAS::BLAS PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
	IMPORTED_LOCATION "${BLAS_LIBRARIES}")
endif()
