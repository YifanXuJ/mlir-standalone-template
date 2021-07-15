add_mlir_dialect_library(MLIRStandalone
        StandaloneDialect.cpp
        StandaloneOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Standalone

        DEPENDS
        MLIRStandaloneOpsIncGen
        MLIRStandaloneTypeIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)

set(LLVM_TARGET_DEFINITIONS StandaloneTypeDefs.td)
mlir_tablegen(StandaloneTypeDefs.h.inc -gen-typedef-decls -typedefs-dialect=test)
mlir_tablegen(StandaloneTypeDefs.cpp.inc -gen-typedef-defs -typedefs-dialect=test)
add_public_tablegen_target(MLIRStandaloneTypeDefIncGen)
add_dependencies(mlir-generic-headers MLIRStandaloneTypeIncGen)