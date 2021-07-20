add_mlir_dialect(StandaloneOps standalone)
add_mlir_doc(StandaloneDialect StandaloneDialect Standalone/ -gen-dialect-doc)
add_mlir_doc(StandaloneOps StandaloneOps Standalone/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS StandaloneTypeDefs.td)
mlir_tablegen(StandaloneTypeDefs.h.inc -gen-typedef-decls -typedefs-dialect=standalone)
mlir_tablegen(StandaloneTypeDefs.cpp.inc -gen-typedef-defs -typedefs-dialect=standalone)
add_public_tablegen_target(MLIRStandaloneTypeIncGen)
add_dependencies(mlir-generic-headers MLIRStandaloneTypeIncGen)
