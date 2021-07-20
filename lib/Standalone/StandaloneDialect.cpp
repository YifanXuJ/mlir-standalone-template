//===- StandaloneDialect.cpp - Standalone dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::standalone;

#define GET_TYPEDEF_CLASSES
#include "Standalone/StandaloneTypeDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void StandaloneDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Standalone/StandaloneOps.cpp.inc"
      >();
}

mlir::Type StandaloneDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef ref;
  if (parser.parseKeyword(&ref)) {
    return {};
  }
  return generatedTypeParser(getContext(), parser, ref);
}

void StandaloneDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
  // Currently the only toy type is a struct type.
  StructType structType = type.cast<StructType>();

  // Print the struct type according to the parser format.
  printer << "multiset<";
  llvm::interleaveComma(structType.getElementTypes(), printer);
  printer << '>';
}