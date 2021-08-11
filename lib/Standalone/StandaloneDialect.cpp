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
#include "llvm/ADT/TypeSwitch.h"

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

  // is the above #include enough? should we add sth below
  // addOperations<NewMultisetOp>();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Standalone/StandaloneTypeDefs.cpp.inc"
    >();
}

mlir::Type StandaloneDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef ref;
  if (parser.parseKeyword(&ref)) {
    return {};
  }
  Type res;
  auto parsed = generatedTypeParser(getContext(), parser, ref, res);
  if (parsed.hasValue() && succeeded(parsed.getValue()))
    return res;
  return {};
}

void StandaloneDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
  auto wasPrinted = generatedTypePrinter(type, printer);
  assert(succeeded(wasPrinted));
}

