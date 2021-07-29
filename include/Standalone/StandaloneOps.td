//===- StandaloneOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "StandaloneDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Standalone_BarOp : Standalone_Op<"bar", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `standalone.bar` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the bar operation to %0
        %1 = standalone.bar %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}


def NewMultisetOp : Standalone_Op<"newMultiset", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Initialize a new Multiset";
    let description = [{
        Given an integer, this op should return a Multiset containing 
        this single integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the bar operation to %0
        %1 = standalone.newMultiset %0 : multiset
        ```
    }];

    let arguments = (ins I32:$input);

    let results = (outs Multiset:$res);

    // Specify a parser and printer method.
    let parser = [{ return ::parseNewMultisetOp(parser, result); }];
    let printer = [{ return ::print(p, *this); }];

    let builders = [
    // In the example, "DenseElementsAttr" seems to be used for tensor value
    // Here we want to build a multiset with an i32 data, should we change this?
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

  ];
}


#endif // STANDALONE_OPS
