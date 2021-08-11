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
include "StandaloneTypeDefs.td"

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


def NewMultisetOp : Standalone_Op<"newMultiset", [NoSideEffect]> {
    let summary = "Initialize a new Multiset";
    let description = [{
        Given an integer, this op should return a Multiset containing 
        this single integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the bar operation to %0
        %1 = standalone.newMultiset %0
        ```
    }];

    let arguments = (ins I32:$input);

    let results = (outs MultisetType:$res);

    // Specify a parser and printer method.
    // let parser = [{ return ::parseNewMultisetOp(parser, result); }];
    // let printer = [{ return ::print(p, *this); }];
    let assemblyFormat = [{ $input attr-dict }];

}

def UnionOp : Standalone_Op<"union", [NoSideEffect]> {
    let summary = "Return the union of two multiset";
    let description = [{
        Given two multisets, this op should return a Multiset containing 
        their union.

        Example:

        ```mlir
        %0 = !standalone.multiset
        %1 = !standalone.multiset
        // Apply the bar operation to %0
        %1 = standalone.newMultiset %0 
        ```
    }];

    // syntax for multiple inputs or outputs
    let arguments = (ins MultisetType:$input1, MultisetType:$input2);

    let results = (outs MultisetType:$res);

    // parser and printer for multiple inputs and outputs
         // let parser = [{ return ::parseNewMultisetOp(parser, result); }];
         // let printer = [{ return ::print(p, *this); }];
    let assemblyFormat = [{ $input attr-dict }];

}


#endif // STANDALONE_OPS
