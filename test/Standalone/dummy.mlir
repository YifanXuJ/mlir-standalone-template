// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = standalone.bar %{{.*}} : i32
        %res = standalone.bar %0 : i32
        return
    }

    // CHECK-LABEL: func @foo()
    func @foo() {
        %0 = constant 1 : i32
        %res = standalone.newMultiset %0
        return
    }
}
