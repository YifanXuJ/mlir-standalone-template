// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = standalone.bar %{{.*}} : i32
        %res = standalone.bar %0 : i32
        return
    }

    func @foo() {
        %0 = constant 1 : i32
        %1 = constant 1 : i32
        // CHECK: %{{.*}} = standalone.foo %{{.*}} : i32, i32
        %res = standalone.foo %0, %1: i32, i32
        return
    }
}
