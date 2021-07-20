// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar(%{{.*}}: !standalone.multiset)
    func @bar(%p:!standalone.multiset){
        return
    }
}