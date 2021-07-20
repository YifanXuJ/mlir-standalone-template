

#ifndef STANDALONE_TYPE
#define STANDALONE_TYPE

include "StandaloneDialect.td"
include "mlir/IR/OpBase.td"

// All of the types will extend this class.
class MultisetType<string name> : TypeDef<Standalone_Dialect, name> { }



// An alternate int type.
def MultisetType : MultisetType<"Multiset"> {
  let mnemonic = "multiset";

  let summary = "A multiset type with special semantics";

  let description = [{
    A multiset type. Similar to List Type, but without ordering and indexing.
  }];


  
  /*
  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();

    Attribute parseAttribute(DialectAsmParser &parser,
                             Type type) const override;
    void printAttribute(Attribute attr,
                        DialectAsmPrinter &printer) const override;

    // Provides a custom printing/parsing for some operations.
    Optional<ParseOpHook>
      getParseOperationHook(StringRef opName) const override;
    LogicalResult printOperation(Operation *op,
                                 OpAsmPrinter &printer) const override;

  }];
  */
  
  //let builders = [
    //TypeBuilder<(ins I32:$intParam)>,
  //];
}

#endif // STANDALONE_TYPE