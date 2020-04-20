#ifndef MLIR_TARGET_RELAYIR_H
#define MLIR_TARGET_RELAYIR_H

#include <memory>
#include "llvm/ADT/ScopedHashTable.h"

namespace mlir {

    class ModuleOp;

    class Pass;

    int translateModuleToRelayIR(ModuleOp m);

    namespace relay {
        std::unique_ptr <mlir::Pass> createRelayAPIPass(llvm::ScopedHashTable <mlir::Value, llvm::StringRef> &hashtable);
    }

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_H