#ifndef MLIR_TARGET_RELAYIR_H
#define MLIR_TARGET_RELAYIR_H

#include <memory>

namespace mlir {

    class ModuleOp;

    class Pass;

    int translateModuleToRelayIR(ModuleOp m);

    namespace relay {
        std::unique_ptr <mlir::Pass> createRelayAPIPass();
    }

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_H