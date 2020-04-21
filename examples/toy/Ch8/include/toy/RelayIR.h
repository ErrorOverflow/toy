#ifndef MLIR_TARGET_RELAYIR_H
#define MLIR_TARGET_RELAYIR_H

#include <memory>
#include <unordered_map>

namespace mlir {

    class ModuleOp;

    class Pass;

    int translateModuleToRelayIR(ModuleOp m);

    namespace relay {
        std::unique_ptr <mlir::Pass> createRelayAPIPass(std::unordered_map <uint32_t, std::string> &hashtable);
    }

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_H