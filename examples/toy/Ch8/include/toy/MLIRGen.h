//===- MLIRGen.h - MLIR Generation from a Toy AST -------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file declares a simple interface to perform IR generation targeting MLIR
// from a Module AST for the Toy language.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_TOY_MLIRGEN_H_
#define MLIR_TUTORIAL_TOY_MLIRGEN_H_

#include <memory>
#include <unordered_map>
#include "llvm/ADT/StringRef.h"

namespace mlir {
    class MLIRContext;

    class OwningModuleRef;
} // namespace mlir

namespace toy {
    class ModuleAST;

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
    mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST,
                                std::unordered_map <uint32_t, std::string> &hashtable,
                                std::unordered_map<std::string, uint32_t> &counter);
} // namespace toy

#endif // MLIR_TUTORIAL_TOY_MLIRGEN_H_
