//====- LowerToAffineLoops.cpp - Partial lowering from Toy to Affine+Std --===//
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
// This file implements a partial lowering of Toy operations to a combination of
// affine loops and standard operations. This lowering expects that all calls
// have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//
#include<stdio.h>

#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "toy/RelayDialect.h"

// #include "mlir/Dialect/AffineOps/AffineOps.h"
// #include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns
//===----------------------------------------------------------------------===//

/// Convert the given TensorType into the corresponding MemRefType.


/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input a rewriter, an array of memRefOperands corresponding
/// to the operands of the input operation, and the set of loop induction
/// variables for the iteration. It returns a value to store at the current
/// index of the iteration.
using RelayFn = function_ref<Value(PatternRewriter & rewriter,
                                   ArrayRef < Value > TensorOperands)>;

// Generate a call to the processing function with the rewriter, the memref
// operands, and the loop induction variables. This function will return the
// value to store at the current index.


namespace {
//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

    template<typename BinaryOp, typename LoweredBinaryOp>
    struct BinaryOpLowering : public ConversionPattern {
        BinaryOpLowering(MLIRContext *ctx) : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

        PatternMatchResult matchAndRewrite(Operation *op, ArrayRef <Value> operands,
                        ConversionPatternRewriter &rewriter) const final {
            auto loc = op->getLoc();
            auto type = op->getResult(0).getType();
            auto binaryopRelay = rewriter.create<LoweredBinaryOp>(loc, type, operands[0], operands[1]);
            rewriter.replaceOp(op, {binaryopRelay});
            return matchSuccess();
        }
    };

    template<typename Op, typename LoweredOp>
    struct ZeroOpLowering : public ConversionPattern {
        ZeroOpLowering(MLIRContext *ctx) : ConversionPattern(Op::getOperationName(), 1, ctx) {}

        PatternMatchResult matchAndRewrite(Operation *op, ArrayRef <Value> operands,
                        ConversionPatternRewriter &rewriter) const final {
            rewriter.replaceOpWithNewOp<LoweredOp>(op);
            return matchSuccess();
        }
    };

    template<typename Op, typename LoweredOp>
    struct UnaryOpLowering : public ConversionPattern {
        UnaryOpLowering(MLIRContext *ctx) : ConversionPattern(Op::getOperationName(), 1, ctx) {}

        PatternMatchResult matchAndRewrite(Operation *op, ArrayRef <Value> operands,
                                           ConversionPatternRewriter &rewriter) const final {
            auto loc = op->getLoc();
            auto type = op->getResult(0).getType();
            auto opRelay = rewriter.create<LoweredOp>(loc, type, operands[0]);
            rewriter.replaceOp(op, {opRelay});
            return matchSuccess();
        }
    };

    using AddOpLowering = BinaryOpLowering<toy::AddOp, relay::AddOp>;
    using MulOpLowering = BinaryOpLowering<toy::MulOp, relay::MulOp>;
    using BgtzOpLowering = BinaryOpLowering<toy::BgtzOp, relay::BgtzOp>;
    using BltzOpLowering = BinaryOpLowering<toy::BltzOp, relay::BltzOp>;
    using BiasAddLowering = BinaryOpLowering<toy::BiasAddOp, relay::BiasAddOp>;
    using DenseLowering = BinaryOpLowering<toy::DenseOp, relay::DenseOp>;
    using Conv1dOpLowering = BinaryOpLowering<toy::Conv1dOp, relay::Conv1dOp>;
    using IfOpLowering = ZeroOpLowering<toy::IfOp, relay::IfOp>;
    using ForOpLowering = ZeroOpLowering<toy::ForOp, relay::ForOp>;
    using LoopFieldOpLowering = ZeroOpLowering<toy::LoopFieldOp, relay::LoopFieldOp>;
    using LoopEndOpLowering = ZeroOpLowering<toy::LoopEndOp, relay::LoopEndOp>;
    using TransposeOpLowering = UnaryOpLowering<toy::TransposeOp, relay::TransposeOp>;
    using SoftmaxOpLowering = UnaryOpLowering<toy::SoftmaxOp, relay::SoftmaxOp>;
    using ReshapeOpLowering = UnaryOpLowering<toy::ReshapeOp, relay::ReshapeOp>;

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

    struct ConstantOpLowering : public OpRewritePattern<toy::ConstantOp> {
        using OpRewritePattern<toy::ConstantOp>::OpRewritePattern;

        PatternMatchResult matchAndRewrite(toy::ConstantOp op,
                                           PatternRewriter &rewriter) const final {
            DenseElementsAttr constantValue = op.value();
            Location loc = op.getLoc();
            auto constantRelay = rewriter.create<relay::ConstantOp>(loc, constantValue.getType(), constantValue);
            rewriter.replaceOp(op, {constantRelay});
            return matchSuccess();
        }
    };

    struct ConstOpLowering : public OpRewritePattern<toy::ConstOp> {
        using OpRewritePattern<toy::ConstOp>::OpRewritePattern;

        PatternMatchResult matchAndRewrite(toy::ConstOp op,
                                           PatternRewriter &rewriter) const final {
            DenseElementsAttr constValue = op.value();
            Location loc = op.getLoc();
            auto constRelay = rewriter.create<relay::ConstOp>(loc, constValue.getType(), constValue);
            rewriter.replaceOp(op, {constRelay});
            return matchSuccess();
        }
    };

    struct IndexOpLowering : public OpRewritePattern<toy::IndexOp> {
        using OpRewritePattern<toy::IndexOp>::OpRewritePattern;

        PatternMatchResult matchAndRewrite(toy::IndexOp op,
                                           PatternRewriter &rewriter) const final {
            auto index = op.index();
            auto name = op.name();
            Location loc = op.getLoc();
            auto indexRelay = rewriter.create<relay::IndexOp>(loc, index.getType(), name, index);
            rewriter.replaceOp(op, {indexRelay});
            return matchSuccess();
        }
    };

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

    struct ReturnOpLowering : public OpRewritePattern<toy::ReturnOp> {
        using OpRewritePattern<toy::ReturnOp>::OpRewritePattern;

        PatternMatchResult matchAndRewrite(toy::ReturnOp op,
                                           PatternRewriter &rewriter) const final {
            // During this lowering, we expect that all function calls have been
            // inlined.
            if (op.hasOperand())
                return matchFailure();

            // We lower "toy.return" directly to "std.return".
            rewriter.replaceOpWithNewOp<relay::ReturnOp>(op);
            //rewriter.eraseOp(op);
            return matchSuccess();
        }
    };

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

    struct PrintOpLowering : public ConversionPattern {
        PrintOpLowering(MLIRContext *ctx)
                : ConversionPattern(toy::PrintOp::getOperationName(), 1, ctx) {}

        PatternMatchResult matchAndRewrite(Operation *op, ArrayRef <Value> operands,
                                           ConversionPatternRewriter &rewriter) const final {
            //auto loc = op->getLoc();
            //auto printopRelay = rewriter.create<relay::PrintOp>(loc,operands[0]);
            rewriter.eraseOp(op);
            //rewriter.replaceOp(op, {printopRelay.getOperand()}, {printopRelay});
            return matchSuccess();
        }
    };

} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// ToyToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
    struct ToyToRelayLoweringPass : public FunctionPass<ToyToRelayLoweringPass> {
        void runOnFunction() final;
    };
} // end anonymous namespace.

void ToyToRelayLoweringPass::runOnFunction() {
    // Verify that the given main has no inputs and results.
    // if (function.getNumArguments() || function.getType().getNumResults()) {
    //   function.emitError("expected 'main' to have 0 inputs and 0 results");
    //   return signalPassFailure();
    // }

    // The first thing to define is the conversion target. This will define the
    // final target for this lowering.
    ConversionTarget target(getContext());

    // We define the specific operations, or dialects, that are legal targets for
    // this lowering. In our case, we are lowering to a combination of the
    // `Affine` and `Standard` dialects.
    target.addLegalDialect<relay::RelayDialect>();
    //target.addLegalDialect<toy::ToyDialect>();
    // We also define the Toy dialect as Illegal so that the conversion will fail
    // if any of these operations are *not* converted. Given that we actually want
    // a partial lowering, we explicitly mark the Toy operations that don't want
    // to lower, `toy.print`, as `legal`.
    //target.addIllegalDialect<toy::ToyDialect>();
    //target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
    target.addLegalOp<FuncOp>();

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the Toy operations.
    OwningRewritePatternList patterns;
    patterns.insert<AddOpLowering, ConstantOpLowering, MulOpLowering, ConstOpLowering,
            SoftmaxOpLowering, BiasAddLowering, DenseLowering, BltzOpLowering, 
            IndexOpLowering, LoopFieldOpLowering, LoopEndOpLowering,
            IfOpLowering, ForOpLowering, ReturnOpLowering, BgtzOpLowering, 
            ReshapeOpLowering, TransposeOpLowering, Conv1dOpLowering, PrintOpLowering>(&getContext());

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyPartialConversion(getFunction(), target, patterns)))
        signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr <Pass> mlir::toy::createLowerToRelayPass() {
    return std::make_unique<ToyToRelayLoweringPass>();
}
