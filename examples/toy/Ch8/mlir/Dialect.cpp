//===- Dialect.cpp - Toy IR Dialect registration in MLIR ------------------===//
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
// This file implements the dialect for the Toy IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::toy;

//===----------------------------------------------------------------------===//
// ToyInlinerInterface
//===----------------------------------------------------------------------===//

/// This class defines the interface for handling inlining with Toy
/// operations.
struct ToyInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    //===--------------------------------------------------------------------===//
    // Analysis Hooks
    //===--------------------------------------------------------------------===//

    /// All operations within toy can be inlined.
    bool isLegalToInline(Operation *, Region *,
                         BlockAndValueMapping &) const final {
        return true;
    }

    //===--------------------------------------------------------------------===//
    // Transformation Hooks
    //===--------------------------------------------------------------------===//

    /// Handle the given inlined terminator(toy.return) by replacing it with a new
    /// operation as necessary.
    void handleTerminator(Operation *op,
                          ArrayRef <Value> valuesToRepl) const final {
        // Only "toy.return" needs to be handled here.
        auto returnOp = cast<ReturnOp>(op);

        // Replace the values directly with the return operands.
        assert(returnOp.getNumOperands() == valuesToRepl.size());
        for (const auto &it : llvm::enumerate(returnOp.getOperands()))
            valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }

    /// Attempts to materialize a conversion for a type mismatch between a call
    /// from this dialect, and a callable region. This method should generate an
    /// operation that takes 'input' as the only operand, and produces a single
    /// result of 'resultType'. If a conversion can not be generated, nullptr
    /// should be returned.
    Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                         Type resultType,
                                         Location conversionLoc) const final {
        return builder.create<CastOp>(conversionLoc, resultType, input);
    }
};

//===----------------------------------------------------------------------===//
// ToyDialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
ToyDialect::ToyDialect(mlir::MLIRContext *ctx) : mlir::Dialect("toy", ctx) {
    addOperations<
#define GET_OP_LIST

#include "toy/Ops.cpp.inc"

    >();
    addInterfaces<ToyInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.

/// Infer the output shape of the CastOp, this is required by the shape
/// inference interface.
void CastOp::inferShapes() { getResult().setType(getOperand().getType()); }

static void buildConstantOp(mlir::Builder *builder, mlir::OperationState &state,
                            double value) {
    auto dataType = RankedTensorType::get({}, builder->getF64Type());
    auto dataAttribute = DenseElementsAttr::get(dataType, value);
    ConstantOp::build(builder, state, dataType, dataAttribute);
}

/// Verifier for the constant operation. This corresponds to the `::verify(...)`
/// in the op definition.
static mlir::LogicalResult verify(ConstantOp op) {
    // If the return type of the constant is not an unranked tensor, the shape
    // must match the shape of the attribute holding the data.
    auto resultType = op.getResult().getType().cast<RankedTensorType>();
    if (!resultType)
        return success();

    // Check that the rank of the attribute type matches the rank of the constant
    // result type.
    auto attrType = op.value().getType().cast<mlir::TensorType>();
    if (attrType.getRank() != resultType.getRank()) {
        return op.emitOpError(
                "return type must match the one of the attached value "
                "attribute: ")
                << attrType.getRank() << " != " << resultType.getRank();
    }

    // Check that each of the dimensions match between the two types.
    for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
        if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
            return op.emitOpError(
                    "return type shape mismatches its attribute at dimension ")
                    << dim << ": " << attrType.getShape()[dim]
                    << " != " << resultType.getShape()[dim];
        }
    }
    return mlir::success();
}

static void buildConstOp(mlir::Builder *builder, mlir::OperationState &state,
                            StringRef data_struct, double value) {
    auto dataType = RankedTensorType::get({}, builder->getF64Type());
    auto dataAttribute = DenseElementsAttr::get(dataType, value);
    state.addTypes(dataType);
    state.addAttribute("value", dataAttribute);
    state.addAttribute("data_struct", builder->getStringAttr(data_struct));
}


//===----------------------------------------------------------------------===//
// AddOp

static void buildAddOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({lhs, rhs});
}

/// Infer the output shape of the AddOp, this is required by the shape inference
/// interface.
void AddOp::inferShapes() { getResult().setType(getOperand(0).getType()); }

static void buildGenericCallOp(mlir::Builder *builder,
                               mlir::OperationState &state, StringRef callee,
                               ArrayRef <mlir::Value> arguments) {
    // Generic call always returns an unranked Tensor initially.
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands(arguments);
    state.addAttribute("callee", builder->getSymbolRefAttr(callee));
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
    return getAttrOfType<SymbolRefAttr>("callee");
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() { return inputs(); }

static void buildIndexOp(mlir::Builder *builder, mlir::OperationState &state,
                       StringRef name, mlir::Value index) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands(index);
    state.addAttribute("name", builder->getStringAttr(name));
}

static void buildBoolOp(mlir::Builder *builder, mlir::OperationState &state,
                       StringRef value) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addAttribute("value", builder->getStringAttr(value));
}

static void buildStringOp(mlir::Builder *builder, mlir::OperationState &state,
                       StringRef value) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addAttribute("value", builder->getStringAttr(value));
}

//===----------------------------------------------------------------------===//
// BinOp
static void buildBinOp(mlir::Builder *builder, mlir::OperationState &state,
                       StringRef op, mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({lhs, rhs});
    state.addAttribute("op", builder->getStringAttr(op));
}

//===----------------------------------------------------------------------===//
// TupleOp
static void buildTupleOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// ReturnOp

static mlir::LogicalResult verify(ReturnOp op) {
    // We know that the parent operation is a function, because of the 'HasParent'
    // trait attached to the operation definition.
    auto function = cast<FuncOp>(op.getParentOp());

    /// ReturnOps can only have a single optional operand.
    if (op.getNumOperands() > 1)
        return op.emitOpError() << "expects at most 1 return operand";

    // The operand number and types must match the function signature.
    const auto &results = function.getType().getResults();
    if (op.getNumOperands() != results.size())
        return op.emitOpError()
                << "does not return the same number of values ("
                << op.getNumOperands() << ") as the enclosing function ("
                << results.size() << ")";

    // If the operation does not have an input, we are done.
    if (!op.hasOperand())
        return mlir::success();

    auto inputType = *op.operand_type_begin();
    auto resultType = results.front();

    // Check that the result type of the function matches the operand type.
    if (inputType == resultType || inputType.isa<mlir::UnrankedTensorType>() ||
        resultType.isa<mlir::UnrankedTensorType>())
        return mlir::success();

    return op.emitError() << "type of return operand ("
                          << *op.operand_type_begin()
                          << ") doesn't match function result type ("
                          << results.front() << ")";
}

//===----------------------------------------------------------------------===//
// TransposeOp

static void buildTransposeOp(mlir::Builder *builder,
                             mlir::OperationState &state, mlir::Value value) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands(value);
}

void TransposeOp::inferShapes() {
    auto arrayTy = getOperand().getType().cast<RankedTensorType>();
    SmallVector<int64_t, 2> dims(llvm::reverse(arrayTy.getShape()));
    getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
}

static mlir::LogicalResult verify(TransposeOp op) {
    auto inputType = op.getOperand().getType().dyn_cast<RankedTensorType>();
    auto resultType = op.getType().dyn_cast<RankedTensorType>();
    if (!inputType || !resultType)
        return mlir::success();

    auto inputShape = inputType.getShape();
    if (!std::equal(inputShape.begin(), inputShape.end(),
                    resultType.getShape().rbegin())) {
        return op.emitError()
                << "expected result shape to be a transpose of the input";
    }
    return mlir::success();
}

//===----------------------------------------------------------------------===//
// VarOp

static void buildVarOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value name, mlir::Value shape) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({name, shape});
}

void VarOp::inferShapes() { getResult().setType(getOperand(0).getType()); }

//===----------------------------------------------------------------------===//
// Conv2dOp

static void buildConv2dOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value data, mlir::Value channels, mlir::Value groups, 
                       mlir::Value kernel_size, mlir::Value strides, mlir::Value padding, 
                       mlir::Value data_layout, mlir::Value kernel_layout, mlir::Value name) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({data, channels, groups, kernel_size, strides, padding,
                        data_layout, kernel_layout, name});
}

void Conv2dOp::inferShapes() { getResult().setType(getOperand(0).getType()); }

//===----------------------------------------------------------------------===//
// MaxPool2dOp

static void buildMaxPool2dOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value data, mlir::Value pool_size, mlir::Value strides, mlir::Value padding) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({data, pool_size, strides, padding});
}

void MaxPool2dOp::inferShapes() { getResult().setType(getOperand(0).getType()); }

//===----------------------------------------------------------------------===//
// AvgPool2dOp

static void buildAvgPool2dOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value data, mlir::Value pool_size, mlir::Value strides, 
                       mlir::Value padding, mlir::Value count_include_pad) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({data, pool_size, strides, padding, count_include_pad});
}

void AvgPool2dOp::inferShapes() { getResult().setType(getOperand(0).getType()); }

//===----------------------------------------------------------------------===//
// GlobalAvgPool2dOp

static void buildGlobalAvgPool2dOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value data, mlir::Value layout) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({data, layout});
}

void GlobalAvgPool2dOp::inferShapes() { getResult().setType(getOperand(0).getType()); }

//===----------------------------------------------------------------------===//
// ConvKernelLayoutOp
// def conv_kernel_layout(data_layout, is_depthwise=False):
static void buildConvKernelLayoutOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value data_layout, mlir::Value is_depthwise) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({data_layout, is_depthwise});
}

void ConvKernelLayoutOp::inferShapes() { getResult().setType(getOperand(0).getType()); }

//===----------------------------------------------------------------------===//
// BatchNormOp
// data = layers.batch_norm_infer(data=data, epsilon=2e-5, scale=False, name='%ibn_data%i')
static void buildBatchNormOp(mlir::Builder *builder, mlir::OperationState &state,
                    mlir::Value data, mlir::Value epsilon, mlir::Value scale,
                    mlir::Value name) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({data, epsilon, scale, name});
}

void BatchNormOp::inferShapes() { getResult().setType(getOperand(0).getType()); }

//===----------------------------------------------------------------------===//
// BatchFlattenOp

static void buildBatchFlattenOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value data) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({data});
}

void BatchFlattenOp::inferShapes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// SoftmaxOp

static void buildSoftmaxOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value input) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({input});
}

void SoftmaxOp::inferShapes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// ReluOp

static void buildReluOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value input) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({input});
}

void ReluOp::inferShapes() { getResult().setType(getOperand().getType()); }

//===----------------------------------------------------------------------===//
// DenseOp

static void buildDenseOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value data, mlir::Value weight, mlir::Value units) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({data, weight, units});
}

void DenseOp::inferShapes() { getResult().setType(getOperand(0).getType()); }

//===----------------------------------------------------------------------===//
// BiasAddOp

static void buildBiasAddOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({lhs, rhs});
}

void BiasAddOp::inferShapes() { getResult().setType(getOperand(0).getType()); }

//===----------------------------------------------------------------------===//
// DenseBiasOp

static void buildDenseBiasOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value data, mlir::Value units) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({data, units});
}

void DenseBiasOp::inferShapes() { getResult().setType(getOperand(0).getType()); }

//===----------------------------------------------------------------------===//
// MakeTupleOp

static void buildMakeTupleOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({lhs, rhs});
}

void MakeTupleOp::inferShapes() { getResult().setType(getOperand(0).getType()); }

//===----------------------------------------------------------------------===//
// AppendOp

static void buildAppendOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value value) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({value});
}

//===----------------------------------------------------------------------===//
// ConcatenateOp

static void buildConcatenateOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({lhs, rhs});
}

void ConcatenateOp::inferShapes() { getResult().setType(getOperand(0).getType()); }

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES

#include "toy/Ops.cpp.inc"
