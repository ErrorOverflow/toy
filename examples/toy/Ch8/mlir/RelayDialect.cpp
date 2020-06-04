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

#include "toy/RelayDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::relay;

//===----------------------------------------------------------------------===//
// ToyInlinerInterface
//===----------------------------------------------------------------------===//

/// This class defines the interface for handling inlining with Toy
/// operations.
struct RelayInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    //===--------------------------------------------------------------------===//
    // Analysis Hooks
    //===--------------------------------------------------------------------===//

    /// All operations within toy can be inlined.
    bool isLegalToInline(Operation *, Region *,
                         BlockAndValueMapping &) const final {
        return true;
    }
};

//===----------------------------------------------------------------------===//
// ToyDialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
RelayDialect::RelayDialect(mlir::MLIRContext *ctx) : mlir::Dialect("relay", ctx) {
    addOperations<
#define GET_OP_LIST

#include "toy/RelayOps.cpp.inc"

    >();
    addInterfaces<RelayInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
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

static mlir::LogicalResult verify(ConstOp op) {
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

static void buildVarOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value name, mlir::Value shape) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({name, shape});
}

static void buildAddOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({lhs, rhs});
}

static void buildBinOp(mlir::Builder *builder, mlir::OperationState &state,
                       StringRef op, mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({lhs, rhs});
    state.addAttribute("op", builder->getStringAttr(op));
}

static void buildTupleOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({lhs, rhs});
}

static void buildConv2dOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value data, mlir::Value channels, mlir::Value groups, 
                       mlir::Value kernel_size, mlir::Value strides, mlir::Value padding, 
                       mlir::Value data_layout, mlir::Value kernel_layout, mlir::Value name) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({data, channels, groups, kernel_size, strides, padding,
                        data_layout, kernel_layout, name});
}

static void buildMaxPool2dOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value data, mlir::Value pool_size, mlir::Value strides, mlir::Value padding) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({data, pool_size, strides, padding});
}

static void buildAvgPool2dOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value data, mlir::Value pool_size, mlir::Value strides, 
                       mlir::Value padding, mlir::Value count_include_pad) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({data, pool_size, strides, padding, count_include_pad});
}

static void buildGlobalAvgPool2dOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value data, mlir::Value layout) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({data, layout});
}

static void buildConvKernelLayoutOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value data_layout, mlir::Value is_depthwise) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({data_layout, is_depthwise});
}

static void buildBatchNormOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value data, mlir::Value epsilon, mlir::Value scale, mlir::Value name) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({data, epsilon, scale, name});
}


static void buildDenseOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value data, mlir::Value weight, mlir::Value units) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({data, weight, units});
}

static void buildBiasAddOp(mlir::Builder *builder, mlir::OperationState &state,
                           mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({lhs, rhs});
}

static void buildDenseBiasOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value data, mlir::Value units) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({data, units});
}

static void buildBatchFlattenOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value data) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({data});
}

static void buildSoftmaxOp(mlir::Builder *builder,
                           mlir::OperationState &state, mlir::Value value) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands(value);
}

static void buildReluOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value input) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({input});
}

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

static void buildTransposeOp(mlir::Builder *builder,
                             mlir::OperationState &state, mlir::Value value) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands(value);
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

static void buildIndexOp(mlir::Builder *builder, mlir::OperationState &state,
                       StringRef name, mlir::Value index) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands(index);
    state.addAttribute("name", builder->getStringAttr(name));
}

static void buildMakeTupleOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({lhs, rhs});
}

static void buildAppendOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value value) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({value});
}

static void buildConcatenateOp(mlir::Builder *builder, mlir::OperationState &state,
                       mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder->getF64Type()));
    state.addOperands({lhs, rhs});
}


//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES

#include "toy/RelayOps.cpp.inc"
