//===- Ops.td - relay dialect operation definitions ----------*- tablegen -*-===//
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
// Defines the operations of the relay dialect.
//
//===----------------------------------------------------------------------===//

#ifdef RELAY_OPS
#else
#define RELAY_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

// Provide a definition of the 'relay' dialect in the ODS framework so that we
// can define our operations.
def Relay_Dialect : Dialect {
  let name = "relay";
  let cppNamespace = "relay";
}

// Base class for relay dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Relay_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Relay_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// relay Operations
//===----------------------------------------------------------------------===//

// We define a relay operation by inherting from our base 'Relay_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'NoSideEffect' as it is a pure operation
// and may be removed if dead.
def ConstantOp : Relay_Op<"constant", [NoSideEffect]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documenatation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "relay.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<"Builder *builder, OperationState &result, "
              "DenseElementsAttr value", [{
      build(builder, result, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<"Builder *builder, OperationState &result, double value", [{
      buildConstantOp(builder, result, value);
    }]>
  ];

  // Invoke a static verify method to verify this constant operation.
  let verifier = [{ return ::verify(*this); }];
}

def ConstOp : Relay_Op<"const", [NoSideEffect]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documenatation of the operations within our dialect.
  let summary = "const";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "relay.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins StrAttr:$data_struct, F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<"Builder *builder, OperationState &result, "
              "StringRef data_struct, DenseElementsAttr value", [{
      build(builder, result, value.getType(), data_struct, value);
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<"Builder *builder, OperationState &result, StringRef data_struct, double value", [{
      buildConstOp(builder, result, data_struct, value);
    }]>
  ];

  // Invoke a static verify method to verify this constant operation.
  let verifier = [{ return ::verify(*this); }];
}

def VarOp : Relay_Op<"variable", [NoSideEffect]> {
    let summary = "variable operation";

    let arguments = (ins F64Tensor:$name, F64Tensor:$shape);
    let results = (outs F64Tensor);

    let builders = [
        OpBuilder<"Builder *b, OperationState &result, Value name, Value shape", [{
            buildVarOp(b, result, name, shape);
        }]>
    ];
}

def LoopFieldOp : Relay_Op<"loop_field", [NoSideEffect]> {
  let summary = "loop_field";
}

def LoopEndOp : Relay_Op<"loop_end", [NoSideEffect]> {
  let summary = "loop_end";
}

def AddOp : Relay_Op<"add"> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<"Builder *b, OperationState &result, Value lhs, Value rhs", [{
      buildAddOp(b, result, lhs, rhs);
    }]
  >];
}

def BinOp : Relay_Op<"bin", [NoSideEffect]> {
  let arguments = (ins StrAttr:$op, F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<"Builder *b, OperationState &result, "
              "StringRef op, Value lhs, Value rhs", [{
      buildBinOp(b, result, op, lhs, rhs);
    }]>
  ];
}

def Conv2dOp : Relay_Op<"conv2d", [NoSideEffect]> {
  let summary = "conv2d operation";

  let arguments = (ins F64Tensor:$data, F64Tensor:$channels, F64Tensor:$groups,
                    F64Tensor:$kernel_size, F64Tensor:$strides, F64Tensor:$padding,
                    F64Tensor:$data_layout, F64Tensor:$kernel_layout, F64Tensor:$x, F64Tensor:$y);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<"Builder *b, OperationState &result, Value data, Value channels,"
                "Value groups, Value kernel_size, Value strides, Value padding,"
                "Value data_layout, Value kernel_layout, Value x, Value y", [{
      buildConv2dOp(b, result, data, channels, groups, kernel_size, strides,
                     padding, data_layout, kernel_layout, x, y);
    }]
    >,
    OpBuilder<"Builder *b, OperationState &result, std::vector<Value> &operands", [{
      buildConv2dOp(b, result, operands[0], operands[1], operands[2], operands[3], operands[4],
                     operands[5], operands[6], operands[7], operands[8], operands[9]);
    }]
    >    
  ];
}

def MaxPool2dOp : Relay_Op<"max_pool2d", [NoSideEffect]> {
  let summary = "max_pool2d operation";

  let arguments = (ins F64Tensor:$data, F64Tensor:$pool_size, F64Tensor:$strides,
                    F64Tensor:$padding);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<"Builder *b, OperationState &result, Value data, Value pool_size,"
                "Value strides, Value padding", [{
      buildMaxPool2dOp(b, result, data, pool_size, strides, padding);
    }]
    >,
    OpBuilder<"Builder *b, OperationState &result, std::vector<Value> &operands", [{
      buildMaxPool2dOp(b, result, operands[0], operands[1], operands[2], operands[3]);
    }]
    >
  ];
}

def GlobalAvgPool2dOp : Relay_Op<"global_avg_pool2d", [NoSideEffect]> {
  let summary = "global_avg_pool2d operation";

  let arguments = (ins F64Tensor:$data, F64Tensor:$layout);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<"Builder *b, OperationState &result, Value data, Value layout", [{
      buildGlobalAvgPool2dOp(b, result, data, layout);
    }]
    >
  ];
}

def BatchNormOp : Relay_Op<"bacth_norm", [NoSideEffect]> {
  let summary = "bacth_norm operation";

  let arguments = (ins F64Tensor:$data, F64Tensor:$epsilon, F64Tensor:$scale, F64Tensor:$x, F64Tensor:$y);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<"Builder *b, OperationState &result, Value data, Value epsilon,"
                "Value scale, Value x, Value y", [{
      buildBatchNormOp(b, result, data, epsilon, scale, x, y);
    }]
    >,
    OpBuilder<"Builder *b, OperationState &result, std::vector<Value> &operands", [{
      buildBatchNormOp(b, result, operands[0], operands[1], operands[2], operands[3], operands[4]);
    }]
    >
  ];
}

def ConvKernelLayoutOp : Relay_Op<"conv_kernel_layout", [NoSideEffect]> {
  let summary = "conv_kernel_layout operation";

  let arguments = (ins F64Tensor:$data_layout, F64Tensor:$is_depthwise);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<"Builder *b, OperationState &result,"
                "Value data_layout, Value is_depthwise", [{
      buildConvKernelLayoutOp(b, result, data_layout, is_depthwise);
    }]
    >,
    OpBuilder<"Builder *b, OperationState &result, std::vector<Value> &operands", [{
      buildConvKernelLayoutOp(b, result, operands[0], operands[1]);
    }]
    >
  ];
}

def BatchFlattenOp : Relay_Op<"batch_flatten", [NoSideEffect]> {
  let arguments = (ins F64Tensor:$data);
  let results = (outs F64Tensor);
  let builders = [
    OpBuilder<"Builder *b, OperationState &result, Value data", [{
      buildBatchFlattenOp(b, result, data);
    }]
  >];
}

def PrintOp : Relay_Op<"print"> {
  let summary = "print operation";

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  // Allow building a TransposeOp with from the two input operands.
  let builders = [
    OpBuilder<"Builder *b, OperationState &result, Value input", [{
      buildPrintOp(b, result, input);
    }]
  >];

  // Invoke a static verify method to verify this transpose operation.
  let verifier = [{ return ::verify(*this); }];
}

def ReshapeOp : Relay_Op<"reshape"> {
  let summary = "tensor reshape operation";
  let description = [{
    Reshape operation is transforming its input tensor into a new tensor with
    the same number of elements but different shapes. For example:

    ```mlir
       %0 = "relay.reshape"(%arg1) : (tensor<10xf64>) -> tensor<5x2xf64>
    ```
  }];

  let arguments = (ins F64Tensor:$input);

  // We expect that the reshape operation returns a statically shaped tensor.
  let results = (outs StaticShapeTensorOf<[F64]>);
}

def ReturnOp : Relay_Op<"return", [Terminator, HasParent<"FuncOp">]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      func @foo() -> tensor<2xf64> {
        ...
        relay.return %0 : tensor<2xf64>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<F64Tensor>:$input);

  // Allow building a ReturnOp with no return operand.
  let builders = [OpBuilder<
    "Builder *b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Invoke a static verify method to verify this return operation.
  let verifier = [{ return ::verify(*this); }];
}

def TransposeOp : Relay_Op<"transpose"> {
  let summary = "transpose operation";

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  // Allow building a TransposeOp with from the two input operands.
  let builders = [
    OpBuilder<"Builder *b, OperationState &result, Value input", [{
      buildTransposeOp(b, result, input);
    }]
  >];

  // Invoke a static verify method to verify this transpose operation.
  let verifier = [{ return ::verify(*this); }];
}

def SoftmaxOp : Relay_Op<"softmax"> {
  let summary = "softmax operation";

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  // Allow building a TransposeOp with from the two input operands.
  let builders = [
    OpBuilder<"Builder *b, OperationState &result, Value input", [{
      buildSoftmaxOp(b, result, input);
    }]
  >];
}

def DenseOp : Relay_Op<"dense"> {
  let summary = "dense operation";
  let description = [{
    The "dense" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<"Builder *b, OperationState &result, Value lhs, Value rhs", [{
      buildDenseOp(b, result, lhs, rhs);
    }]
  >];
}

def BiasAddOp : Relay_Op<"bias_add"> {
  let summary = "bias_add operation";
  let description = [{
    The "bias_add" operation performs element-wise addition between two tensors.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<"Builder *b, OperationState &result, Value lhs, Value rhs", [{
      buildBiasAddOp(b, result, lhs, rhs);
    }]
  >];
}

def DenseBiasOp : Relay_Op<"dense_add_bias"> {
  let summary = "dense_add_bias operation";
  let description = [{
    The "dense_add_bias" operation performs element-wise addition between two tensors.
  }];

  let arguments = (ins F64Tensor:$data, F64Tensor:$units);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<"Builder *b, OperationState &result, Value data, Value units", [{
      buildDenseBiasOp(b, result, data, units);
    }]
  >];
}

def IfOp : Relay_Op<"if", [NoSideEffect]>{
    let summary = "if branch";
    let description = [{
        The "if" branch is envolved with two tensor and one operation.
    }];
}

def ForOp : Relay_Op<"for", [NoSideEffect]>{
    let summary = "for branch";
    let description = [{
        The "for" branch is envolved with two tensor and one operation.
    }];
}

def IndexOp : Relay_Op<"index", [NoSideEffect]> {

  let arguments = (ins StrAttr:$name, F64Tensor:$index);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<"Builder *b, OperationState &result, "
              "StringRef name, Value index", [{
      buildIndexOp(b, result, name, index);
    }]>
  ];
}
#endif // Relay_OpS
