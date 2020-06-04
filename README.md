# Multi-Level Intermediate Representation

See [https://mlir.llvm.org/](https://mlir.llvm.org/) for more information.

cmake --build . --target check-mlir
./toyc-ch8 /home/wml/llvm-project-master/llvm-project/mlir/inception.toy -emit=relay

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \

cmake --build . --target check-mlir

toyc-ch2 /home/wml/llvm-project-master/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo

echo 'def main() { print([[1, 2], [3, 4]]); }' | ./toyc-ch6 -emit=jit

./toyc-ch6 /home/wml/桌面/codegen.toy -emit=jit

./toyc-ch8 /home/wml/桌面/codegen.toy -emit=jit

./toyc-ch6 /home/wml/llvm-project-master/llvm-project/mlir/test/Examples/Toy/Ch6/codegen.toy -emit=jit

./toyc-ch8 /home/wml/llvm-project-master/llvm-project/mlir/test/Examples/Toy/Ch6/codegen.toy -emit=relay
