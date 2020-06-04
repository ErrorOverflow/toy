# TOY2Relay 文档

### 1.如何构建项目

首先需要在Linux下构建LLVM，我参考的是https://github.com/llvm/llvm-project 的代码和 https://mlir.llvm.org/getting_started/ 的构建方法。当然还要感谢孙明利学长的前期工作 https://github.com/MingliSun/MLIR-TVM 。

如果你已经完成了LLVM的构建并且已经完成了官方文档对toy语言的教学，接着把本项目的项目名“toy”重命名为“mlir”，覆盖复制进LLVM的一级目录，这是因为mlir是一个开发活跃的项目，新老版本大概率不兼容。接着可以把本项目中的tvm文件夹剪切出去，它是用来做tvm下的网络测试，和mlir部分无关。

下一步是编译我们自己修改的mlir，mlir/example下面包含toy的所有源代码文件。如果你之前自己亲手编译了LLVM项目，那么应该指导LLVM项目编译的build文件夹，去那里在终端中输入

```
cmake --build . --target check-mlir
```

即可只编译mlir项目来节约编译时间，toy编译器的生成文件会在build/bin内，你可以在这里执行toy编译。

### 2.如何执行toy编译

在1.中提到的build/bin内，终端输入

```
./toyc-ch8 /path to LLVM/mlir/inception.toy -emit=relay
```

可以完成对一个toy语言源文件inception.toy编译，记得替换成自己的LLVM项目绝对路径名，最后一个emit是编译选项，支持多种编译输出：

```
-emit=ast //输出AST 
-emit=mlir //输出toy dialect 
-emit=mlir-relay //输出relay dialect 
-emit=relay //输出python-relay代码   
```

输出文件会在/path to LLVM/mlir/examples/toy/out.py，写毕业设计的时候没有用相对路径，这个部分需要你自己去translateModuleToRelayIR.cpp中修改（ε=ε=ε=┏(゜ロ゜;)┛。当然out.py里面应该是有一个我生成的inception_v3网络的代码可以先将就着用。

### 3.如何在TVM中运行编译器的生成代码

在1.我们提到的那个和mlir无关的文件夹这就派上用场了，里面一级目录有tutorial和python.tvm.relay.testing两个文件夹，如果你对TVM项目足够熟悉的话应该指导前者是TVM的教材文件，后者是TVM的测试文件（包括了用relay层编写的不同深度学习网络的样例），我在前者中的relay_test_net.py中会调用后者对应的网络文件（比如inception_v3_2.py）来执行网络。

考虑到TVM也是一个开发活跃的项目，我也建议把我的tutorial和python.tvm.relay.testing文件夹覆盖复制进你clone下来的TVM项目中。如果你觉得python.tvm.relay.testing中的代码和史老师那边开发文档中不一致的话，还是以python.tvm.relay.testing为准。

### 4.开发中的一些坑

4.1 SSA问题我处理得比较武断，这在我的论文中有描述，显然用phi指令的话代码会更优美。

4.2 有一个重载函数runOnFunction，你得注意它是并行执行的，这是mlir本身设计的，一个module中的多个function不会从上到下的一个一个编译，它们会同时执行，这在输出relay代码的时候影响很大，我的解决办法是每个function生成一个文件，再将这些文件合并为一个out.py。
