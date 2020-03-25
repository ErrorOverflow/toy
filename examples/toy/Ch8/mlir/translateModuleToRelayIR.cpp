#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/FunctionSupport.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#include "toy/RelayIR.h"
#include "toy/RelayDialect.h"

#define N 100

int mlir::translateModuleToRelayIR(ModuleOp module) {
    return 0;
}

namespace {
    class RelayAPIPass : public mlir::FunctionPass<RelayAPIPass> {
    private:
        unsigned int num = 0;
        unsigned int tmp_num = 0;
        std::vector<mlir::Value> each_result;
        std::vector<std::string> each_name;
        std::string func_para_define;

        int Unary2Relay(mlir::Operation &op, std::string op_name, std::string convert_name){
            if(op.getName().getStringRef() != op_name)
                return 0;
            std::cout << "    tmp" << tmp_num << " = " << convert_name << "(";
            size_t len = each_result.size();
            size_t i;
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;
            if (i == len) std::cout << "error occured!\n";
            else std::cout << each_name[i] << ")\n";
            each_result.push_back(op.getResult(0));
            std::string tmp = "tmp" + std::to_string(tmp_num);
            each_name.push_back(tmp);
            tmp_num++;
            return 0;
        }

        int Binary2Relay(mlir::Operation &op, std::string op_name, std::string convert_name){
            if(op.getName().getStringRef() != op_name)
                return 0;
            std::cout << "    tmp" << tmp_num << " = " << convert_name << "(";
            size_t len = each_result.size();
            size_t i;
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(1)) break;
            std::cout<<each_name[i-1].c_str()<<", "<< each_name[i] <<")\n";
            each_result.push_back(op.getResult(0));
            std::string tmp = "tmp" + std::to_string(tmp_num);
            each_name.push_back(tmp);
            tmp_num++;    
            return 0;        
        };

        int Return2Relay(mlir::Operation &op){
            if(op.getName().getStringRef() != "toy.return")
                return 0;
            size_t i;
            size_t len = each_result.size();
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;
            std::string return_var = each_name[i];
            std::cout << "    return run_infer_type(relay.Function(relay.analysis.free_vars("
                << return_var << "),"<< return_var << "))\n";
            return 0;
        }

        int Constant2Relay(mlir::Operation &op){
            if (op.getName().getStringRef() != "relay.constant") 
                return 0;
            auto constantop = mlir::dyn_cast<mlir::relay::ConstantOp>(&op);
            auto constantValue = constantop.value();
            auto valueIt = constantValue.getValues<double>().begin();
            mlir::Operation *oop = constantop;
            auto tensorType = (*oop->result_type_begin()).cast<mlir::TensorType>();
            auto shape = tensorType.getShape();
            int dataNum = 1;
            std::vector<int> shape_vector;
            std::cout << "    var" << num <<" = relay.var(\"var"<< num <<"\",shape=(";
            for (size_t i = 0; i < shape.size(); i++) {
                if (i != shape.size() - 1) std::cout<< shape[i] << ",";
                else std::cout << shape[i] << "), dtype=\"float64\")\n";
                dataNum *= shape[i];
                shape_vector.push_back(shape[i]);
            }
            std::stringstream tmp_para_define;
            tmp_para_define << std::string("    data") << num << std::string(" = ");
            std::vector<double> data;
            for (int i = 0; i < dataNum; i++) {
                data.push_back(*valueIt++);
            }
            std::vector <std::string> result;
            for (int i = 0; i < dataNum; i++) {
                result.push_back(std::to_string(data[i]));
            }
            getDenseElement(result, shape_vector, 0);
            tmp_para_define << result[0] << "\n";
            func_para_define.append(tmp_para_define.str());
            each_result.push_back(oop->getResult(0));
            std::string tmp = "var" + std::to_string(num);
            each_name.push_back(tmp);
            num++;
            return 0;     
        }

        int Print2Relay(mlir::Operation &op){
            if (op.getName().getStringRef() != "relay.print")
                return 0;
            std::cout << "    f1 = relay.Function([";
            for (unsigned int i = 0; i < num; i++) {
                std::cout << "var" << i;
                if (i != num - 1) std::cout << ",";
                else std::cout<<"],";
            }
            size_t len = each_result.size();
            size_t i;
            for (i = 0; i < len; i++) {
                if (each_result[i] == op.getOperand(0)) break;
            }
            if (i == len) std::cout << "error occured!\n";
            else std::cout << each_name[i].c_str() << ")\n";
            std::cout << "    mod = relay.Module.from_expr(f1)\n"
                << "    mod = relay.transform.InferType()(mod)\n"
                << "    opt_level = 3\n"
                << "    target = tvm.target.cuda()\n"
                << "    with relay.build_config(opt_level=opt_level):\n"
                << "        graph, lib, params = relay.build_module.build(mod, target)\n"
                << "    ctx = tvm.gpu()\n"
                << "    module = graph_runtime.create(graph, lib, ctx)\n";
            for (unsigned int i = 0; i < num; i++) {
                std::cout << "    module.set_input(\"var"<< i << "\",data"<< i <<")\n";
            }
            std::cout << ("    module.run()\n")
                << "    out = module.get_output(0).asnumpy()\n"
                << "    print(out)\n";            
            return 0;
        }

        void getDenseElement(std::vector <std::string> &result, std::vector<int> shape, size_t num) {
            if (num == shape.size()) return;
            getDenseElement(result, shape, num + 1);
            std::vector <std::string> tmp;
            for (size_t i = 0; i < result.size() / shape[num] + 1; i++) {
                std::string str = "[";
                for (int j = 0; j < shape[num]; j++) {
                    int index = shape[num] * i + j;
                    if (j != shape[num] - 1) str = str + result[index] + ",";
                    else str = str + result[index] + "]";
                }
                tmp.push_back(str);
            }
            result.clear();
            result.assign(tmp.begin(), tmp.end());
        }

        void NetBuild(){
            unsigned int count = 0;
            std::cout << "net(";
            if(num == 0 ){
                for (mlir::Block &block : getFunction()) 
                    for (mlir::Operation &op : llvm::make_early_inc_range(block))
                        if (op.getName().getStringRef() == "relay.constant")   count++;
                for(unsigned int i = 0; i < count; i++) std::cout << "data" << i << ((i == count-1)? "" : ",") ;  
            }
            else for(unsigned int i = 0; i < num; i++) std::cout << "data" << i << ((i == num-1)? "" : ",");  
            std::cout << ")"; 
        }


    public:
        void runOnFunction() override {
            std::cout << ("from tvm import relay\nimport tvm\nimport numpy as np\nfrom tvm.contrib import graph_runtime\n");
            std::cout << "def ";
            NetBuild();
            std::cout << ":\n";
            for (mlir::Block &block : getFunction()) {
                for (mlir::Operation &op : llvm::make_early_inc_range(block)) {
                    Constant2Relay(op);
                    Unary2Relay(op, "relay.transpose", "relay.op.transpose");
                    Unary2Relay(op, "toy.reshape", "relay.op.reshape");
                    Unary2Relay(op, "relay.softmax", "relay.nn.softmax");
                    Print2Relay(op);
                    Return2Relay(op);
                    Binary2Relay(op, "relay.add", "relay.op.add");
                    Binary2Relay(op, "relay.mul", "relay.op.multiply");
                    Binary2Relay(op, "relay.conv1d", "relay.nn.conv1d");
                    Binary2Relay(op, "relay.dense", "relay.nn.dense");
                    Binary2Relay(op, "relay.bias_add", "relay.nn.bias_add");
                }
            }
            std::cout << "if __name__ == \"__main__\":\n";
            std::cout << func_para_define << "\n";
            std::cout << "\n    mod = relay.Module.from_expr(net)\n"
                << "    mod = relay.transform.InferType()(mod)\n"
                << "    opt_level = 3\n"
                << "    target = tvm.target.cuda()\n"
                << "    with relay.build_config(opt_level=opt_level):\n"
                << "        graph, lib, params = relay.build_module.build(mod, target)\n"
                << "    ctx = tvm.gpu()\n"
                << "    module = graph_runtime.create(graph, lib, ctx)\n";
            for (unsigned int i = 0; i < num; i++) {
                std::cout << "    module.set_input(\"var"<< i << "\",data"<< i <<")\n";
            }
            std::cout << ("    module.run()\n")
                << "    out = module.get_output(0).asnumpy()\n"
                << "    print(out)";
            std::cout << std::endl;
        }
    };
}

std::unique_ptr <mlir::Pass> mlir::relay::createRelayAPIPass() {
    return std::make_unique<RelayAPIPass>();
}

