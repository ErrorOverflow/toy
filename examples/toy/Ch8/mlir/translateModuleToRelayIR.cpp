#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>

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
        int num = 0;
        int tmp_num = 0;
        std::vector <mlir::Value> each_result;
        std::vector <std::string> each_name;

    public:
        int Unary2Relay(mlir::Operation &op, std::string op_name){
            if(op.getName().getStringRef() != op_name)
                return 0;
            std::cout << "    tmp" << tmp_num << " = " << op_name << "(";
            size_t len = each_result.size();
            size_t i;
            for (i = 0; i < len; i++) {
                if (each_result[i] == op.getOperand(0)) break;
            }
            if (i == len) printf("error occured!\n");
            else printf("%s)\n", each_name[i].c_str());
            each_result.push_back(op.getResult(0));
            std::string tmp = "tmp" + std::to_string(tmp_num);
            each_name.push_back(tmp);
            tmp_num++;
            return 0;
        }

        int Binary2Relay(mlir::Operation &op, std::string op_name){
            if(op.getName().getStringRef() != op_name)
                return 0;
            std::cout << "    tmp" << tmp_num << " = " << op_name << "(";
            size_t len = each_result.size();
            size_t i;
            for (i = 0; i < len; i++) {
                if (each_result[i] == op.getOperand(1)) break;
            }
            std::cout<<each_name[i-1].c_str()<<", "<<each_name[i].c_str()<<")\n";
            each_result.push_back(op.getResult(0));
            std::string tmp = "tmp" + std::to_string(tmp_num);
            each_name.push_back(tmp);
            tmp_num++;    
            return 0;        
        };

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
            printf("    var%d = relay.var(\"var%d\",shape=(", num, num);
            for (size_t i = 0; i < shape.size(); i++) {
                if (i != shape.size() - 1) std::cout<< shape[i] << ",";
                else std::cout << shape[i] << "),dtype=\"float64\")\n";
                dataNum *= shape[i];
                shape_vector.push_back(shape[i]);
            }
            printf("    data%d = ", num);
            std::vector<double> data;
            for (int i = 0; i < dataNum; i++) {
                data.push_back(*valueIt++);
            }
            std::vector <std::string> result;
            for (int i = 0; i < dataNum; i++) {
                result.push_back(std::to_string(data[i]));
            }
            getDenseElement(result, shape_vector, 0);
            std::cout << result[0] << std::endl;
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
            for (int i = 0; i < num; i++) {
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
            for (int i = 0; i < num; i++) {
                std::cout << "    module.set_input(\"var"<< i << "\",data"<< i <<")\n";
            }
            std::cout << ("    module.run()\n")
                << "    out = module.get_output(0).asnumpy()\n"
                << "    print(out)\n";            
            return 0;
        }

        void runOnFunction() override {
            std::cout << ("from tvm import relay\nimport tvm\nimport numpy as np\nfrom tvm.contrib import graph_runtime\n");
            //std::cout << "def net():\n";
            std::cout << "if __name__ == \"__main__\":\n";
            for (mlir::Block &block : getFunction()) {
                for (mlir::Operation &op : llvm::make_early_inc_range(block)) {
                    Constant2Relay(op);
                    Unary2Relay(op, "relay.transpose");
                    Unary2Relay(op, "toy.reshape");
                    Unary2Relay(op, "relay.softmax");
                    Print2Relay(op);
                    Binary2Relay(op, "relay.add");
                    Binary2Relay(op, "relay.mul");
                    Binary2Relay(op, "relay.conv1d");
                    Binary2Relay(op, "relay.dense");
                    Binary2Relay(op, "relay.bias_add");
                }
            }
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
    };
}


std::unique_ptr <mlir::Pass> mlir::relay::createRelayAPIPass() {
    return std::make_unique<RelayAPIPass>();
}

