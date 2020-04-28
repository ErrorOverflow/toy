#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>

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
#include "llvm/ADT/ScopedHashTable.h"

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#include "toy/RelayIR.h"
#include "toy/RelayDialect.h"

using std::unordered_map;
using std::string;
using std::cout;
using std::endl;

int32_t mlir::translateModuleToRelayIR(ModuleOp module) {
    return 0;
}

namespace {
    class RelayAPIPass : public mlir::FunctionPass<RelayAPIPass> {
    private:
        uint32_t num = 0;
        uint32_t tmp_num = 0;
        uint32_t loop_flag = 0;
        uint32_t indent = 0;
        bool is_loop_field = false;
        std::vector <mlir::Value> each_result;
        std::vector <std::string> while_end;
        std::vector <uint32_t> loop_round;
        std::string func_para_define;
        unordered_map <uint32_t, std::string> &hashtable;
        //unordered_map <uint32_t, std::string> const_table;

        void Unary2Relay(mlir::Operation &op, std::string convert_name) {
            INDENT();
            std::cout << getString(tmp_num) << " = " << convert_name << "(";
            size_t len = each_result.size();
            size_t i;
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;
            if (i == len) std::cout << "error occured!\n";
            else std::cout << getString(i) << ")\n";
            each_result.push_back(op.getResult(0));
            tmp_num++;
        }

        void Reshape2Relay(mlir::Operation &op, std::string convert_name) {
            INDENT();
            std::cout << "tmp" << tmp_num << " = " << convert_name << "(";
            size_t len = each_result.size();
            size_t i;
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;
            if (i == len) std::cout << "error occured!\n";
            else std::cout << getString(i) << ")\n";
            each_result.push_back(op.getResult(0));
            tmp_num++;
        }

        void Binary2Relay(mlir::Operation &op, std::string convert_name) {
            std::stringstream tmp_expr;
            tmp_expr << getString(tmp_num) << " = " << convert_name << "(";
            size_t len = each_result.size();
            size_t i;
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;
            tmp_expr << getString(i);
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(1)) break;
            tmp_expr << ", " << getString(i) << ")\n";
            each_result.push_back(op.getResult(0));
            tmp_num++;
            if (is_loop_field) {
                while_end.push_back(tmp_expr.str());
            } else {
                INDENT();
                std::cout << tmp_expr.str();
            }
        };

        void Return2Relay(mlir::Operation &op) {
            size_t i;
            size_t len = each_result.size();
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;
            std::string return_var = getString(i);
            INDENT();
            std::cout << "return run_infer_type(relay.Function(relay.analysis.free_vars("
                      << return_var << ")," << return_var << "))\n";
        }

        void Bltz2Relay(mlir::Operation &op) {
            size_t i;
            size_t len = each_result.size();
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;
            cout << "(" << getString(i);
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(1)) break;
            cout << " < " << getString(i) << "):\n";
            each_result.push_back(op.getResult(0));
            tmp_num++;
        }

        void If2Relay(mlir::Operation &op) {
            // TODO:
            is_loop_field = true;
            loop_flag = 1;
            INDENT();
            std::cout << "if";
        }

        void For2Relay(mlir::Operation &op) {
            // TODO:
            INDENT();
            loop_flag = 2;
            is_loop_field = true;
            std::cout << "while";
        }

        void Constant2Relay(mlir::Operation &op) {
            auto constantop = mlir::dyn_cast<mlir::relay::ConstantOp>(&op);
            auto constantValue = constantop.value();
            auto valueIt = constantValue.getValues<double>().begin();
            mlir::Operation *oop = constantop;
            auto tensorType = (*oop->result_type_begin()).cast<mlir::TensorType>();
            auto shape = tensorType.getShape();
            int32_t dataNum = 1;
            std::vector <int32_t> shape_vector;
            INDENT();
            std::cout << "var" << num << " = relay.var(\"var" << num << "\",shape=(";
            for (size_t i = 0; i < shape.size(); i++) {
                if (i != shape.size() - 1) std::cout << shape[i] << ",";
                else std::cout << shape[i] << "), dtype=\"float64\")\n";
                dataNum *= shape[i];
                shape_vector.push_back(shape[i]);
            }
            std::stringstream tmp_para_define;
            tmp_para_define << std::string("\tdata") << num << std::string(" = ");
            std::vector<double> data;
            for (int32_t i = 0; i < dataNum; i++) {
                data.push_back(*valueIt++);
            }
            std::vector <std::string> result;
            for (int32_t i = 0; i < dataNum; i++) {
                result.push_back(std::to_string(data[i]));
            }
            getDenseElement(result, tmp_para_define, shape_vector, 0, 0, result.size());
            func_para_define.append(tmp_para_define.str().append(";\n"));
            each_result.push_back(oop->getResult(0));
            num++;
            tmp_num++;
        }

        void Const2Relay(mlir::Operation &op) {
            auto constantop = mlir::dyn_cast<mlir::relay::ConstOp>(&op);
            auto constantValue = constantop.value();
            auto valueIt = constantValue.getValues<double>().begin();
            mlir::Operation *oop = constantop;
            auto tensorType = (*oop->result_type_begin()).cast<mlir::TensorType>();
            auto shape = tensorType.getShape();
            int32_t dataNum = 1;
            std::vector <int32_t> shape_vector;
            INDENT();
            for (size_t i = 0; i < shape.size(); i++) {
                dataNum *= shape[i];
                shape_vector.push_back(shape[i]);
            }
            std::stringstream tmp_para_define;
            tmp_para_define << getString(tmp_num) << std::string(" = ");
            std::vector<double> data;
            for (int32_t i = 0; i < dataNum; i++) {
                data.push_back(*valueIt++);
            }
            std::vector <std::string> result;
            for (int32_t i = 0; i < dataNum; i++) {
                result.push_back(std::to_string(data[i]));
            }
            getDenseElement(result, tmp_para_define, shape_vector, 0, 0, result.size());
            cout << tmp_para_define.str() << ";\n";
            each_result.push_back(oop->getResult(0));
            tmp_num++;
        }

        void LoopField2Relay(mlir::Operation &op) {
            indent++;
            is_loop_field = false;
        }

        void LoopEnd2Relay(mlir::Operation &op) {
            dumpWhileEnd();
            indent--;
        }

        void Index2Relay(mlir::Operation &op){
            auto indexop = mlir::dyn_cast<mlir::relay::IndexOp>(&op);
            auto name = indexop.name();
            INDENT();
            cout << getString(tmp_num) << " = ";
            size_t i;
            size_t len = each_result.size();
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;            
            cout << name.str() << "[" << getString(i) << "]\n";
            each_result.push_back(op.getResult(0));
            tmp_num++;
        }

        int getDenseElement(std::vector <std::string> &result, std::stringstream &out, std::vector <int32_t> shape,
                             size_t n, size_t low, size_t high) {
            if(shape.size() == 0){
                out << result[0];
                return 0;
            }
            if (n == shape.size() - 1) {
                out << "[";
                for (size_t i = low; i < high; i++) {
                    //std::cout<< i<<std::endl;
                    out << result[i];
                    if (i != high - 1) out << ",";
                }
                out << "]";
                return 0;
            }
            size_t size = (high - low) / shape[n];
            out << "[";
            for (int32_t i = 0; i < shape[n]; i++) {
                getDenseElement(result, out, shape, n + 1, low + i * size, low + (i + 1) * size);
            }
            out << "]";
            return 0;
        }

        void NetBuild(string origin_name) {
            string s = string("func_");
            s += origin_name;
            std::cout << "def ";
            std::cout << s << "():\n";
        }

        void dumpWhileEnd(){
            INDENT();
            cout << while_end.back();
            while_end.pop_back();
        }

        void INDENT() {
            for(uint32_t i = 0; i<indent; i++){
                std::cout << "\t";
            }
        }

        std::string getString(uint32_t n){
            return hashtable.find(n)->second;
        }

    public:
        RelayAPIPass(unordered_map <uint32_t, std::string> &hashtable) : hashtable(hashtable){}
        void runOnFunction() override {
            indent = 1;
            for(auto i : hashtable){
                 cout << i.first << " " << i.second << endl;
            }
            NetBuild(getFunction().getName().str());
            for (mlir::Block &block : getFunction()) {
                for (mlir::Operation &op : llvm::make_early_inc_range(block)) {
                    std::string op_name = op.getName().getStringRef().str();
                    //std::cout << "###" << op_name << "###" << std::endl;
                    if (op_name == "relay.constant")
                        Constant2Relay(op);
                    else if (op_name == "relay.const")
                        Const2Relay(op);
                    else if (op_name == "relay.reshape")
                        Reshape2Relay(op, "relay.op.reshape");
                    else if (op_name == "relay.softmax")
                        Unary2Relay(op, "relay.nn.softmax");
                    else if (op_name == "relay.reshape")
                        Unary2Relay(op, "np.reshape");
                    else if (op_name == "toy.return")
                        Return2Relay(op);
                    else if (op_name == "relay.if")
                        If2Relay(op);
                    else if (op_name == "relay.for")
                        For2Relay(op);
                    else if (op_name == "relay.add")
                        Binary2Relay(op, "relay.op.add");
                    else if (op_name == "relay.mul")
                        Binary2Relay(op, "relay.op.multiply");
                    else if (op_name == "relay.conv1d")
                        Binary2Relay(op, "relay.nn.conv1d");
                    else if (op_name == "relay.dense")
                        Binary2Relay(op, "relay.nn.dense");
                    else if (op_name == "relay.bias_add")
                        Binary2Relay(op, "relay.nn.bias_add");
                    else if (op_name == "relay.bltz")
                        Bltz2Relay(op);
                    else if (op_name == "relay.loop_field")
                        LoopField2Relay(op);
                    else if (op_name == "relay.loop_end")
                        LoopEnd2Relay(op);
                    else if (op_name == "relay.index")
                        Index2Relay(op);
                }
            }
            if(getFunction().getName() == "main"){
                std::cout << "if __name__ == \"__main__\":\n";
                std::cout << func_para_define << "\n";
                std::cout << "\tfor target, ctx in ctx_list():\n"
                        << "\t\tintrp = relay.create_executor(ctx=ctx, target=target)\n"
                        << "\t\top_res, op_grad = intrp.evaluate(func_main())(\t\\\n\t\t\t";
                for (uint32_t i = 0; i < num; i++) {
                    std::cout << "data" << i;
                    if(i==num-1) cout << ")\n";
                    else cout << ",";
                }
                std::cout << std::endl;
            }
            cout << endl;
        }
    };
}

std::unique_ptr <mlir::Pass> mlir::relay::createRelayAPIPass(unordered_map <uint32_t, std::string> &hashtable) {
    cout << "from tvm import relay\n"
            << "import tvm\nimport numpy as np\n"
            << "from tvm.contrib import graph_runtime\n";    
    return std::make_unique<RelayAPIPass>(hashtable);
}

