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
#include "toy/Dialect.h"

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
        uint32_t tmp_num;
        uint32_t loop_flag = 0;
        uint32_t indent = 0;
        uint32_t *counter;
        bool is_loop_field = false;

        std::string func_name;
        std::string func_para_define;
        std::vector <mlir::Value> each_result;
        std::vector <std::string> while_end;
        std::vector <uint32_t> loop_round;

        unordered_map <uint32_t, std::string> &hashtable;

        void Op2Realy(mlir::Operation &op, std::string convert_name){
            std::stringstream tmp_expr;
            size_t len = each_result.size();
            size_t p;
            tmp_expr << getString(tmp_num) << " = " << convert_name << "(";
            for(uint32_t i = 0; i < op.getNumOperands(); i++){
                for (p = 0; p < len; p++) if (each_result[p] == op.getOperand(i)) break;
                tmp_expr << getString(p + *counter);
                if(i != op.getNumOperands()-1)
                    tmp_expr << ", ";
            }
            tmp_expr << ");\n";
            if (is_loop_field){
                while_end.push_back(tmp_expr.str());
                loop_flag--;
            }
            else {
                INDENT();
                std::cout << tmp_expr.str();
            }
            each_result.push_back(op.getResult(0));
            tmp_num++;
        }

        void Binary2Relay(mlir::Operation &op, std::string convert_name) {
            std::stringstream tmp_expr;
            tmp_expr << getString(tmp_num) << " = " << convert_name << "(";
            size_t len = each_result.size();
            size_t i;
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;
            tmp_expr << getString(i + *counter);
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(1)) break;
            tmp_expr << ", " << getString(i + *counter) << ")\n";
            each_result.push_back(op.getResult(0));
            tmp_num++;
            if (is_loop_field) {
                loop_flag--;
                while_end.push_back(tmp_expr.str());
            } else {
                INDENT();
                std::cout << tmp_expr.str();
            }
        };

        void String2Relay(mlir::Operation &op){
            auto string_op = mlir::dyn_cast<mlir::relay::StringOp>(&op);
            auto str = string_op.value().str();
            INDENT();
            cout << getString(tmp_num) << " = \"" << str << "\"\n";
            each_result.push_back(op.getResult(0));
            tmp_num++;          
        }

        void Bool2Relay(mlir::Operation &op){
            auto bool_op = mlir::dyn_cast<mlir::relay::BoolOp>(&op);
            auto str = bool_op.value().str();
            INDENT();
            cout << getString(tmp_num) << " = " << str << "\n";
            each_result.push_back(op.getResult(0));
            tmp_num++;        
        }

        void Bin2Relay(mlir::Operation &op){
            auto bin_op = mlir::dyn_cast<mlir::relay::BinOp>(&op);
            //std::cout << is_loop_field << " " << loop_flag << std::endl;
            if (!is_loop_field || (is_loop_field && loop_flag == 2)) {
                std::stringstream tmp_expr;
                tmp_expr << getString(tmp_num) << " = ";
                size_t len = each_result.size();
                size_t i;
                for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;
                tmp_expr << getString(i + *counter) << " ";
                tmp_expr << bin_op.op().str() << " ";
                for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(1)) break;
                tmp_expr << getString(i + *counter) << "\n";
                if(is_loop_field)
                    while_end.push_back(tmp_expr.str());
                else {
                    INDENT();
                    std::cout << tmp_expr.str();
                }
            } else {
                size_t len = each_result.size();
                size_t i;
                for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;
                std::cout << getString(i + *counter) << " ";
                std::cout << bin_op.op().str() << " ";
                for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(1)) break;
                std::cout << getString(i + *counter) << ":\n";                
            }
            if(is_loop_field)
                loop_flag--;
            each_result.push_back(op.getResult(0));
            tmp_num++;
        }

        void Call2Relay(mlir::Operation &op){
            size_t p;
            size_t len = each_result.size();
            auto call_op = mlir::dyn_cast<mlir::toy::GenericCallOp>(&op);
            INDENT();
            cout << getString(tmp_num) << " = func_" << call_op.callee().getRootReference().str() <<"(";
            for(uint32_t i = 0; i < op.getNumOperands(); i++){
                for (p = 0; p < len; p++) if (each_result[p] == op.getOperand(i)) break;
                cout << getString(p + *counter);
                if(i != op.getNumOperands()-1)
                    cout << ",";
            }
            cout << ");\n";
            each_result.push_back(op.getResult(0));
            tmp_num++;
        }

        void Return2Relay(mlir::Operation &op) {
            size_t i;
            size_t len = each_result.size();
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;
            std::string return_var = getString(i + *counter);
            INDENT();
            if(func_name == string("main"))
                std::cout << "return create_workload(relay.Function(relay.analysis.free_vars("
                        << return_var << ")," << return_var << ")\n";
            else cout << "return(" << return_var << ")\n";
        }

        void If2Relay(mlir::Operation &op) {
            is_loop_field = true;
            loop_flag = 1;
            INDENT();
            std::cout << "if ";
        }

        void For2Relay(mlir::Operation &op) {
            loop_flag = 3;
            is_loop_field = true;
            INDENT();
            std::cout << "while ";
        }

        void Constant2Relay(mlir::Operation &op) {
            auto constantop = mlir::dyn_cast<mlir::relay::ConstantOp>(&op);
            auto constantValue = constantop.value();
            auto valueIt = constantValue.getValues<double>().begin();
            mlir::Operation *oop = constantop;
            auto tensorType = (*oop->result_type_begin()).cast<mlir::TensorType>();
            auto shape = tensorType.getShape();
            uint32_t dataNum = 1;
            std::vector <uint32_t> shape_vector;
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
            for (uint32_t i = 0; i < dataNum; i++) {
                data.push_back(*valueIt++);
            }
            std::vector <std::string> result;
            for (uint32_t i = 0; i < dataNum; i++) {
                result.push_back(std::to_string(data[i]));
            }
            getDenseElement(result, tmp_para_define, '[', ']',shape_vector, 0, 0, result.size());
            func_para_define.append(tmp_para_define.str().append(";\n"));
            each_result.push_back(oop->getResult(0));
            num++;
            tmp_num++;
        }

        void Const2Relay(mlir::Operation &op) {
            auto constop = mlir::dyn_cast<mlir::relay::ConstOp>(&op);
            auto valueIt = constop.value().getValues<double>().begin();
            mlir::Operation *oop = constop;
            auto tensorType = (*oop->result_type_begin()).cast<mlir::TensorType>();
            auto shape = tensorType.getShape();
            uint32_t dataNum = 1;
            std::vector <uint32_t> shape_vector;
            INDENT();
            for (size_t i = 0; i < shape.size(); i++) {
                dataNum *= shape[i];
                shape_vector.push_back(shape[i]);
            }
            std::stringstream tmp_para_define;
            tmp_para_define << getString(tmp_num) << std::string(" = ");
            std::vector<double> data;
            for (uint32_t i = 0; i < dataNum; i++) {
                data.push_back(*valueIt++);
            }
            std::vector <std::string> result;
            for (uint32_t i = 0; i < dataNum; i++) {
                result.push_back(std::to_string(data[i]));
            }

            llvm::StringRef data_struct = constop.data_struct();
            char leftParen, rightParen;
            if(data_struct == "list"){
                leftParen = '[';
                rightParen = ']';
            } else if(data_struct == "tuple"){
                leftParen = '(';
                rightParen = ')';                
            }
            getDenseElement(result, tmp_para_define, leftParen, rightParen, shape_vector, 0, 0, result.size());
            cout << tmp_para_define.str() << ";\n";
            each_result.push_back(oop->getResult(0));
            tmp_num++;
        }

        void Var2Relay(mlir::Operation &op, std::string convert_name){
            INDENT();
            cout << getString(tmp_num) << " = ";
            size_t i;
            size_t len = each_result.size();
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;            
            cout << convert_name << "(name=" << getString(i + *counter) << ", ";
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(1)) break;            
            cout << "shape=" << getString(i + *counter) << ")\n";            
            each_result.push_back(op.getResult(0));
            tmp_num++;
        }

        void LoopField2Relay(mlir::Operation &op) {
            indent++;
            is_loop_field = false;
            loop_flag = 0;
        }

        void WhileEnd2Relay(mlir::Operation &op) {
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
            cout << name.str() << "[" << getString(i + *counter) << "];\n";
            each_result.push_back(op.getResult(0));
            tmp_num++;
        }

        void BatchNorm2Relay(mlir::Operation &op, std::string convert_name){
            std::stringstream tmp_expr;
            tmp_expr << getString(tmp_num) << " = " << convert_name << "(";
            size_t len = each_result.size();
            size_t i;
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;
            tmp_expr << "data = " << getString(i + *counter) << ", ";
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(1)) break;
            tmp_expr << "episilon = " << getString(i + *counter) << ", ";
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(2)) break;
            tmp_expr << "scale = " << getString(i + *counter) << ", ";

            tmp_expr << "name = ";
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(3)) break;     
            tmp_expr << getString(i + *counter) << "))\n";

            each_result.push_back(op.getResult(0));
            tmp_num++;
            if (is_loop_field) {
                while_end.push_back(tmp_expr.str());
            } else {
                INDENT();
                std::cout << tmp_expr.str();
            }
        }

        void Conv2Relay(mlir::Operation &op, std::string convert_name){
            std::stringstream tmp_expr;
            tmp_expr << getString(tmp_num) << " = " << convert_name << "(";
            size_t len = each_result.size();
            size_t i;
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;
            tmp_expr << "data = " << getString(i + *counter) << ", ";
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(1)) break;
            tmp_expr << "channels = " << getString(i + *counter) << ", ";
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(2)) break;
            tmp_expr << "groups = " << getString(i + *counter) << ", ";
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(3)) break;
            tmp_expr << "kernel_size = " << getString(i + *counter) << ", ";
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(4)) break;
            tmp_expr << "strides = " << getString(i + *counter) << ", ";
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(5)) break;
            tmp_expr << "padding = " << getString(i + *counter) << ", ";
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(6)) break;
            tmp_expr << "data_layout = " << getString(i + *counter) << ", ";
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(7)) break;
            tmp_expr << "kernel_layout = " << getString(i + *counter) << ", ";

            tmp_expr << "name = ";
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(8)) break;     
            tmp_expr << getString(i + *counter) <<")\n";

            each_result.push_back(op.getResult(0));
            tmp_num++;
            if (is_loop_field) {
                while_end.push_back(tmp_expr.str());
            } else {
                INDENT();
                std::cout << tmp_expr.str();
            }
        }

        void MaxPool2Relay(mlir::Operation &op, std::string convert_name){
            std::stringstream tmp_expr;
            tmp_expr << getString(tmp_num) << " = " << convert_name << "(";
            size_t len = each_result.size();
            size_t i;
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;
            tmp_expr << "data = " << getString(i + *counter) << ", ";
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(1)) break;
            tmp_expr << "pool_size = " << getString(i + *counter) << ", ";
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(2)) break;
            tmp_expr << "strides = " << getString(i + *counter) << ", ";
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(3)) break;
            tmp_expr << "padding = " << getString(i + *counter) << ")\n";

            each_result.push_back(op.getResult(0));
            tmp_num++;
            if (is_loop_field) {
                while_end.push_back(tmp_expr.str());
            } else {
                INDENT();
                std::cout << tmp_expr.str();
            }
        }

        void GlobalAvgPool2Relay(mlir::Operation &op, std::string convert_name){
            std::stringstream tmp_expr;
            tmp_expr << getString(tmp_num) << " = " << convert_name << "(";
            size_t len = each_result.size();
            size_t i;
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;
            tmp_expr << "data = " << getString(i + *counter) << ", ";
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(1)) break;
            tmp_expr << "layout = " << getString(i + *counter) << ")\n";

            each_result.push_back(op.getResult(0));
            tmp_num++;
            if (is_loop_field) {
                while_end.push_back(tmp_expr.str());
            } else {
                INDENT();
                std::cout << tmp_expr.str();
            }
        }

        void ConvKernelLayout2Relay(mlir::Operation &op, std::string convert_name){
            std::stringstream tmp_expr;
            tmp_expr << getString(tmp_num) << " = " << convert_name << "(";
            size_t len = each_result.size();
            size_t i;
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;
            tmp_expr << "data_layout = " << getString(i + *counter) << ", ";
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(1)) break;
            tmp_expr << "is_depthwise = " << getString(i + *counter) << ")\n";

            each_result.push_back(op.getResult(0));
            tmp_num++;
            if (is_loop_field) {
                while_end.push_back(tmp_expr.str());
            } else {
                INDENT();
                std::cout << tmp_expr.str();
            }
        }

        void BatchFlatten2Relay(mlir::Operation &op, std::string convert_name){
            std::stringstream tmp_expr;
            tmp_expr << getString(tmp_num) << " = " << convert_name << "(";
            size_t len = each_result.size();
            size_t i;
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;
            tmp_expr << "data = " << getString(i + *counter) << ")\n";
            
            each_result.push_back(op.getResult(0));
            tmp_num++;
            if (is_loop_field) {
                while_end.push_back(tmp_expr.str());
            } else {
                INDENT();
                std::cout << tmp_expr.str();
            }
        }

        void DenseBias2Relay(mlir::Operation &op, std::string convert_name){
            std::stringstream tmp_expr;
            tmp_expr << getString(tmp_num) << " = " << convert_name << "(";
            size_t len = each_result.size();
            size_t i;
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(0)) break;
            tmp_expr << "data = " << getString(i + *counter) << ", ";
            for (i = 0; i < len; i++) if (each_result[i] == op.getOperand(1)) break;
            tmp_expr << "units = " << getString(i + *counter) << ")\n";

            each_result.push_back(op.getResult(0));
            tmp_num++;
            if (is_loop_field) {
                while_end.push_back(tmp_expr.str());
            } else {
                INDENT();
                std::cout << tmp_expr.str();
            }
        }

        int getDenseElement(std::vector <std::string> &result, std::stringstream &out, 
                            char leftParen, char rightParen, std::vector <uint32_t> shape, 
                            size_t n, size_t low, size_t high) {
            if(shape.size() == 0){
                out << result[0];
                return 0;
            }
            if (n == shape.size() - 1) {
                out << leftParen;
                for (size_t i = low; i < high; i++) {
                    out << result[i];
                    if (i != high - 1) out << ",";
                }
                out << rightParen;
                return 0;
            }
            size_t size = (high - low) / shape[n];
            out << leftParen;
            for (uint32_t i = 0; i < shape[n]; i++) {
                getDenseElement(result, out, leftParen, rightParen, shape, 
                                n + 1, low + i * size, low + (i + 1) * size);
            }
            out << rightParen;
            return 0;
        }

        void FuncBuild() {
            string s = string("func_");
            s += func_name;
            std::cout << "def ";
            std::cout << s << "(";
            auto configs = getFunction().getArguments();
            for(auto arg : configs){
                each_result.push_back(arg);
                cout << getString(tmp_num++);
                if(arg != configs.back()) cout << ", ";
            }
            cout << "):\n";
        }

        void dumpWhileEnd(){
            INDENT();
            cout << while_end.back();
            while_end.pop_back();
        }

        void INDENT() {
            for(uint32_t i = 0; i<indent; i++){
                std::cout << "    ";
            }
        }

        void runOnFunctionInitial(){
            indent = 1;
            tmp_num = *counter;
            each_result.clear();
            func_name = getFunction().getName().str();
            //cout << tmp_num << endl;
            // if(getFunction().getName() == "main")
            //     for(auto i : hashtable){
            //         cout << i.first << " " << i.second << endl;
            //     }
        }

        std::string getString(uint32_t n){
            return (hashtable.find(n)->second);
        }

    public:
        RelayAPIPass(unordered_map <uint32_t, std::string> &hashtable, uint32_t *counter)
                 : counter(counter), hashtable(hashtable){}
        void runOnFunction() override {
            runOnFunctionInitial();
            FuncBuild();
            for (mlir::Block &block : getFunction()) {
                for (mlir::Operation &op : llvm::make_early_inc_range(block)) {
                    std::string op_name = op.getName().getStringRef().str();
                    //std::cout << "###" << op_name << "###" << std::endl;
                    if (op_name == "relay.constant")
                        Constant2Relay(op);
                    else if (op_name == "relay.const")
                        Const2Relay(op);
                    else if (op_name == "relay.variable")
                        Var2Relay(op, "relay.var");
                    else if (op_name == "relay.string")
                        String2Relay(op);
                    else if (op_name == "relay.bool")
                        Bool2Relay(op);
                    else if (op_name == "relay.softmax")
                        Op2Realy(op, "relay.nn.softmax");
                    else if (op_name == "relay.relu")
                        Op2Realy(op, "relay.nn.relu");
                    else if (op_name == "toy.return")
                        Return2Relay(op);
                    else if (op_name == "relay.if")
                        If2Relay(op);
                    else if (op_name == "relay.for")
                        For2Relay(op);
                    else if (op_name == "relay.add")
                        Binary2Relay(op, "relay.op.add");
                    else if (op_name == "relay.bin")
                        Bin2Relay(op);
                    else if (op_name == "relay.conv1d")
                        Op2Realy(op, "relay.nn.conv1d");
                    else if (op_name == "relay.dense")
                        Op2Realy(op, "relay.nn.dense");
                    else if (op_name == "relay.bias_add")
                        Op2Realy(op, "relay.nn.bias_add");
                    else if (op_name == "relay.loop_field")
                        LoopField2Relay(op);
                    else if (op_name == "relay.while_end")
                        WhileEnd2Relay(op);
                    else if (op_name == "relay.if_end")
                        indent--;
                    else if (op_name == "relay.index")
                        Index2Relay(op);
                    else if (op_name == "toy.generic_call")
                        Call2Relay(op);
                    else if (op_name == "relay.batch_norm")
                        BatchNorm2Relay(op, "layers.batch_norm_infer");
                    else if (op_name == "relay.conv2d")
                        Conv2Relay(op, "layers.conv2d");
                    else if (op_name == "relay.max_pool2d")
                        MaxPool2Relay(op, "relay.nn.max_pool2d");
                    else if (op_name == "relay.global_avg_pool2d")
                        GlobalAvgPool2Relay(op, "relay.nn.global_avg_pool2d");
                    else if (op_name == "relay.conv_kernel_layout")
                        ConvKernelLayout2Relay(op, "layers.conv_kernel_layout");
                    else if (op_name == "relay.batch_flatten")
                        BatchFlatten2Relay(op, "relay.nn.batch_flatten");
                    else if (op_name == "relay.dense_add_bias")
                        DenseBias2Relay(op, "layers.dense_add_bias");
                }
            }
            // if(getFunction().getName() == "main"){
            //     std::cout << "if __name__ == \"__main__\":\n";
            //     std::cout << func_para_define << "\n";
            //     std::cout << "\tfor target, ctx in ctx_list():\n"
            //             << "\t\tintrp = relay.create_executor(ctx=ctx, target=target)\n"
            //             << "\t\top_res, op_grad = intrp.evaluate(func_main())(\t\\\n\t\t\t";
            //     for (uint32_t i = 0; i < num; i++) {
            //         std::cout << "data" << i;
            //         if(i==num-1) cout << ")\n";
            //         else cout << ",";
            //     }
            //     std::cout << std::endl;
            // }
            cout << endl;
            *counter=tmp_num;
        }
    };
}

std::unique_ptr <mlir::Pass> mlir::relay::createRelayAPIPass(
            unordered_map <uint32_t, std::string> &hashtable, uint32_t *counter) {
    cout << "import tvm\n" 
            << "import numpy as np\n"
            << "from . import layers\n";    
    return std::make_unique<RelayAPIPass>(hashtable, counter);
}

