//===- AST.h - Node definition for the Toy AST ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST for the Toy language. It is optimized for
// simplicity, not efficiency. The AST forms a tree structure where each node
// references its children using std::unique_ptr<>.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_TOY_AST_H_
#define MLIR_TUTORIAL_TOY_AST_H_

#include "toy/Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <vector>

namespace toy {

/// A variable type with shape information.
    struct VarType {
        std::vector <int64_t> shape;
    };

/// Base class for all expression nodes.
    class ExprAST {
    public:
        enum ExprASTKind {
            Expr_VarDecl,
            Expr_Return,
            Expr_Num,
            Expr_Literal,
            Expr_Var,
            Expr_BinOp,
            Expr_Call,
            Expr_Print,
            Expr_IfOp,
            Expr_ForOp,
            Expr_Exe,
            Expr_Const,
            Expr_Tuple,
            Expr_Break,
            Expr_Index,
            Expr_Bool,
            Expr_String
        };

        ExprAST(ExprASTKind kind, Location location)
                : kind(kind), location(location) {}

        virtual ~ExprAST() = default;

        ExprASTKind getKind() const { return kind; }

        const Location &loc() { return location; }

    private:
        const ExprASTKind kind;
        Location location;
    };

/// A block-list of expressions.
    using ExprASTList = std::vector <std::unique_ptr<ExprAST>>;

/// Expression class for numeric literals like "1.0".
    class NumberExprAST : public ExprAST {
        double Val;

    public:
        NumberExprAST(Location loc, double val) : ExprAST(Expr_Num, loc), Val(val) {}

        double getValue() { return Val; }

        /// LLVM style RTTI
        static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }
    };

/// Expression class for a literal value.
    class LiteralExprAST : public ExprAST {
        std::vector <std::unique_ptr<ExprAST>> values;
        std::vector <int64_t> dims;

    public:
        LiteralExprAST(Location loc, std::vector <std::unique_ptr<ExprAST>> values,
                       std::vector <int64_t> dims)
                : ExprAST(Expr_Literal, loc), values(std::move(values)),
                  dims(std::move(dims)) {}

        llvm::ArrayRef <std::unique_ptr<ExprAST>> getValues() { return values; }

        llvm::ArrayRef <int64_t> getDims() { return dims; }

        /// LLVM style RTTI
        static bool classof(const ExprAST *c) { return c->getKind() == Expr_Literal; }
    };

/// Expression class for referencing a variable, like "a".
    class VariableExprAST : public ExprAST {
        std::string name;

    public:
        VariableExprAST(Location loc, llvm::StringRef name)
                : ExprAST(Expr_Var, loc), name(name) {}

        llvm::StringRef getName() { return name; }

        /// LLVM style RTTI
        static bool classof(const ExprAST *c) { return c->getKind() == Expr_Var; }
    };

/// Expression class for defining a variable.
    class VarDeclExprAST : public ExprAST {
        std::string name;
        VarType type;
        std::unique_ptr <ExprAST> initVal;

    public:
        VarDeclExprAST(Location loc, llvm::StringRef name, VarType type,
                       std::unique_ptr <ExprAST> initVal)
                : ExprAST(Expr_VarDecl, loc), name(name), type(std::move(type)),
                  initVal(std::move(initVal)) {}

        llvm::StringRef getName() { return name; }

        ExprAST *getInitVal() { return initVal.get(); }

        const VarType &getType() { return type; }

        /// LLVM style RTTI
        static bool classof(const ExprAST *c) { return c->getKind() == Expr_VarDecl; }
    };

/// Expression class for defining a variable.
    class ConstExprAST : public ExprAST {
        std::string name;
        VarType type;
        std::unique_ptr <ExprAST> initVal;

    public:
        ConstExprAST(Location loc, llvm::StringRef name, VarType type,
                       std::unique_ptr <ExprAST> initVal)
                : ExprAST(Expr_Const, loc), name(name), type(std::move(type)),
                  initVal(std::move(initVal)) {}

        llvm::StringRef getName() { return name; }

        ExprAST *getInitVal() { return initVal.get(); }

        const VarType &getType() { return type; }

        /// LLVM style RTTI
        static bool classof(const ExprAST *c) { return c->getKind() == Expr_Const; }
    };

    class TupleExprAST : public ExprAST {
        std::vector <std::unique_ptr<ExprAST>> values;
        std::vector <int64_t> dims;

    public:
        TupleExprAST(Location loc, std::vector <std::unique_ptr<ExprAST>> values,
                       std::vector <int64_t> dims)
                : ExprAST(Expr_Tuple, loc), values(std::move(values)),
                  dims(std::move(dims)) {}

        llvm::ArrayRef <std::unique_ptr<ExprAST>> getValues() { return values; }

        llvm::ArrayRef <int64_t> getDims() { return dims; }

        /// LLVM style RTTI
        static bool classof(const ExprAST *c) { return c->getKind() == Expr_Tuple; }
    };    

/// Expression class for execution.
    class ExeExprAST : public ExprAST {
        std::string lhs;
        std::unique_ptr <ExprAST> rhs;

    public:
        ExeExprAST(Location loc, llvm::StringRef lhs,
                   std::unique_ptr <ExprAST> rhs)
                : ExprAST(Expr_Exe, loc), lhs(lhs),
                  rhs(std::move(rhs)) {}

        llvm::StringRef getLHS() { return lhs; }

        ExprAST *getRHS() { return rhs.get(); }

        /// LLVM style RTTI
        static bool classof(const ExprAST *c) { return c->getKind() == Expr_Exe; }
    };


/// Expression class for a return operator.
    class ReturnExprAST : public ExprAST {
        llvm::Optional <std::unique_ptr<ExprAST>> expr;

    public:
        ReturnExprAST(Location loc, llvm::Optional <std::unique_ptr<ExprAST>> expr)
                : ExprAST(Expr_Return, loc), expr(std::move(expr)) {}

        llvm::Optional<ExprAST *> getExpr() {
            if (expr.hasValue())
                return expr->get();
            return llvm::None;
        }

        /// LLVM style RTTI
        static bool classof(const ExprAST *c) { return c->getKind() == Expr_Return; }
    };

/// Expression class for a binary operator.
    class BinaryExprAST : public ExprAST {
        std::string op;
        std::unique_ptr <ExprAST> lhs, rhs;

    public:
        llvm::StringRef getOp() { return op; }

        ExprAST *getLHS() { return lhs.get(); }

        ExprAST *getRHS() { return rhs.get(); }

        BinaryExprAST(Location loc, llvm::StringRef Op, std::unique_ptr <ExprAST> lhs,
                      std::unique_ptr <ExprAST> rhs)
                : ExprAST(Expr_BinOp, loc), op(Op), lhs(std::move(lhs)),
                  rhs(std::move(rhs)) {}

        /// LLVM style RTTI
        static bool classof(const ExprAST *c) { return c->getKind() == Expr_BinOp; }
    };

/// Expression class for function calls.
    class CallExprAST : public ExprAST {
        std::string callee;
        std::vector <std::unique_ptr<ExprAST>> args;

    public:
        CallExprAST(Location loc, const std::string &callee,
                    std::vector <std::unique_ptr<ExprAST>> args)
                : ExprAST(Expr_Call, loc), callee(callee), args(std::move(args)) {}

        llvm::StringRef getCallee() { return callee; }

        llvm::ArrayRef <std::unique_ptr<ExprAST>> getArgs() { return args; }

        /// LLVM style RTTI
        static bool classof(const ExprAST *c) { return c->getKind() == Expr_Call; }
    };

/// Expression class for builtin print calls.
    class PrintExprAST : public ExprAST {
        std::unique_ptr <ExprAST> arg;

    public:
        PrintExprAST(Location loc, std::unique_ptr <ExprAST> arg)
                : ExprAST(Expr_Print, loc), arg(std::move(arg)) {}

        ExprAST *getArg() { return arg.get(); }

        /// LLVM style RTTI
        static bool classof(const ExprAST *c) { return c->getKind() == Expr_Print; }
    };

/// Expression class for index.
    class IndexExprAST : public ExprAST {
        std::string name;
        std::unique_ptr <ExprAST> index;

    public:
        IndexExprAST(Location loc, std::string &name, std::unique_ptr <ExprAST> index)
                : ExprAST(Expr_Index, loc), name(name), index(std::move(index)) {}

        llvm::StringRef getName() { return name; }

        ExprAST *getIndex() { return index.get(); }

        /// LLVM style RTTI
        static bool classof(const ExprAST *c) { return c->getKind() == Expr_Index; }
    };

/// Expression class for string.
    class StringExprAST : public ExprAST {
        std::string name;
        std::string str;

    public:
        StringExprAST(Location loc, llvm::StringRef name, llvm::StringRef str)
                : ExprAST(Expr_String, loc), name(name), str(str) {}

        llvm::StringRef getName() { return name; }

        llvm::StringRef getStr() { return str; }

        /// LLVM style RTTI
        static bool classof(const ExprAST *c) { return c->getKind() == Expr_String; }
    };


/// Expression class for bool type.
    class BoolExprAST : public ExprAST {
        std::string name;
        std::string value;

    public:
        BoolExprAST(Location loc, llvm::StringRef name, llvm::StringRef value)
                : ExprAST(Expr_Bool, loc), name(name), value(value) {}

        llvm::StringRef getName() { return name; }

        llvm::StringRef getValue() { return value; }

        /// LLVM style RTTI
        static bool classof(const ExprAST *c) { return c->getKind() == Expr_Bool; }
    };

/// Expression class for a if operator.
    class IfExprAST : public ExprAST {
        Location location;
        std::unique_ptr <ExprAST> value;
        std::unique_ptr <ExprASTList> body;

    public:
        ExprAST *getValue() { return value.get(); }

        ExprASTList *getBody() { return body.get(); }

        uint32_t getBodyNum() { return body.get()->size(); }

        IfExprAST(Location loc, std::unique_ptr <ExprAST> value,
                  std::unique_ptr <ExprASTList> body) : ExprAST(Expr_IfOp, loc), value(std::move(value)),
                                                        body(std::move(body)) {}

        static bool classof(const ExprAST *c) { return c->getKind() == Expr_IfOp; }
    };

/// Expression class for a 'for' operator.
    class ForExprAST : public ExprAST {
        Location location;
        std::unique_ptr <ConstExprAST> decl;
        std::unique_ptr <ExprAST> value;
        std::unique_ptr <ExeExprAST> expr;
        std::unique_ptr <ExprASTList> body;

    public:
        ConstExprAST *getDecl() { return decl.get(); }

        ExprAST *getValue() { return value.get(); }

        ExeExprAST *getExpr() { return expr.get(); }

        ExprASTList *getBody() { return body.get(); }

        uint32_t getBodyNum() { return body.get()->size(); }

        ForExprAST(Location loc, std::unique_ptr <ConstExprAST> decl,
                   std::unique_ptr <ExprAST> value, std::unique_ptr <ExeExprAST> expr,
                   std::unique_ptr <ExprASTList> body) : ExprAST(Expr_ForOp, loc), decl(std::move(decl)),
                                                         value(std::move(value)), expr(std::move(expr)),
                                                         body(std::move(body)) {}

        static bool classof(const ExprAST *c) { return c->getKind() == Expr_ForOp; }
    };

/// This class represents the "prototype" for a function, which captures its
/// name, and its argument names (thus implicitly the number of arguments the
/// function takes).
    class PrototypeAST {
        Location location;
        std::string name;
        std::vector <std::unique_ptr<VariableExprAST>> args;

    public:
        PrototypeAST(Location location, const std::string &name,
                     std::vector <std::unique_ptr<VariableExprAST>> args)
                : location(location), name(name), args(std::move(args)) {}

        const Location &loc() { return location; }

        llvm::StringRef getName() const { return name; }

        llvm::ArrayRef <std::unique_ptr<VariableExprAST>> getArgs() { return args; }
    };

/// This class represents a function definition itself.
    class FunctionAST {
        std::unique_ptr <PrototypeAST> proto;
        std::unique_ptr <ExprASTList> body;

    public:
        FunctionAST(std::unique_ptr <PrototypeAST> proto,
                    std::unique_ptr <ExprASTList> body)
                : proto(std::move(proto)), body(std::move(body)) {}

        PrototypeAST *getProto() { return proto.get(); }

        ExprASTList *getBody() { return body.get(); }
    };

/// This class represents a list of functions to be processed together
    class ModuleAST {
        std::vector <FunctionAST> functions;

    public:
        ModuleAST(std::vector <FunctionAST> functions)
                : functions(std::move(functions)) {}

        auto begin() -> decltype(functions.begin()) { return functions.begin(); }

        auto end() -> decltype(functions.end()) { return functions.end(); }
    };

    void dump(ModuleAST &);

} // namespace toy

#endif // MLIR_TUTORIAL_TOY_AST_H_
