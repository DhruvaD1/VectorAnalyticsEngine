#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Sema/Sema.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/Analyses/Dominators.h"
#include "clang/Analysis/Analyses/PostDominators.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <numeric>
#include <regex>

using namespace clang;
using namespace llvm;

namespace {

struct DependenceVector {
    enum Type { Flow, Anti, Output, Input } type;
    int distance;
    const Expr *source;
    const Expr *sink;
    bool proven;
};

struct PolynomialExpr {
    std::unordered_map<std::string, int> coefficients;
    int constant;
    bool isAffine() const {
        return std::all_of(coefficients.begin(), coefficients.end(),
            [](const auto& pair) { return std::abs(pair.second) <= 1; });
    }
};

class AdvancedDependenceAnalyzer {
private:
    ASTContext &Context;
    std::unordered_map<std::string, PolynomialExpr> inductionVars;
    std::vector<DependenceVector> dependencies;

public:
    AdvancedDependenceAnalyzer(ASTContext &Ctx) : Context(Ctx) {}

    std::vector<DependenceVector> analyzeDependencies(const ForStmt *Loop) {
        dependencies.clear();
        buildInductionVariables(Loop);
        
        std::vector<ArraySubscriptExpr*> memoryAccesses;
        collectMemoryAccesses(Loop->getBody(), memoryAccesses);
        
        for (size_t i = 0; i < memoryAccesses.size(); ++i) {
            for (size_t j = i + 1; j < memoryAccesses.size(); ++j) {
                analyzePair(memoryAccesses[i], memoryAccesses[j]);
            }
        }
        
        return dependencies;
    }

private:
    void buildInductionVariables(const ForStmt *Loop) {
        if (const auto *DS = dyn_cast<DeclStmt>(Loop->getInit())) {
            if (DS->isSingleDecl()) {
                if (const auto *VD = dyn_cast<VarDecl>(DS->getSingleDecl())) {
                    std::string varName = VD->getNameAsString();
                    inductionVars[varName] = PolynomialExpr{{varName, 1}, 0};
                }
            }
        }
    }

    void collectMemoryAccesses(const Stmt *S, std::vector<ArraySubscriptExpr*> &accesses) {
        if (!S) return;
        
        if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(S)) {
            accesses.push_back(const_cast<ArraySubscriptExpr*>(ASE));
        }
        
        for (const auto *child : S->children()) {
            collectMemoryAccesses(child, accesses);
        }
    }

    PolynomialExpr analyzeExpression(const Expr *E) {
        if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
            std::string varName = DRE->getDecl()->getNameAsString();
            if (inductionVars.count(varName)) {
                return inductionVars[varName];
            }
        } else if (const auto *IL = dyn_cast<IntegerLiteral>(E)) {
            return PolynomialExpr{{}, (int)IL->getValue().getSExtValue()};
        } else if (const auto *BO = dyn_cast<BinaryOperator>(E)) {
            auto lhs = analyzeExpression(BO->getLHS());
            auto rhs = analyzeExpression(BO->getRHS());
            
            if (BO->getOpcode() == BO_Add) {
                PolynomialExpr result = lhs;
                for (const auto& pair : rhs.coefficients) {
                    result.coefficients[pair.first] += pair.second;
                }
                result.constant += rhs.constant;
                return result;
            } else if (BO->getOpcode() == BO_Sub) {
                PolynomialExpr result = lhs;
                for (const auto& pair : rhs.coefficients) {
                    result.coefficients[pair.first] -= pair.second;
                }
                result.constant -= rhs.constant;
                return result;
            }
        }
        
        return PolynomialExpr{{}, 0};
    }

    void analyzePair(ArraySubscriptExpr *access1, ArraySubscriptExpr *access2) {
        auto expr1 = analyzeExpression(access1->getIdx());
        auto expr2 = analyzeExpression(access2->getIdx());
        
        if (!expr1.isAffine() || !expr2.isAffine()) return;
        
        bool sameArray = false;
        if (const auto *DRE1 = dyn_cast<DeclRefExpr>(access1->getBase()->IgnoreImpCasts())) {
            if (const auto *DRE2 = dyn_cast<DeclRefExpr>(access2->getBase()->IgnoreImpCasts())) {
                sameArray = (DRE1->getDecl() == DRE2->getDecl());
            }
        }
        
        if (sameArray) {
            int distance = expr2.constant - expr1.constant;
            for (const auto& pair : expr1.coefficients) {
                distance += pair.second * (expr2.coefficients.count(pair.first) ? 
                    expr2.coefficients.at(pair.first) - pair.second : -pair.second);
            }
            
            DependenceVector::Type depType = DependenceVector::Flow;
            if (access1->isLValue() && access2->isLValue()) {
                depType = DependenceVector::Output;
            } else if (!access1->isLValue() && access2->isLValue()) {
                depType = DependenceVector::Anti;
            }
            
            dependencies.push_back({depType, distance, access1, access2, true});
        }
    }
};

class PolyhediralAnalyzer {
private:
    ASTContext &Context;
    
public:
    PolyhediralAnalyzer(ASTContext &Ctx) : Context(Ctx) {}
    
    struct ConstraintMatrix {
        std::vector<std::vector<int>> inequalities;
        std::vector<std::vector<int>> equalities;
        std::vector<std::string> variables;
    };
    
    ConstraintMatrix buildConstraintSystem(const ForStmt *Loop) {
        ConstraintMatrix matrix;
        
        if (const auto *DS = dyn_cast<DeclStmt>(Loop->getInit())) {
            if (DS->isSingleDecl()) {
                if (const auto *VD = dyn_cast<VarDecl>(DS->getSingleDecl())) {
                    matrix.variables.push_back(VD->getNameAsString());
                    
                    if (const auto *InitExpr = dyn_cast<IntegerLiteral>(VD->getInit())) {
                        int initVal = (int)InitExpr->getValue().getSExtValue();
                        matrix.inequalities.push_back({-1, initVal});
                    }
                }
            }
        }
        
        if (const auto *BO = dyn_cast<BinaryOperator>(Loop->getCond())) {
            if (BO->getOpcode() == BO_LT) {
                if (const auto *RHS = dyn_cast<IntegerLiteral>(BO->getRHS())) {
                    int bound = (int)RHS->getValue().getSExtValue();
                    matrix.inequalities.push_back({1, -bound + 1});
                }
            }
        }
        
        return matrix;
    }
    
    bool isLegalForVectorization(const ConstraintMatrix &matrix) {
        for (const auto& ineq : matrix.inequalities) {
            if (ineq.size() > 2) return false;
        }
        return true;
    }
};

class CostModel {
private:
    std::string targetArch;
    std::unordered_map<std::string, int> operationCosts;
    
public:
    CostModel(const std::string &arch = "x86_64") : targetArch(arch) {
        initializeCosts();
    }
    
    void initializeCosts() {
        if (targetArch == "x86_64") {
            operationCosts["add"] = 1;
            operationCosts["mul"] = 3;
            operationCosts["div"] = 20;
            operationCosts["load"] = 3;
            operationCosts["store"] = 1;
            operationCosts["vector_add"] = 1;
            operationCosts["vector_mul"] = 2;
            operationCosts["vector_load"] = 1;
            operationCosts["vector_store"] = 1;
        }
    }
    
    struct VectorizationBenefit {
        double speedup;
        int vectorWidth;
        int scalarCost;
        int vectorCost;
        bool profitable;
    };
    
    VectorizationBenefit estimateVectorization(const ForStmt *Loop, int tripCount = 1000) {
        int scalarCost = estimateScalarCost(Loop->getBody());
        
        VectorizationBenefit benefit;
        benefit.vectorWidth = getOptimalVectorWidth();
        benefit.vectorCost = estimateVectorCost(Loop->getBody(), benefit.vectorWidth);
        benefit.scalarCost = scalarCost * tripCount;
        
        int vectorIterations = tripCount / benefit.vectorWidth;
        int remainder = tripCount % benefit.vectorWidth;
        
        int totalVectorCost = vectorIterations * benefit.vectorCost + remainder * scalarCost;
        
        benefit.speedup = (double)benefit.scalarCost / totalVectorCost;
        benefit.profitable = benefit.speedup > 1.2;
        
        return benefit;
    }
    
private:
    int getOptimalVectorWidth() {
        if (targetArch == "x86_64") return 8;
        return 4;
    }
    
    int estimateScalarCost(const Stmt *S) {
        if (!S) return 0;
        
        int cost = 0;
        if (const auto *BO = dyn_cast<BinaryOperator>(S)) {
            std::string op = BO->getOpcodeStr().str();
            if (op == "+") cost += operationCosts["add"];
            else if (op == "*") cost += operationCosts["mul"];
            else if (op == "/") cost += operationCosts["div"];
        } else if (isa<ArraySubscriptExpr>(S)) {
            cost += operationCosts["load"];
        }
        
        for (const auto *child : S->children()) {
            cost += estimateScalarCost(child);
        }
        
        return cost;
    }
    
    int estimateVectorCost(const Stmt *S, int vectorWidth) {
        int scalarCost = estimateScalarCost(S);
        return (scalarCost * vectorWidth) / vectorWidth + 2;
    }
};

class AdvancedLoopTransformer {
private:
    ASTContext &Context;
    Rewriter &R;
    
public:
    AdvancedLoopTransformer(ASTContext &Ctx, Rewriter &Rewriter) 
        : Context(Ctx), R(Rewriter) {}
    
    bool transformLoop(ForStmt *Loop, const CostModel::VectorizationBenefit &benefit) {
        if (!benefit.profitable) return false;
        
        SourceLocation StartLoc = Loop->getBeginLoc();
        
        std::string pragma = "#pragma clang loop vectorize(enable) vectorize_width(" + 
                           std::to_string(benefit.vectorWidth) + ") interleave_count(2)\n";
        
        R.InsertText(StartLoc, pragma);
        
        if (benefit.vectorWidth >= 8) {
            std::string annotation = "[[clang::vectorize]] ";
            R.InsertText(Loop->getForLoc(), annotation);
        }
        
        return true;
    }
    
    bool unrollLoop(ForStmt *Loop, int factor = 4) {
        const Stmt *Body = Loop->getBody();
        if (!Body) return false;
        
        std::string bodyText;
        llvm::raw_string_ostream OS(bodyText);
        
        for (int i = 0; i < factor; ++i) {
            std::string unrolledBody = generateUnrolledIteration(Body, i);
            OS << unrolledBody << "\n";
        }
        
        SourceRange bodyRange = Body->getSourceRange();
        R.ReplaceText(bodyRange, "{\n" + OS.str() + "}");
        
        return true;
    }
    
private:
    std::string generateUnrolledIteration(const Stmt *Body, int offset) {
        std::string result;
        llvm::raw_string_ostream OS(result);
        Body->printPretty(OS, nullptr, Context.getPrintingPolicy());
        
        std::string bodyStr = OS.str();
        std::regex iPattern("\\[i\\]");
        std::string replacement = "[i + " + std::to_string(offset) + "]";
        return std::regex_replace(bodyStr, iPattern, replacement);
    }
};

class InterproceduralAnalyzer {
private:
    ASTContext &Context;
    std::unordered_map<const FunctionDecl*, std::unordered_set<std::string>> sideEffects;
    std::unordered_map<const FunctionDecl*, bool> analyzed;
    
public:
    InterproceduralAnalyzer(ASTContext &Ctx) : Context(Ctx) {}
    
    bool hasSideEffects(const FunctionDecl *FD) {
        if (analyzed.count(FD)) {
            return !sideEffects[FD].empty();
        }
        
        analyzed[FD] = true;
        if (!FD->hasBody()) {
            sideEffects[FD].insert("external_call");
            return true;
        }
        
        analyzeFunctionBody(FD, FD->getBody());
        return !sideEffects[FD].empty();
    }
    
private:
    void analyzeFunctionBody(const FunctionDecl *FD, const Stmt *S) {
        if (!S) return;
        
        if (const auto *CE = dyn_cast<CallExpr>(S)) {
            if (const auto *Callee = CE->getDirectCallee()) {
                if (Callee->getNameAsString() == "printf" || 
                    Callee->getNameAsString() == "malloc") {
                    sideEffects[FD].insert("io_or_allocation");
                } else if (hasSideEffects(Callee)) {
                    sideEffects[FD].insert("indirect_side_effect");
                }
            }
        }
        
        for (const auto *child : S->children()) {
            analyzeFunctionBody(FD, child);
        }
    }
};

class AdvancedVectorizabilityAnalyzer {
private:
    ASTContext &Context;
    DiagnosticsEngine &Diags;
    Rewriter &R;
    AdvancedDependenceAnalyzer DepAnalyzer;
    PolyhediralAnalyzer PolyAnalyzer;
    CostModel CostEstimator;
    AdvancedLoopTransformer Transformer;
    InterproceduralAnalyzer IPAnalyzer;
    
    unsigned DiagIDAdvanced;
    unsigned DiagIDCost;
    unsigned DiagIDTransform;

public:
    AdvancedVectorizabilityAnalyzer(ASTContext &Ctx, DiagnosticsEngine &D, Rewriter &Rewriter)
        : Context(Ctx), Diags(D), R(Rewriter), DepAnalyzer(Ctx), PolyAnalyzer(Ctx),
          Transformer(Ctx, Rewriter), IPAnalyzer(Ctx) {
        
        DiagIDAdvanced = Diags.getCustomDiagID(DiagnosticsEngine::Warning,
            "advanced vectorization analysis: %0");
        DiagIDCost = Diags.getCustomDiagID(DiagnosticsEngine::Note,
            "cost model: %0 (speedup: %1x, profitable: %2)");
        DiagIDTransform = Diags.getCustomDiagID(DiagnosticsEngine::Note,
            "applied transformation: %0");
    }
    
    void analyzeAdvanced(ForStmt *Loop) {
        auto dependencies = DepAnalyzer.analyzeDependencies(Loop);
        auto constraints = PolyAnalyzer.buildConstraintSystem(Loop);
        auto benefit = CostEstimator.estimateVectorization(Loop);
        
        if (!dependencies.empty()) {
            bool hasFlowDependence = std::any_of(dependencies.begin(), dependencies.end(),
                [](const DependenceVector& dep) { 
                    return dep.type == DependenceVector::Flow && dep.distance > 0; 
                });
                
            if (hasFlowDependence) {
                emitAdvancedDiagnostic(Loop, "flow dependence prevents vectorization");
                return;
            }
        }
        
        if (!PolyAnalyzer.isLegalForVectorization(constraints)) {
            emitAdvancedDiagnostic(Loop, "polyhedral analysis shows non-affine bounds");
            return;
        }
        
        emitCostAnalysis(Loop, benefit);
        
        if (benefit.profitable) {
            if (Transformer.transformLoop(Loop, benefit)) {
                emitTransformation(Loop, "vectorization hints and pragmas");
            }
            
            if (benefit.speedup > 2.0) {
                if (Transformer.unrollLoop(Loop, 4)) {
                    emitTransformation(Loop, "4x loop unrolling");
                }
            }
        }
        
        analyzeInterproceduralEffects(Loop);
    }
    
private:
    void emitAdvancedDiagnostic(const ForStmt *Loop, const std::string &reason) {
        SourceLocation Loc = Loop->getForLoc();
        Diags.Report(Loc, DiagIDAdvanced) << reason;
    }
    
    void emitCostAnalysis(const ForStmt *Loop, const CostModel::VectorizationBenefit &benefit) {
        SourceLocation Loc = Loop->getForLoc();
        Diags.Report(Loc, DiagIDCost) 
            << ("scalar_cost=" + std::to_string(benefit.scalarCost) + 
                " vector_cost=" + std::to_string(benefit.vectorCost))
            << llvm::format("%.2f", benefit.speedup)
            << (benefit.profitable ? "yes" : "no");
    }
    
    void emitTransformation(const ForStmt *Loop, const std::string &transform) {
        SourceLocation Loc = Loop->getForLoc();
        Diags.Report(Loc, DiagIDTransform) << transform;
    }
    
    void analyzeInterproceduralEffects(const ForStmt *Loop) {
        class CallVisitor : public RecursiveASTVisitor<CallVisitor> {
        public:
            InterproceduralAnalyzer &Analyzer;
            std::vector<const FunctionDecl*> CalledFunctions;
            
            CallVisitor(InterproceduralAnalyzer &A) : Analyzer(A) {}
            
            bool VisitCallExpr(CallExpr *CE) {
                if (const auto *FD = CE->getDirectCallee()) {
                    CalledFunctions.push_back(FD);
                }
                return true;
            }
        };
        
        CallVisitor visitor(IPAnalyzer);
        visitor.TraverseStmt(const_cast<Stmt*>(Loop->getBody()));
        
        for (const auto *FD : visitor.CalledFunctions) {
            if (IPAnalyzer.hasSideEffects(FD)) {
                emitAdvancedDiagnostic(Loop, "interprocedural side effects detected in " + 
                                     FD->getNameAsString());
            }
        }
    }
};

class AdvancedAutoVecVisitor : public RecursiveASTVisitor<AdvancedAutoVecVisitor> {
private:
    ASTContext &Context;
    Rewriter &R;
    AdvancedVectorizabilityAnalyzer Analyzer;

public:
    AdvancedAutoVecVisitor(ASTContext &Ctx, Rewriter &Rewriter) 
        : Context(Ctx), R(Rewriter), Analyzer(Ctx, Ctx.getDiagnostics(), Rewriter) {}

    bool VisitForStmt(ForStmt *FS) {
        Analyzer.analyzeAdvanced(FS);
        return true;
    }
};

class AdvancedAutoVecConsumer : public ASTConsumer {
private:
    AdvancedAutoVecVisitor Visitor;
    Rewriter R;

public:
    AdvancedAutoVecConsumer(ASTContext &Context, StringRef InFile) 
        : Visitor(Context, R) {
        R.setSourceMgr(Context.getSourceManager(), Context.getLangOpts());
    }

    void HandleTranslationUnit(ASTContext &Context) override {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
        
        const RewriteBuffer *RewriteBuf = R.getRewriteBufferFor(Context.getSourceManager().getMainFileID());
        if (RewriteBuf) {
            std::string TransformedCode = std::string(RewriteBuf->begin(), RewriteBuf->end());
            llvm::outs() << "=== TRANSFORMED CODE ===\n";
            llvm::outs() << TransformedCode << "\n";
            llvm::outs() << "========================\n";
        }
    }
};

class AdvancedAutoVecAction : public PluginASTAction {
protected:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
        return std::make_unique<AdvancedAutoVecConsumer>(CI.getASTContext(), InFile);
    }

    bool ParseArgs(const CompilerInstance &CI, const std::vector<std::string> &args) override {
        for (const std::string &arg : args) {
            if (arg == "-help") {
                llvm::errs() << "Advanced Auto Vectorizer Plugin Options:\n";
                llvm::errs() << "  -cost-model: Enable cost model analysis\n";
                llvm::errs() << "  -transform: Enable automatic transformations\n";
                llvm::errs() << "  -polyhedral: Enable polyhedral analysis\n";
                return false;
            }
        }
        return true;
    }
};

}

static FrontendPluginRegistry::Add<AdvancedAutoVecAction>
X("advanced-auto-vec", "Advanced C++ Auto Vectorizer with Polyhedral Analysis and Transformations");
