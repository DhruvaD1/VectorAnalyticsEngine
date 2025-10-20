#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Sema/Sema.h"
#include "clang/Analysis/Analyses/Dominators.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace llvm;

namespace {

enum class VectorizationBlocker {
  None,
  NonCanonicalLoop,
  NonLinearIndex,
  MemoryAlias,
  UnpredictableBranch,
  EarlyExit,
  SideEffects,
  VolatileAccess,
  NonAffineBounds
};

class LoopVectorizabilityAnalyzer {
private:
  ASTContext &Context;
  DiagnosticsEngine &Diags;
  unsigned DiagIDNonVectorizable;
  unsigned DiagIDNote;

public:
  LoopVectorizabilityAnalyzer(ASTContext &Ctx, DiagnosticsEngine &D) 
    : Context(Ctx), Diags(D) {
    DiagIDNonVectorizable = Diags.getCustomDiagID(DiagnosticsEngine::Warning,
        "loop not vectorizable: %0");
    DiagIDNote = Diags.getCustomDiagID(DiagnosticsEngine::Note, "%0");
  }

  void analyzeLoop(const ForStmt *Loop) {
    VectorizationBlocker blocker = checkVectorizability(Loop);
    
    if (blocker != VectorizationBlocker::None) {
      emitDiagnostic(Loop, blocker);
    }
  }

private:
  VectorizationBlocker checkVectorizability(const ForStmt *Loop) {
    if (!isCanonicalLoop(Loop)) {
      return VectorizationBlocker::NonCanonicalLoop;
    }

    if (hasNonLinearIndexing(Loop)) {
      return VectorizationBlocker::NonLinearIndex;
    }

    if (hasMemoryAliasing(Loop)) {
      return VectorizationBlocker::MemoryAlias;
    }

    if (hasUnpredictableBranches(Loop)) {
      return VectorizationBlocker::UnpredictableBranch;
    }

    if (hasEarlyExits(Loop)) {
      return VectorizationBlocker::EarlyExit;
    }

    if (hasSideEffects(Loop)) {
      return VectorizationBlocker::SideEffects;
    }

    return VectorizationBlocker::None;
  }

  bool isCanonicalLoop(const ForStmt *Loop) {
    const Stmt *Init = Loop->getInit();
    const Expr *Cond = Loop->getCond();
    const Expr *Inc = Loop->getInc();

    if (!Init || !Cond || !Inc) return false;

    if (const auto *DS = dyn_cast<DeclStmt>(Init)) {
      if (DS->isSingleDecl()) {
        if (const auto *VD = dyn_cast<VarDecl>(DS->getSingleDecl())) {
          if (!VD->getType()->isIntegerType()) return false;
          if (!VD->hasInit()) return false;
          
          if (const auto *InitExpr = dyn_cast<IntegerLiteral>(VD->getInit())) {
            if (InitExpr->getValue() != 0) return false;
          } else {
            return false;
          }
        }
      }
    } else {
      return false;
    }

    if (const auto *BO = dyn_cast<BinaryOperator>(Cond)) {
      if (BO->getOpcode() != BO_LT && BO->getOpcode() != BO_LE) return false;
      
      const auto *LHS = dyn_cast<DeclRefExpr>(BO->getLHS());
      if (!LHS) return false;
      
      const auto *VD = dyn_cast<VarDecl>(LHS->getDecl());
      if (!VD) return false;
      
      if (const auto *DS = dyn_cast<DeclStmt>(Init)) {
        if (DS->isSingleDecl() && DS->getSingleDecl() != VD) {
          return false;
        }
      }
    } else {
      return false;
    }

    if (const auto *UO = dyn_cast<UnaryOperator>(Inc)) {
      if (UO->getOpcode() != UO_PreInc && UO->getOpcode() != UO_PostInc) {
        return false;
      }
    } else if (const auto *BO = dyn_cast<BinaryOperator>(Inc)) {
      if (BO->getOpcode() != BO_AddAssign) return false;
      if (const auto *RHS = dyn_cast<IntegerLiteral>(BO->getRHS())) {
        if (RHS->getValue() != 1) return false;
      } else {
        return false;
      }
    } else {
      return false;
    }

    return true;
  }

  bool hasNonLinearIndexing(const ForStmt *Loop) {
    class IndexingVisitor : public RecursiveASTVisitor<IndexingVisitor> {
    public:
      bool NonLinearFound = false;
      std::string InductionVar;

      IndexingVisitor(std::string IV) : InductionVar(IV) {}

      bool VisitArraySubscriptExpr(ArraySubscriptExpr *ASE) {
        if (const auto *DRE = dyn_cast<DeclRefExpr>(ASE->getIdx())) {
          if (DRE->getDecl()->getNameAsString() == InductionVar) {
            return true;
          }
        } else if (isa<CallExpr>(ASE->getIdx()) || 
                   isa<BinaryOperator>(ASE->getIdx())) {
          NonLinearFound = true;
          return false;
        }
        return true;
      }
    };

    std::string inductionVar = getInductionVariableName(Loop);
    IndexingVisitor visitor(inductionVar);
    visitor.TraverseStmt(const_cast<Stmt*>(Loop->getBody()));
    return visitor.NonLinearFound;
  }

  bool hasMemoryAliasing(const ForStmt *Loop) {
    class AliasVisitor : public RecursiveASTVisitor<AliasVisitor> {
    public:
      bool PotentialAlias = false;
      std::set<std::string> WrittenArrays;
      std::set<std::string> ReadArrays;

      bool VisitArraySubscriptExpr(ArraySubscriptExpr *ASE) {
        if (const auto *DRE = dyn_cast<DeclRefExpr>(ASE->getBase()->IgnoreImpCasts())) {
          std::string arrayName = DRE->getDecl()->getNameAsString();
          
          if (isLValue(ASE)) {
            WrittenArrays.insert(arrayName);
          } else {
            ReadArrays.insert(arrayName);
          }
        }
        return true;
      }

    private:
      bool isLValue(const Expr *E) {
        return E->isLValue();
      }
    };

    AliasVisitor visitor;
    visitor.TraverseStmt(const_cast<Stmt*>(Loop->getBody()));

    for (const std::string &written : visitor.WrittenArrays) {
      for (const std::string &read : visitor.ReadArrays) {
        if (written == read) {
          continue;
        }
        visitor.PotentialAlias = true;
        break;
      }
    }

    return visitor.PotentialAlias;
  }

  bool hasUnpredictableBranches(const ForStmt *Loop) {
    class BranchVisitor : public RecursiveASTVisitor<BranchVisitor> {
    public:
      bool UnpredictableBranchFound = false;

      bool VisitIfStmt(IfStmt *IS) {
        if (const auto *Cond = IS->getCond()) {
          if (!isSimpleCondition(Cond)) {
            UnpredictableBranchFound = true;
            return false;
          }
        }
        return true;
      }

    private:
      bool isSimpleCondition(const Expr *E) {
        if (isa<DeclRefExpr>(E) || isa<IntegerLiteral>(E)) {
          return true;
        }
        if (const auto *BO = dyn_cast<BinaryOperator>(E)) {
          return isSimpleCondition(BO->getLHS()) && isSimpleCondition(BO->getRHS());
        }
        return false;
      }
    };

    BranchVisitor visitor;
    visitor.TraverseStmt(const_cast<Stmt*>(Loop->getBody()));
    return visitor.UnpredictableBranchFound;
  }

  bool hasEarlyExits(const ForStmt *Loop) {
    class ExitVisitor : public RecursiveASTVisitor<ExitVisitor> {
    public:
      bool EarlyExitFound = false;

      bool VisitBreakStmt(BreakStmt *BS) {
        EarlyExitFound = true;
        return false;
      }

      bool VisitReturnStmt(ReturnStmt *RS) {
        EarlyExitFound = true;
        return false;
      }

      bool VisitGotoStmt(GotoStmt *GS) {
        EarlyExitFound = true;
        return false;
      }
    };

    ExitVisitor visitor;
    visitor.TraverseStmt(const_cast<Stmt*>(Loop->getBody()));
    return visitor.EarlyExitFound;
  }

  bool hasSideEffects(const ForStmt *Loop) {
    class SideEffectVisitor : public RecursiveASTVisitor<SideEffectVisitor> {
    public:
      bool SideEffectFound = false;

      bool VisitCallExpr(CallExpr *CE) {
        if (const auto *FD = CE->getDirectCallee()) {
          if (FD->getNameAsString() == "printf" || 
              FD->getNameAsString() == "scanf" ||
              FD->getNameAsString() == "malloc" ||
              FD->getNameAsString() == "free") {
            SideEffectFound = true;
            return false;
          }
        }
        return true;
      }

      bool VisitDeclRefExpr(DeclRefExpr *DRE) {
        if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
          if (VD->getType().isVolatileQualified()) {
            SideEffectFound = true;
            return false;
          }
        }
        return true;
      }
    };

    SideEffectVisitor visitor;
    visitor.TraverseStmt(const_cast<Stmt*>(Loop->getBody()));
    return visitor.SideEffectFound;
  }

  std::string getInductionVariableName(const ForStmt *Loop) {
    if (const auto *DS = dyn_cast<DeclStmt>(Loop->getInit())) {
      if (DS->isSingleDecl()) {
        if (const auto *VD = dyn_cast<VarDecl>(DS->getSingleDecl())) {
          return VD->getNameAsString();
        }
      }
    }
    return "";
  }

  void emitDiagnostic(const ForStmt *Loop, VectorizationBlocker blocker) {
    SourceLocation Loc = Loop->getForLoc();
    
    switch (blocker) {
      case VectorizationBlocker::NonCanonicalLoop:
        Diags.Report(Loc, DiagIDNonVectorizable) << "loop structure is not canonical";
        Diags.Report(Loc, DiagIDNote) << "vectorization requires 'for(int i=0; i<N; ++i)' pattern";
        break;
        
      case VectorizationBlocker::NonLinearIndex:
        Diags.Report(Loc, DiagIDNonVectorizable) << "index expression is non-linear";
        Diags.Report(Loc, DiagIDNote) << "vectorization requires affine indexing like a[i + constant]";
        break;
        
      case VectorizationBlocker::MemoryAlias:
        Diags.Report(Loc, DiagIDNonVectorizable) << "potential memory alias between pointers";
        Diags.Report(Loc, DiagIDNote) << "cannot guarantee safe parallel writes";
        break;
        
      case VectorizationBlocker::UnpredictableBranch:
        Diags.Report(Loc, DiagIDNonVectorizable) << "unpredictable conditional branch";
        Diags.Report(Loc, DiagIDNote) << "complex conditions prevent vectorization";
        break;
        
      case VectorizationBlocker::EarlyExit:
        Diags.Report(Loc, DiagIDNonVectorizable) << "early exit statements found";
        Diags.Report(Loc, DiagIDNote) << "break/return/goto statements prevent vectorization";
        break;
        
      case VectorizationBlocker::SideEffects:
        Diags.Report(Loc, DiagIDNonVectorizable) << "side effects detected";
        Diags.Report(Loc, DiagIDNote) << "I/O operations or volatile access prevent vectorization";
        break;
        
      default:
        break;
    }
  }
};

class AutoVecVisitor : public RecursiveASTVisitor<AutoVecVisitor> {
private:
  ASTContext &Context;
  LoopVectorizabilityAnalyzer Analyzer;

public:
  AutoVecVisitor(ASTContext &Ctx) : Context(Ctx), Analyzer(Ctx, Ctx.getDiagnostics()) {}

  bool VisitForStmt(ForStmt *FS) {
    Analyzer.analyzeLoop(FS);
    return true;
  }
};

class AutoVecConsumer : public ASTConsumer {
private:
  AutoVecVisitor Visitor;

public:
  AutoVecConsumer(ASTContext &Context) : Visitor(Context) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

class AutoVecAction : public PluginASTAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef) override {
    return std::make_unique<AutoVecConsumer>(CI.getASTContext());
  }

  bool ParseArgs(const CompilerInstance &CI, const std::vector<std::string> &args) override {
    return true;
  }
};

}

static FrontendPluginRegistry::Add<AutoVecAction>
X("auto-vec", "C++ Auto Vectorizer Loop Analyzer");
