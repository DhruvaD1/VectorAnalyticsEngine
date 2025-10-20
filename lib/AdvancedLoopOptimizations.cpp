#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <queue>
#include <set>

using namespace clang;
using namespace llvm;

namespace {

struct LoopNest {
    std::vector<ForStmt*> loops;
    std::vector<std::string> inductionVars;
    std::vector<int> tripCounts;
    int depth;
    bool isPerfectNest;
};

class LoopFusionAnalyzer {
private:
    ASTContext &Context;
    std::vector<LoopNest> candidateLoops;
    
public:
    LoopFusionAnalyzer(ASTContext &Ctx) : Context(Ctx) {}
    
    struct FusionCandidate {
        std::vector<ForStmt*> loops;
        std::string fusionType;
        double expectedSpeedup;
        std::vector<std::string> transformations;
        bool dataLocalityBenefit;
        std::string fusedLoopCode;
    };
    
    std::vector<FusionCandidate> analyzeFusionOpportunities(const std::vector<ForStmt*> &loops) {
        std::vector<FusionCandidate> candidates;
        
        for (size_t i = 0; i < loops.size(); ++i) {
            for (size_t j = i + 1; j < loops.size(); ++j) {
                if (canFuseLoops(loops[i], loops[j])) {
                    FusionCandidate candidate;
                    candidate.loops = {loops[i], loops[j]};
                    candidate.fusionType = determineFusionType(loops[i], loops[j]);
                    candidate.expectedSpeedup = estimateFusionBenefit(loops[i], loops[j]);
                    candidate.transformations = generateFusionTransformations(loops[i], loops[j]);
                    candidate.dataLocalityBenefit = analyzeDataLocality(loops[i], loops[j]);
                    candidate.fusedLoopCode = generateFusedLoop(loops[i], loops[j]);
                    candidates.push_back(candidate);
                }
            }
        }
        
        std::sort(candidates.begin(), candidates.end(),
                 [](const FusionCandidate &a, const FusionCandidate &b) {
                     return a.expectedSpeedup > b.expectedSpeedup;
                 });
        
        return candidates;
    }
    
private:
    bool canFuseLoops(ForStmt *loop1, ForStmt *loop2) {
        if (!haveSameBounds(loop1, loop2)) return false;
        if (hasDataDependencyBetween(loop1, loop2)) return false;
        if (!areAdjacent(loop1, loop2)) return false;
        return true;
    }
    
    bool haveSameBounds(ForStmt *loop1, ForStmt *loop2) {
        std::string bounds1 = getBoundsSignature(loop1);
        std::string bounds2 = getBoundsSignature(loop2);
        return bounds1 == bounds2;
    }
    
    std::string getBoundsSignature(ForStmt *loop) {
        std::string signature;
        llvm::raw_string_ostream OS(signature);
        
        if (loop->getInit()) {
            loop->getInit()->printPretty(OS, nullptr, PrintingPolicy(LangOptions()));
        }
        OS << ";";
        if (loop->getCond()) {
            loop->getCond()->printPretty(OS, nullptr, PrintingPolicy(LangOptions()));
        }
        OS << ";";
        if (loop->getInc()) {
            loop->getInc()->printPretty(OS, nullptr, PrintingPolicy(LangOptions()));
        }
        
        return OS.str();
    }
    
    bool hasDataDependencyBetween(ForStmt *loop1, ForStmt *loop2) {
        std::set<std::string> writes1 = getWrittenArrays(loop1);
        std::set<std::string> reads2 = getReadArrays(loop2);
        std::set<std::string> writes2 = getWrittenArrays(loop2);
        std::set<std::string> reads1 = getReadArrays(loop1);
        
        for (const std::string &write : writes1) {
            if (reads2.count(write) || writes2.count(write)) {
                return true;
            }
        }
        
        for (const std::string &write : writes2) {
            if (reads1.count(write)) {
                return true;
            }
        }
        
        return false;
    }
    
    std::set<std::string> getWrittenArrays(ForStmt *loop) {
        class WriteVisitor : public RecursiveASTVisitor<WriteVisitor> {
        public:
            std::set<std::string> arrays;
            
            bool VisitArraySubscriptExpr(ArraySubscriptExpr *ASE) {
                if (ASE->isLValue()) {
                    if (const auto *DRE = dyn_cast<DeclRefExpr>(ASE->getBase()->IgnoreImpCasts())) {
                        arrays.insert(DRE->getDecl()->getNameAsString());
                    }
                }
                return true;
            }
        };
        
        WriteVisitor visitor;
        visitor.TraverseStmt(loop->getBody());
        return visitor.arrays;
    }
    
    std::set<std::string> getReadArrays(ForStmt *loop) {
        class ReadVisitor : public RecursiveASTVisitor<ReadVisitor> {
        public:
            std::set<std::string> arrays;
            
            bool VisitArraySubscriptExpr(ArraySubscriptExpr *ASE) {
                if (!ASE->isLValue()) {
                    if (const auto *DRE = dyn_cast<DeclRefExpr>(ASE->getBase()->IgnoreImpCasts())) {
                        arrays.insert(DRE->getDecl()->getNameAsString());
                    }
                }
                return true;
            }
        };
        
        ReadVisitor visitor;
        visitor.TraverseStmt(loop->getBody());
        return visitor.arrays;
    }
    
    bool areAdjacent(ForStmt *loop1, ForStmt *loop2) {
        return true; // Simplified - would need CFG analysis
    }
    
    std::string determineFusionType(ForStmt *loop1, ForStmt *loop2) {
        std::set<std::string> arrays1 = getWrittenArrays(loop1);
        std::set<std::string> arrays2 = getWrittenArrays(loop2);
        
        std::set<std::string> commonArrays;
        std::set_intersection(arrays1.begin(), arrays1.end(),
                            arrays2.begin(), arrays2.end(),
                            std::inserter(commonArrays, commonArrays.begin()));
        
        if (commonArrays.empty()) {
            return "independent_fusion";
        } else {
            return "dependent_fusion";
        }
    }
    
    double estimateFusionBenefit(ForStmt *loop1, ForStmt *loop2) {
        int memAccesses1 = countMemoryAccesses(loop1);
        int memAccesses2 = countMemoryAccesses(loop2);
        
        int totalAccesses = memAccesses1 + memAccesses2;
        int fusedAccesses = totalAccesses - countSharedArrays(loop1, loop2);
        
        double memoryBenefit = (double)totalAccesses / fusedAccesses;
        double cacheBenefit = 1.2; // Assumed cache locality improvement
        
        return memoryBenefit * cacheBenefit;
    }
    
    int countMemoryAccesses(ForStmt *loop) {
        class MemoryVisitor : public RecursiveASTVisitor<MemoryVisitor> {
        public:
            int count = 0;
            bool VisitArraySubscriptExpr(ArraySubscriptExpr *ASE) {
                count++;
                return true;
            }
        };
        
        MemoryVisitor visitor;
        visitor.TraverseStmt(loop->getBody());
        return visitor.count;
    }
    
    int countSharedArrays(ForStmt *loop1, ForStmt *loop2) {
        std::set<std::string> arrays1 = getWrittenArrays(loop1);
        arrays1.merge(getReadArrays(loop1));
        
        std::set<std::string> arrays2 = getWrittenArrays(loop2);
        arrays2.merge(getReadArrays(loop2));
        
        std::set<std::string> shared;
        std::set_intersection(arrays1.begin(), arrays1.end(),
                            arrays2.begin(), arrays2.end(),
                            std::inserter(shared, shared.begin()));
        
        return shared.size();
    }
    
    std::vector<std::string> generateFusionTransformations(ForStmt *loop1, ForStmt *loop2) {
        std::vector<std::string> transformations;
        transformations.push_back("Combine loop headers");
        transformations.push_back("Merge loop bodies");
        transformations.push_back("Optimize shared memory accesses");
        return transformations;
    }
    
    bool analyzeDataLocality(ForStmt *loop1, ForStmt *loop2) {
        return countSharedArrays(loop1, loop2) > 0;
    }
    
    std::string generateFusedLoop(ForStmt *loop1, ForStmt *loop2) {
        std::string fusedCode = "// Fused loop\n";
        
        std::string header1;
        llvm::raw_string_ostream OS1(header1);
        loop1->printPretty(OS1, nullptr, PrintingPolicy(LangOptions()));
        
        size_t bodyStart = header1.find('{');
        if (bodyStart != std::string::npos) {
            fusedCode += header1.substr(0, bodyStart + 1) + "\n";
        }
        
        std::string body1;
        llvm::raw_string_ostream OSB1(body1);
        loop1->getBody()->printPretty(OSB1, nullptr, PrintingPolicy(LangOptions()));
        
        std::string body2;
        llvm::raw_string_ostream OSB2(body2);
        loop2->getBody()->printPretty(OSB2, nullptr, PrintingPolicy(LangOptions()));
        
        fusedCode += "    // From first loop:\n" + body1 + "\n";
        fusedCode += "    // From second loop:\n" + body2 + "\n";
        fusedCode += "}\n";
        
        return fusedCode;
    }
};

class LoopTilingOptimizer {
private:
    ASTContext &Context;
    
public:
    LoopTilingOptimizer(ASTContext &Ctx) : Context(Ctx) {}
    
    struct TilingStrategy {
        std::vector<int> tileSizes;
        std::string tilingPattern;
        double cacheEfficiency;
        std::string transformedCode;
        std::vector<std::string> optimizations;
    };
    
    TilingStrategy analyzeTilingOpportunity(ForStmt *loop) {
        TilingStrategy strategy;
        
        LoopNest nest = analyzeLoopNest(loop);
        
        if (nest.depth >= 2) {
            strategy.tileSizes = calculateOptimalTileSizes(nest);
            strategy.tilingPattern = selectTilingPattern(nest);
            strategy.cacheEfficiency = estimateCacheEfficiency(nest, strategy.tileSizes);
            strategy.transformedCode = generateTiledLoop(loop, strategy.tileSizes);
            strategy.optimizations = generateTilingOptimizations(nest);
        }
        
        return strategy;
    }
    
private:
    LoopNest analyzeLoopNest(ForStmt *loop) {
        LoopNest nest;
        nest.depth = 0;
        
        ForStmt *current = loop;
        while (current) {
            nest.loops.push_back(current);
            nest.inductionVars.push_back(getInductionVariable(current));
            nest.tripCounts.push_back(estimateTripCount(current));
            nest.depth++;
            
            // Look for nested loop
            if (const auto *compound = dyn_cast<CompoundStmt>(current->getBody())) {
                current = nullptr;
                for (const auto *stmt : compound->body()) {
                    if (const auto *nestedLoop = dyn_cast<ForStmt>(stmt)) {
                        current = const_cast<ForStmt*>(nestedLoop);
                        break;
                    }
                }
            } else if (const auto *nestedLoop = dyn_cast<ForStmt>(current->getBody())) {
                current = const_cast<ForStmt*>(nestedLoop);
            } else {
                break;
            }
        }
        
        nest.isPerfectNest = (nest.depth > 1);
        return nest;
    }
    
    std::string getInductionVariable(ForStmt *loop) {
        if (const auto *DS = dyn_cast<DeclStmt>(loop->getInit())) {
            if (DS->isSingleDecl()) {
                if (const auto *VD = dyn_cast<VarDecl>(DS->getSingleDecl())) {
                    return VD->getNameAsString();
                }
            }
        }
        return "i";
    }
    
    int estimateTripCount(ForStmt *loop) {
        if (const auto *BO = dyn_cast<BinaryOperator>(loop->getCond())) {
            if (const auto *RHS = dyn_cast<IntegerLiteral>(BO->getRHS())) {
                return (int)RHS->getValue().getSExtValue();
            }
        }
        return 100;
    }
    
    std::vector<int> calculateOptimalTileSizes(const LoopNest &nest) {
        std::vector<int> tileSizes;
        
        const int L1_CACHE_SIZE = 32768;  // 32KB L1 cache
        const int ELEMENT_SIZE = 8;       // 8 bytes per double
        
        if (nest.depth >= 2) {
            int totalElements = 1;
            for (int tripCount : nest.tripCounts) {
                totalElements *= std::min(tripCount, 1000);
            }
            
            int maxTileElements = L1_CACHE_SIZE / ELEMENT_SIZE / 2;
            
            if (nest.depth == 2) {
                int tileSize = (int)std::sqrt(maxTileElements);
                tileSizes = {std::min(tileSize, nest.tripCounts[0]), 
                           std::min(tileSize, nest.tripCounts[1])};
            } else if (nest.depth == 3) {
                int tileSize = (int)std::cbrt(maxTileElements);
                tileSizes = {std::min(tileSize, nest.tripCounts[0]),
                           std::min(tileSize, nest.tripCounts[1]),
                           std::min(tileSize, nest.tripCounts[2])};
            }
        }
        
        return tileSizes;
    }
    
    std::string selectTilingPattern(const LoopNest &nest) {
        if (nest.depth == 2) {
            return "rectangular_tiling";
        } else if (nest.depth == 3) {
            return "cubic_tiling";
        } else if (nest.depth > 3) {
            return "hierarchical_tiling";
        }
        return "no_tiling";
    }
    
    double estimateCacheEfficiency(const LoopNest &nest, const std::vector<int> &tileSizes) {
        if (tileSizes.empty()) return 1.0;
        
        long long originalAccesses = 1;
        for (int tripCount : nest.tripCounts) {
            originalAccesses *= tripCount;
        }
        
        long long tiledAccesses = 1;
        for (size_t i = 0; i < tileSizes.size() && i < nest.tripCounts.size(); ++i) {
            int numTiles = (nest.tripCounts[i] + tileSizes[i] - 1) / tileSizes[i];
            tiledAccesses *= numTiles;
        }
        
        double cacheReuse = (double)originalAccesses / tiledAccesses;
        return std::min(10.0, cacheReuse); // Cap at 10x improvement
    }
    
    std::string generateTiledLoop(ForStmt *loop, const std::vector<int> &tileSizes) {
        if (tileSizes.empty()) return "// No tiling applied\n";
        
        std::string tiledCode = "// Tiled loop transformation\n";
        
        LoopNest nest = analyzeLoopNest(loop);
        
        if (nest.depth == 2 && tileSizes.size() >= 2) {
            tiledCode += "for (int ii = 0; ii < " + std::to_string(nest.tripCounts[0]) + 
                        "; ii += " + std::to_string(tileSizes[0]) + ") {\n";
            tiledCode += "    for (int jj = 0; jj < " + std::to_string(nest.tripCounts[1]) + 
                        "; jj += " + std::to_string(tileSizes[1]) + ") {\n";
            tiledCode += "        for (int i = ii; i < std::min(ii + " + std::to_string(tileSizes[0]) + 
                        ", " + std::to_string(nest.tripCounts[0]) + "); i++) {\n";
            tiledCode += "            for (int j = jj; j < std::min(jj + " + std::to_string(tileSizes[1]) + 
                        ", " + std::to_string(nest.tripCounts[1]) + "); j++) {\n";
            
            std::string originalBody;
            llvm::raw_string_ostream OS(originalBody);
            nest.loops.back()->getBody()->printPretty(OS, nullptr, PrintingPolicy(LangOptions()));
            
            tiledCode += "                " + originalBody + "\n";
            tiledCode += "            }\n";
            tiledCode += "        }\n";
            tiledCode += "    }\n";
            tiledCode += "}\n";
        }
        
        return tiledCode;
    }
    
    std::vector<std::string> generateTilingOptimizations(const LoopNest &nest) {
        std::vector<std::string> optimizations;
        
        optimizations.push_back("Cache-conscious tile sizes selected");
        optimizations.push_back("Memory access patterns optimized");
        
        if (nest.depth >= 2) {
            optimizations.push_back("Loop blocking applied for better data locality");
        }
        
        if (nest.depth >= 3) {
            optimizations.push_back("Hierarchical tiling for multi-level cache optimization");
        }
        
        return optimizations;
    }
};

class AutotuningSystem {
private:
    std::unordered_map<std::string, std::vector<double>> performanceHistory;
    
public:
    struct AutotuneConfig {
        std::vector<int> vectorWidths;
        std::vector<int> tileSizes;
        std::vector<std::string> optimizationFlags;
        int bestVectorWidth;
        std::vector<int> bestTileSizes;
        double bestPerformance;
    };
    
    AutotuneConfig generateAutotuneConfigurations(ForStmt *loop) {
        AutotuneConfig config;
        
        config.vectorWidths = {2, 4, 8, 16, 32};
        config.tileSizes = {16, 32, 64, 128, 256};
        config.optimizationFlags = {"-O2", "-O3", "-Ofast", "-march=native"};
        
        std::string loopSignature = generateLoopSignature(loop);
        
        if (performanceHistory.count(loopSignature)) {
            auto bestConfig = selectBestConfiguration(loopSignature);
            config.bestVectorWidth = bestConfig.first;
            config.bestTileSizes = {bestConfig.second};
            config.bestPerformance = getBestPerformance(loopSignature);
        } else {
            config.bestVectorWidth = 8;  // Default
            config.bestTileSizes = {64}; // Default
            config.bestPerformance = 1.0;
        }
        
        return config;
    }
    
    void recordPerformanceResult(const std::string &loopSignature, 
                               int vectorWidth, int tileSize, double performance) {
        std::string configKey = loopSignature + "_" + std::to_string(vectorWidth) + 
                              "_" + std::to_string(tileSize);
        performanceHistory[configKey].push_back(performance);
    }
    
    std::string generateBenchmarkCode(ForStmt *loop, const AutotuneConfig &config) {
        std::string benchmarkCode = R"(
// Auto-generated benchmark code
#include <chrono>
#include <vector>
#include <iostream>

double benchmark_configuration(int vector_width, int tile_size) {
    const int N = 1000;
    std::vector<double> a(N), b(N), c(N);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        a[i] = i * 0.1;
        b[i] = i * 0.2;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Original loop with optimizations applied
)";
        
        std::string loopCode;
        llvm::raw_string_ostream OS(loopCode);
        loop->printPretty(OS, nullptr, PrintingPolicy(LangOptions()));
        benchmarkCode += loopCode;
        
        benchmarkCode += R"(
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count();
}

int main() {
    std::cout << "Running autotuning benchmarks...\n";
    
)";
        
        for (int vw : config.vectorWidths) {
            for (int ts : config.tileSizes) {
                benchmarkCode += "    double time_" + std::to_string(vw) + "_" + 
                               std::to_string(ts) + " = benchmark_configuration(" + 
                               std::to_string(vw) + ", " + std::to_string(ts) + ");\n";
                benchmarkCode += "    std::cout << \"VW=" + std::to_string(vw) + 
                               ", TS=" + std::to_string(ts) + ", Time=\" << time_" + 
                               std::to_string(vw) + "_" + std::to_string(ts) + " << \"us\\n\";\n";
            }
        }
        
        benchmarkCode += "    return 0;\n}\n";
        
        return benchmarkCode;
    }
    
private:
    std::string generateLoopSignature(ForStmt *loop) {
        std::string signature;
        llvm::raw_string_ostream OS(signature);
        loop->printPretty(OS, nullptr, PrintingPolicy(LangOptions()));
        
        std::hash<std::string> hasher;
        return std::to_string(hasher(OS.str()));
    }
    
    std::pair<int, int> selectBestConfiguration(const std::string &loopSignature) {
        double bestPerf = 0.0;
        int bestVW = 8, bestTS = 64;
        
        for (const auto &entry : performanceHistory) {
            if (entry.first.find(loopSignature) != std::string::npos) {
                for (double perf : entry.second) {
                    if (perf > bestPerf) {
                        bestPerf = perf;
                        // Parse VW and TS from key
                        bestVW = 8;  // Simplified
                        bestTS = 64;
                    }
                }
            }
        }
        
        return {bestVW, bestTS};
    }
    
    double getBestPerformance(const std::string &loopSignature) {
        double best = 1.0;
        for (const auto &entry : performanceHistory) {
            if (entry.first.find(loopSignature) != std::string::npos) {
                for (double perf : entry.second) {
                    best = std::max(best, perf);
                }
            }
        }
        return best;
    }
};

}
