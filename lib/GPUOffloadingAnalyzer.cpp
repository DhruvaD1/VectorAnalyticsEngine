#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>

using namespace clang;
using namespace llvm;

namespace {

enum class GPUArchitecture {
    NVIDIA_PASCAL,
    NVIDIA_VOLTA,
    NVIDIA_TURING,
    NVIDIA_AMPERE,
    NVIDIA_HOPPER,
    AMD_VEGA,
    AMD_RDNA2,
    AMD_RDNA3,
    INTEL_XEHP,
    APPLE_M1,
    UNKNOWN
};

struct GPUCharacteristics {
    int computeUnits;
    int maxThreadsPerBlock;
    int maxBlocksPerSM;
    int sharedMemorySize;
    double memoryBandwidth;
    int warpSize;
    std::vector<std::string> supportedFeatures;
};

class GPUOffloadingAnalyzer {
private:
    std::unordered_map<GPUArchitecture, GPUCharacteristics> gpuSpecs;
    
public:
    GPUOffloadingAnalyzer() {
        initializeGPUSpecs();
    }
    
    struct OffloadingRecommendation {
        bool shouldOffload;
        GPUArchitecture targetGPU;
        std::string kernelStrategy;
        int recommendedBlockSize;
        int recommendedGridSize;
        std::vector<std::string> optimizations;
        double estimatedSpeedup;
        std::string generatedKernel;
    };
    
    OffloadingRecommendation analyzeForGPU(const ForStmt *Loop, GPUArchitecture targetArch) {
        OffloadingRecommendation result;
        result.targetGPU = targetArch;
        
        LoopCharacteristics loopChar = analyzeLoop(Loop);
        
        result.shouldOffload = evaluateOffloadingFeasibility(loopChar, targetArch);
        if (!result.shouldOffload) {
            result.estimatedSpeedup = 1.0;
            return result;
        }
        
        result.kernelStrategy = selectKernelStrategy(loopChar, targetArch);
        result.recommendedBlockSize = calculateOptimalBlockSize(loopChar, targetArch);
        result.recommendedGridSize = calculateOptimalGridSize(loopChar, targetArch);
        result.optimizations = generateOptimizations(loopChar, targetArch);
        result.estimatedSpeedup = estimateGPUSpeedup(loopChar, targetArch);
        result.generatedKernel = generateCUDAKernel(Loop, loopChar, targetArch);
        
        return result;
    }

private:
    struct LoopCharacteristics {
        int tripCount;
        int memoryAccesses;
        int computeOperations;
        bool hasDataDependencies;
        bool hasIrregularAccess;
        int branchDivergence;
        double computeIntensity;
        std::vector<std::string> arrayAccesses;
    };
    
    void initializeGPUSpecs() {
        gpuSpecs[GPUArchitecture::NVIDIA_AMPERE] = {
            .computeUnits = 108,
            .maxThreadsPerBlock = 1024,
            .maxBlocksPerSM = 16,
            .sharedMemorySize = 98304,
            .memoryBandwidth = 1555.2,
            .warpSize = 32,
            .supportedFeatures = {"tensor_cores", "async_copy", "cooperative_groups"}
        };
        
        gpuSpecs[GPUArchitecture::NVIDIA_HOPPER] = {
            .computeUnits = 144,
            .maxThreadsPerBlock = 1024,
            .maxBlocksPerSM = 24,
            .sharedMemorySize = 228096,
            .memoryBandwidth = 3352.0,
            .warpSize = 32,
            .supportedFeatures = {"tensor_cores_4th_gen", "thread_block_cluster", "async_copy_bulk"}
        };
        
        gpuSpecs[GPUArchitecture::AMD_RDNA3] = {
            .computeUnits = 96,
            .maxThreadsPerBlock = 1024,
            .maxBlocksPerSM = 8,
            .sharedMemorySize = 65536,
            .memoryBandwidth = 960.0,
            .warpSize = 64,
            .supportedFeatures = {"wave64", "infinity_cache", "smart_access_memory"}
        };
    }
    
    LoopCharacteristics analyzeLoop(const ForStmt *Loop) {
        LoopCharacteristics characteristics;
        
        characteristics.tripCount = estimateTripCount(Loop);
        characteristics.computeIntensity = calculateComputeIntensity(Loop->getBody());
        characteristics.hasDataDependencies = checkDataDependencies(Loop);
        characteristics.branchDivergence = analyzeBranchDivergence(Loop->getBody());
        characteristics.arrayAccesses = extractArrayAccesses(Loop->getBody());
        
        return characteristics;
    }
    
    int estimateTripCount(const ForStmt *Loop) {
        if (const auto *BO = dyn_cast<BinaryOperator>(Loop->getCond())) {
            if (const auto *RHS = dyn_cast<IntegerLiteral>(BO->getRHS())) {
                return (int)RHS->getValue().getSExtValue();
            }
        }
        return 1000; // Default estimate
    }
    
    double calculateComputeIntensity(const Stmt *Body) {
        class ComputeVisitor : public RecursiveASTVisitor<ComputeVisitor> {
        public:
            int operations = 0;
            int memoryOps = 0;
            
            bool VisitBinaryOperator(BinaryOperator *BO) {
                if (BO->isArithmeticOp()) operations++;
                return true;
            }
            
            bool VisitArraySubscriptExpr(ArraySubscriptExpr *ASE) {
                memoryOps++;
                return true;
            }
        };
        
        ComputeVisitor visitor;
        visitor.TraverseStmt(const_cast<Stmt*>(Body));
        
        return visitor.memoryOps > 0 ? (double)visitor.operations / visitor.memoryOps : 1.0;
    }
    
    bool checkDataDependencies(const ForStmt *Loop) {
        class DependencyVisitor : public RecursiveASTVisitor<DependencyVisitor> {
        public:
            bool hasDependency = false;
            std::vector<std::string> writeAccesses;
            std::vector<std::string> readAccesses;
            
            bool VisitArraySubscriptExpr(ArraySubscriptExpr *ASE) {
                std::string accessPattern = getAccessPattern(ASE);
                if (ASE->isLValue()) {
                    writeAccesses.push_back(accessPattern);
                } else {
                    readAccesses.push_back(accessPattern);
                }
                return true;
            }
            
        private:
            std::string getAccessPattern(ArraySubscriptExpr *ASE) {
                std::string result;
                llvm::raw_string_ostream OS(result);
                ASE->printPretty(OS, nullptr, PrintingPolicy(LangOptions()));
                return OS.str();
            }
        };
        
        DependencyVisitor visitor;
        visitor.TraverseStmt(const_cast<Stmt*>(Loop->getBody()));
        
        // Check for read-after-write dependencies
        for (const std::string &write : visitor.writeAccesses) {
            for (const std::string &read : visitor.readAccesses) {
                if (write.find("i-1") != std::string::npos || 
                    write.find("i+1") != std::string::npos) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    int analyzeBranchDivergence(const Stmt *Body) {
        class BranchVisitor : public RecursiveASTVisitor<BranchVisitor> {
        public:
            int divergentBranches = 0;
            
            bool VisitIfStmt(IfStmt *IS) {
                if (isThreadDependent(IS->getCond())) {
                    divergentBranches++;
                }
                return true;
            }
            
        private:
            bool isThreadDependent(const Expr *E) {
                if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
                    return DRE->getDecl()->getNameAsString() == "i";
                }
                return false;
            }
        };
        
        BranchVisitor visitor;
        visitor.TraverseStmt(const_cast<Stmt*>(Body));
        return visitor.divergentBranches;
    }
    
    std::vector<std::string> extractArrayAccesses(const Stmt *Body) {
        class AccessVisitor : public RecursiveASTVisitor<AccessVisitor> {
        public:
            std::vector<std::string> accesses;
            
            bool VisitArraySubscriptExpr(ArraySubscriptExpr *ASE) {
                std::string access;
                llvm::raw_string_ostream OS(access);
                ASE->printPretty(OS, nullptr, PrintingPolicy(LangOptions()));
                accesses.push_back(OS.str());
                return true;
            }
        };
        
        AccessVisitor visitor;
        visitor.TraverseStmt(const_cast<Stmt*>(Body));
        return visitor.accesses;
    }
    
    bool evaluateOffloadingFeasibility(const LoopCharacteristics &characteristics, 
                                     GPUArchitecture targetArch) {
        if (characteristics.hasDataDependencies) return false;
        if (characteristics.tripCount < 1000) return false;
        if (characteristics.computeIntensity < 1.0) return false;
        if (characteristics.branchDivergence > 2) return false;
        
        return true;
    }
    
    std::string selectKernelStrategy(const LoopCharacteristics &characteristics, 
                                   GPUArchitecture targetArch) {
        const GPUCharacteristics &specs = gpuSpecs[targetArch];
        
        if (characteristics.computeIntensity > 10.0) {
            return "compute_intensive";
        } else if (characteristics.memoryAccesses > 8) {
            return "memory_bound";
        } else if (characteristics.tripCount > 1000000) {
            return "large_scale_parallel";
        }
        
        return "standard_parallel";
    }
    
    int calculateOptimalBlockSize(const LoopCharacteristics &characteristics, 
                                GPUArchitecture targetArch) {
        const GPUCharacteristics &specs = gpuSpecs[targetArch];
        
        if (characteristics.branchDivergence > 0) {
            return std::min(256, specs.warpSize * 4);
        }
        
        if (characteristics.computeIntensity > 5.0) {
            return specs.maxThreadsPerBlock;
        }
        
        return 256; // Good default
    }
    
    int calculateOptimalGridSize(const LoopCharacteristics &characteristics, 
                               GPUArchitecture targetArch) {
        const GPUCharacteristics &specs = gpuSpecs[targetArch];
        int blockSize = calculateOptimalBlockSize(characteristics, targetArch);
        
        int minGridSize = (characteristics.tripCount + blockSize - 1) / blockSize;
        int maxGridSize = specs.computeUnits * specs.maxBlocksPerSM;
        
        return std::min(minGridSize, maxGridSize);
    }
    
    std::vector<std::string> generateOptimizations(const LoopCharacteristics &characteristics,
                                                 GPUArchitecture targetArch) {
        std::vector<std::string> optimizations;
        const GPUCharacteristics &specs = gpuSpecs[targetArch];
        
        if (characteristics.memoryAccesses > 4) {
            optimizations.push_back("Use shared memory for data reuse");
        }
        
        if (characteristics.computeIntensity > 8.0 && 
            std::find(specs.supportedFeatures.begin(), specs.supportedFeatures.end(), "tensor_cores") 
            != specs.supportedFeatures.end()) {
            optimizations.push_back("Consider tensor core utilization");
        }
        
        if (characteristics.branchDivergence > 0) {
            optimizations.push_back("Minimize warp divergence with predication");
        }
        
        optimizations.push_back("Use memory coalescing for optimal bandwidth");
        
        return optimizations;
    }
    
    double estimateGPUSpeedup(const LoopCharacteristics &characteristics, 
                            GPUArchitecture targetArch) {
        const GPUCharacteristics &specs = gpuSpecs[targetArch];
        
        double parallelism = std::min((double)characteristics.tripCount, 
                                    (double)(specs.computeUnits * specs.maxThreadsPerBlock));
        
        double baseSpeedup = parallelism / 8.0; // Assume 8-core CPU baseline
        
        // Apply efficiency factors
        if (characteristics.branchDivergence > 0) {
            baseSpeedup *= 0.7; // Penalty for divergence
        }
        
        if (characteristics.computeIntensity < 2.0) {
            baseSpeedup *= 0.6; // Memory-bound penalty
        }
        
        // GPU overhead penalty for small problems
        if (characteristics.tripCount < 10000) {
            baseSpeedup *= 0.3;
        }
        
        return std::max(1.0, baseSpeedup);
    }
    
    std::string generateCUDAKernel(const ForStmt *Loop, 
                                 const LoopCharacteristics &characteristics,
                                 GPUArchitecture targetArch) {
        std::string kernel = "// Auto-generated CUDA kernel\n";
        kernel += "__global__ void vectorized_loop_kernel(";
        
        // Extract parameters from array accesses
        std::set<std::string> arrays;
        for (const std::string &access : characteristics.arrayAccesses) {
            size_t bracket = access.find('[');
            if (bracket != std::string::npos) {
                arrays.insert(access.substr(0, bracket));
            }
        }
        
        bool first = true;
        for (const std::string &array : arrays) {
            if (!first) kernel += ", ";
            kernel += "float* " + array;
            first = false;
        }
        kernel += ", int N) {\n";
        
        kernel += "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n";
        kernel += "    if (i < N) {\n";
        
        // Generate loop body
        std::string bodyCode;
        llvm::raw_string_ostream OS(bodyCode);
        Loop->getBody()->printPretty(OS, nullptr, PrintingPolicy(LangOptions()));
        
        // Simple transformation: remove braces and adjust indentation
        std::string body = OS.str();
        if (body.front() == '{' && body.back() == '}') {
            body = body.substr(1, body.length() - 2);
        }
        
        kernel += "        " + body + "\n";
        kernel += "    }\n";
        kernel += "}\n";
        
        return kernel;
    }
};

class HeterogeneousComputeOptimizer {
private:
    GPUOffloadingAnalyzer gpuAnalyzer;
    
public:
    struct HeterogeneousStrategy {
        bool useCPU;
        bool useGPU;
        bool useFPGA;
        double cpuWorkload;
        double gpuWorkload;
        std::string schedulingStrategy;
        std::vector<std::string> optimizations;
    };
    
    HeterogeneousStrategy optimizeForHeterogeneousSystem(const ForStmt *Loop) {
        HeterogeneousStrategy strategy;
        
        auto gpuRec = gpuAnalyzer.analyzeForGPU(Loop, GPUArchitecture::NVIDIA_AMPERE);
        
        if (gpuRec.shouldOffload && gpuRec.estimatedSpeedup > 3.0) {
            strategy.useGPU = true;
            strategy.useCPU = false;
            strategy.gpuWorkload = 1.0;
            strategy.cpuWorkload = 0.0;
            strategy.schedulingStrategy = "gpu_primary";
        } else if (gpuRec.shouldOffload && gpuRec.estimatedSpeedup > 1.5) {
            strategy.useGPU = true;
            strategy.useCPU = true;
            strategy.gpuWorkload = 0.8;
            strategy.cpuWorkload = 0.2;
            strategy.schedulingStrategy = "hybrid_load_balance";
        } else {
            strategy.useGPU = false;
            strategy.useCPU = true;
            strategy.gpuWorkload = 0.0;
            strategy.cpuWorkload = 1.0;
            strategy.schedulingStrategy = "cpu_optimized";
        }
        
        return strategy;
    }
};

}
