#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>
#include <unordered_map>
#include <cmath>
#include <fstream>

using namespace clang;
using namespace llvm;

namespace {

struct LoopFeatures {
    int tripCount;
    int bodyComplexity;
    int memoryAccesses;
    int computeIntensity;
    int branchCount;
    int callCount;
    double cacheLocality;
    double parallelizability;
    std::string targetArch;
};

class NeuralNetworkCostModel {
private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    bool modelLoaded;

public:
    NeuralNetworkCostModel() : modelLoaded(false) {
        loadPretrainedModel();
    }

    struct PredictionResult {
        double speedup;
        double confidence;
        int optimalVectorWidth;
        std::vector<std::string> recommendations;
    };

    PredictionResult predictVectorization(const LoopFeatures &features) {
        if (!modelLoaded) {
            return fallbackPrediction(features);
        }

        std::vector<double> input = featuresToVector(features);
        std::vector<double> output = forwardPass(input);
        
        PredictionResult result;
        result.speedup = std::max(1.0, output[0] * 20.0);
        result.confidence = sigmoid(output[1]);
        result.optimalVectorWidth = selectVectorWidth(features, output[2]);
        result.recommendations = generateRecommendations(features, output);
        
        return result;
    }

private:
    void loadPretrainedModel() {
        weights.resize(3);
        weights[0] = {0.8, -0.2, 0.6, 0.4, -0.3, 0.1, 0.7, 0.5, 0.2};
        weights[1] = {0.3, 0.7, -0.1, 0.4, 0.6, 0.2, -0.2, 0.8, 0.1};
        weights[2] = {-0.1, 0.5, 0.8, 0.3, 0.2, 0.7, 0.4, -0.3, 0.6};
        
        biases = {0.1, -0.2, 0.05};
        modelLoaded = true;
    }

    std::vector<double> featuresToVector(const LoopFeatures &features) {
        std::vector<double> input;
        input.push_back(std::log(std::max(1, features.tripCount)) / 10.0);
        input.push_back(features.bodyComplexity / 10.0);
        input.push_back(features.memoryAccesses / 5.0);
        input.push_back(features.computeIntensity / 8.0);
        input.push_back(features.branchCount / 3.0);
        input.push_back(features.callCount / 2.0);
        input.push_back(features.cacheLocality);
        input.push_back(features.parallelizability);
        input.push_back(features.targetArch == "x86_64" ? 1.0 : 0.0);
        return input;
    }

    std::vector<double> forwardPass(const std::vector<double> &input) {
        std::vector<double> output(3);
        for (size_t i = 0; i < 3; ++i) {
            output[i] = biases[i];
            for (size_t j = 0; j < input.size(); ++j) {
                output[i] += input[j] * weights[i][j];
            }
            output[i] = tanh(output[i]);
        }
        return output;
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    int selectVectorWidth(const LoopFeatures &features, double networkOutput) {
        double normalized = sigmoid(networkOutput);
        if (features.targetArch == "x86_64") {
            if (normalized > 0.8) return 16;
            if (normalized > 0.6) return 8;
            if (normalized > 0.4) return 4;
            return 2;
        }
        return std::min(8, (int)(normalized * 8) + 2);
    }

    std::vector<std::string> generateRecommendations(const LoopFeatures &features, 
                                                    const std::vector<double> &output) {
        std::vector<std::string> recommendations;
        
        if (features.branchCount > 0) {
            recommendations.push_back("Consider branch reduction or predication");
        }
        if (features.cacheLocality < 0.7) {
            recommendations.push_back("Improve memory access patterns for better cache usage");
        }
        if (features.callCount > 0) {
            recommendations.push_back("Function calls may prevent vectorization - consider inlining");
        }
        if (output[0] < 0.3) {
            recommendations.push_back("Loop may benefit from manual SIMD intrinsics");
        }
        
        return recommendations;
    }

    PredictionResult fallbackPrediction(const LoopFeatures &features) {
        PredictionResult result;
        result.speedup = 2.0 + features.parallelizability * 2.0;
        result.confidence = 0.6;
        result.optimalVectorWidth = 4;
        result.recommendations.push_back("Using fallback cost model - train ML model for better predictions");
        return result;
    }
};

class PerformanceProfiling {
private:
    std::unordered_map<std::string, std::vector<double>> benchmarkData;

public:
    void addBenchmarkResult(const std::string &loopSignature, double actualSpeedup) {
        benchmarkData[loopSignature].push_back(actualSpeedup);
    }

    double getHistoricalSpeedup(const std::string &loopSignature) {
        auto it = benchmarkData.find(loopSignature);
        if (it == benchmarkData.end() || it->second.empty()) {
            return 1.0;
        }
        
        double sum = 0.0;
        for (double speedup : it->second) {
            sum += speedup;
        }
        return sum / it->second.size();
    }

    void exportBenchmarkData(const std::string &filename) {
        std::ofstream file(filename);
        file << "LoopSignature,AverageSpeedup,SampleCount\n";
        
        for (const auto &entry : benchmarkData) {
            double avg = getHistoricalSpeedup(entry.first);
            file << entry.first << "," << avg << "," << entry.second.size() << "\n";
        }
        file.close();
    }
};

class AdvancedArchitectureAnalyzer {
private:
    std::unordered_map<std::string, ArchConfig> archConfigs;

    struct ArchConfig {
        int maxVectorWidth;
        int l1CacheSize;
        int l2CacheSize;
        double memoryBandwidth;
        int simdUnits;
        std::vector<std::string> supportedInstructions;
    };

public:
    AdvancedArchitectureAnalyzer() {
        initializeArchitectures();
    }

    struct ArchOptimization {
        int recommendedVectorWidth;
        std::vector<std::string> optimizationHints;
        double expectedBandwidthUtilization;
        std::string targetInstructionSet;
    };

    ArchOptimization analyzeForArchitecture(const std::string &targetArch, 
                                          const LoopFeatures &features) {
        ArchOptimization result;
        
        if (archConfigs.count(targetArch) == 0) {
            result.recommendedVectorWidth = 4;
            result.optimizationHints.push_back("Unknown architecture - using generic settings");
            return result;
        }

        const ArchConfig &config = archConfigs[targetArch];
        
        result.recommendedVectorWidth = selectOptimalWidth(config, features);
        result.optimizationHints = generateArchSpecificHints(config, features);
        result.expectedBandwidthUtilization = calculateBandwidthUsage(config, features);
        result.targetInstructionSet = selectInstructionSet(config, features);
        
        return result;
    }

private:
    void initializeArchitectures() {
        archConfigs["x86_64"] = {
            .maxVectorWidth = 16,
            .l1CacheSize = 32768,
            .l2CacheSize = 262144,
            .memoryBandwidth = 25.6,
            .simdUnits = 2,
            .supportedInstructions = {"AVX2", "AVX512", "SSE4.2"}
        };
        
        archConfigs["aarch64"] = {
            .maxVectorWidth = 16,
            .l1CacheSize = 65536,
            .l2CacheSize = 524288,
            .memoryBandwidth = 34.1,
            .simdUnits = 2,
            .supportedInstructions = {"NEON", "SVE"}
        };
        
        archConfigs["riscv64"] = {
            .maxVectorWidth = 8,
            .l1CacheSize = 32768,
            .l2CacheSize = 131072,
            .memoryBandwidth = 12.8,
            .simdUnits = 1,
            .supportedInstructions = {"RVV"}
        };
    }

    int selectOptimalWidth(const ArchConfig &config, const LoopFeatures &features) {
        int maxPossible = config.maxVectorWidth;
        
        if (features.memoryAccesses > 8) {
            maxPossible = std::min(maxPossible, 8);
        }
        if (features.branchCount > 0) {
            maxPossible = std::min(maxPossible, 4);
        }
        
        return std::max(2, maxPossible);
    }

    std::vector<std::string> generateArchSpecificHints(const ArchConfig &config, 
                                                      const LoopFeatures &features) {
        std::vector<std::string> hints;
        
        if (features.memoryAccesses * 64 > config.l1CacheSize / 4) {
            hints.push_back("Consider loop tiling for better L1 cache utilization");
        }
        
        if (config.simdUnits > 1 && features.parallelizability > 0.8) {
            hints.push_back("Loop can utilize multiple SIMD units simultaneously");
        }
        
        return hints;
    }

    double calculateBandwidthUsage(const ArchConfig &config, const LoopFeatures &features) {
        double bytesPerIteration = features.memoryAccesses * 8.0;
        double iterationsPerSecond = 2.5e9 / std::max(1, features.bodyComplexity);
        double bandwidthNeeded = (bytesPerIteration * iterationsPerSecond) / 1e9;
        
        return std::min(1.0, bandwidthNeeded / config.memoryBandwidth);
    }

    std::string selectInstructionSet(const ArchConfig &config, const LoopFeatures &features) {
        if (features.computeIntensity > 6) {
            for (const std::string &instr : config.supportedInstructions) {
                if (instr == "AVX512" || instr == "SVE") {
                    return instr;
                }
            }
        }
        
        return config.supportedInstructions.empty() ? "generic" : config.supportedInstructions[0];
    }
};

}
