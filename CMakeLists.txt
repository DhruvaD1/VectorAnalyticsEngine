cmake_minimum_required(VERSION 3.20)
project(AutoVecPlugin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

include_directories(${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
include_directories(include)

set(BASIC_SOURCES
    lib/AutoVecPlugin.cpp
)

set(ADVANCED_SOURCES
    lib/AdvancedAutoVecPlugin.cpp
    lib/MLCostModel.cpp
    lib/GPUOffloadingAnalyzer.cpp
    lib/AdvancedLoopOptimizations.cpp
    lib/AdvancedFrontendOptimizations.cpp
)

add_library(AutoVecPlugin MODULE ${BASIC_SOURCES})
add_library(AdvancedAutoVecPlugin MODULE ${ADVANCED_SOURCES})

target_link_libraries(AutoVecPlugin
    PRIVATE
    ${LLVM_LIBRARIES}
    ${CLANG_LIBRARIES}
)

target_link_libraries(AdvancedAutoVecPlugin
    PRIVATE
    ${LLVM_LIBRARIES}
    ${CLANG_LIBRARIES}
)

llvm_map_components_to_libnames(llvm_libs support core irreader analysis transformutils)
target_link_libraries(AutoVecPlugin PRIVATE ${llvm_libs})
target_link_libraries(AdvancedAutoVecPlugin PRIVATE ${llvm_libs})

target_compile_features(AutoVecPlugin PRIVATE cxx_range_for cxx_auto_type)

if(APPLE)
    set_target_properties(AutoVecPlugin PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
    )
endif()
