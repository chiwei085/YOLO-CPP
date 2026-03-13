#pragma once

#include <cstddef>
#include <vector>

namespace yolo
{

enum class ExecutionProvider
{
    cpu,
    cuda,
    tensorrt,
};

enum class GraphOptimizationLevel
{
    disable,
    basic,
    extended,
    all,
};

struct ExecutionProviderOptions
{
    ExecutionProvider provider{ExecutionProvider::cpu};
    int device_index{0};
};

struct SessionOptions
{
    std::size_t intra_op_threads{0};
    std::size_t inter_op_threads{0};
    bool enable_profiling{false};
    bool enable_memory_pattern{true};
    bool enable_fp16{false};
    GraphOptimizationLevel graph_optimization{GraphOptimizationLevel::all};
    std::vector<ExecutionProviderOptions> providers{
        {ExecutionProvider::cpu, 0}};
};

}  // namespace yolo
