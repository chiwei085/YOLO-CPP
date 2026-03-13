#pragma once

#include <optional>
#include <string>
#include <vector>

#include "yolo/core/image.hpp"
#include "yolo/core/tensor.hpp"
#include "yolo/core/types.hpp"

namespace yolo
{

struct InferenceMetadata
{
    TaskKind task{TaskKind::detect};
    std::optional<std::string> model_name{};
    std::optional<std::string> adapter_name{};
    std::optional<std::string> provider_name{};
    std::optional<Size2i> original_image_size{};
    std::optional<PreprocessRecord> preprocess{};
    std::vector<TensorInfo> outputs{};
    std::optional<double> latency_ms{};
};

}  // namespace yolo
