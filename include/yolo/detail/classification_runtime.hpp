#pragma once

#include <cstddef>
#include <vector>

#include "yolo/adapters/ultralytics.hpp"
#include "yolo/detail/tensor_utils.hpp"
#include "yolo/tasks/classification.hpp"

namespace yolo::detail
{

struct ClassificationDecodeSpec
{
    std::size_t output_index{0};
    std::size_t class_count{0};
    adapters::ultralytics::ClassificationScoreKind score_kind{
        adapters::ultralytics::ClassificationScoreKind::unknown};
};

[[nodiscard]] Result<ClassificationDecodeSpec>
classification_decode_spec_from_binding(
    const adapters::ultralytics::AdapterBindingSpec& binding);

[[nodiscard]] Result<std::vector<float>> decode_classification_scores(
    const RawOutputTensors& outputs,
    const ClassificationDecodeSpec& decode_spec);

[[nodiscard]] std::vector<Classification> postprocess_classification(
    const std::vector<float>& scores, const ClassificationOptions& options,
    const ModelSpec& spec);

}  // namespace yolo::detail
