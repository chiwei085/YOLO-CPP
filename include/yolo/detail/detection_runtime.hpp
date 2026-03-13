#pragma once

#include <vector>

#include "yolo/adapters/ultralytics.hpp"
#include "yolo/detail/tensor_utils.hpp"
#include "yolo/tasks/detection.hpp"

namespace yolo::detail
{

struct DetectionCandidate
{
    RectF bbox{};
    float score{0.0F};
    ClassId class_id{0};
};

enum class DetectionLayout
{
    xywh_class_scores_last,
    xywh_class_scores_first,
    xyxy_score_class,
};

struct DetectionDecodeSpec
{
    std::size_t output_index{0};
    DetectionLayout layout{DetectionLayout::xywh_class_scores_last};
    std::size_t proposal_count{0};
    std::size_t class_count{0};
};

[[nodiscard]] Result<DetectionDecodeSpec> detection_decode_spec_from_binding(
    const adapters::ultralytics::AdapterBindingSpec& binding);

[[nodiscard]] Result<std::vector<DetectionCandidate>> decode_detections(
    const RawOutputTensors& outputs, const DetectionDecodeSpec& spec);

[[nodiscard]] std::vector<Detection> postprocess_detections(
    std::vector<DetectionCandidate> candidates, const PreprocessRecord& record,
    const DetectionOptions& options, const ModelSpec& spec);

}  // namespace yolo::detail
